from pathlib import Path
from tqdm import tqdm
import numpy as np
import logging
import torch
import pickle
import cv2
import copy

from .base_dataset import BaseDataset
from .view import read_view
from .view import numpy_image_to_torch
from .view import torch_image_to_numpy
from ..geometry import Camera, Pose
from ...settings import DATA_PATH

logger = logging.getLogger(__name__)


CAMERAS = '''c0 OPENCV 1024 768 868.993378 866.063001 525.942323 420.042529 -0.399431 0.188924 0.000153 0.000571
c1 OPENCV 1024 768 873.382641 876.489513 529.324138 397.272397 -0.397066 0.181925 0.000176 -0.000579'''


class CMU(BaseDataset):
    default_conf = {
        'dataset_dir': 'CMU/',
        'info_dir': 'cmu_pixloc_training/',

        'train_slices': [8, 9, 10, 11, 12, 22, 23, 24, 25],
        'val_slices': [6, 13, 21],
        'train_num_per_slice': 1000,
        'val_num_per_slice': 80,

        'two_view': True,
        'min_overlap': 0.3,
        'max_overlap': 1.,
        'min_baseline': None,
        'max_baseline': None,
        'sort_by_overlap': False,

        'grayscale': False,
        'resize': None,
        'resize_by': 'max',
        'crop': None,
        'pad': None,
        'optimal_crop': True,
        'seed': 0,

        'undistort_images': True,
        'warp_PY_images': False,
        'use_rotational_homography_augmentation': False,
        'max_inplane_angle': 0,
        'max_tilt_angle': 0,

        'max_num_points3D': 512,
        'force_num_points3D': False,
    }

    def _init(self, conf):
        pass

    def get_dataset(self, split):
        assert split != 'test', 'Not supported'
        return _Dataset(self.conf, split)


class _Dataset(torch.utils.data.Dataset):
    def __init__(self, conf, split):
        self.root = Path(DATA_PATH, conf.dataset_dir)
        self.slices = conf.get(split+'_slices')
        self.conf, self.split = conf, split

        self.info = {}
        for slice_ in self.slices:
            path = Path(DATA_PATH, self.conf.info_dir, f'slice{slice_}.pkl')
            assert path.exists(), path
            with open(path, 'rb') as f:
                info = pickle.load(f)
            self.info[slice_] = {k: info[k] for k in info if 'matrix' not in k}

        self.cameras = {}
        for c in CAMERAS.split('\n'):
            data = c.split()
            name, camera_model, width, height = data[:4]
            params = np.array(data[4:], float)
            camera = Camera.from_colmap(dict(
                    model=camera_model, params=params,
                    width=int(width), height=int(height)))
            self.cameras[name] = camera

        self.sample_new_items(conf.seed)

    def sample_new_items(self, seed):
        logger.info(f'Sampling new images or pairs with seed {seed}')
        self.items = []
        for slice_ in tqdm(self.slices):
            num = self.conf[self.split+'_num_per_slice']

            if self.conf.two_view:
                path = Path(
                        DATA_PATH, self.conf.info_dir, f'slice{slice_}.pkl')
                assert path.exists(), path
                with open(path, 'rb') as f:
                    info = pickle.load(f)

                mat = info['query_overlap_matrix']
                pairs = (
                    (mat > self.conf.min_overlap)
                    & (mat <= self.conf.max_overlap))
                if self.conf.min_baseline:
                    pairs &= (info['query_to_ref_distance_matrix']
                              > self.conf.min_baseline)
                if self.conf.max_baseline:
                    pairs &= (info['query_to_ref_distance_matrix']
                              < self.conf.max_baseline)
                pairs = np.stack(np.where(pairs), -1)
                if len(pairs) > num:
                    selected = np.random.RandomState(seed).choice(
                        len(pairs), num, replace=False)
                    pairs = pairs[selected]
                pairs = [(slice_, i, j, mat[i, j]) for i, j in pairs]
                self.items.extend(pairs)
            else:
                ids = np.arange(len(self.images[slice_]))
                if len(ids) > num:
                    ids = np.random.RandomState(seed).choice(
                        ids, num, replace=False)
                ids = [(slice_, i) for i in ids]
                self.items.extend(ids)

        if self.conf.two_view and self.conf.sort_by_overlap:
            self.items.sort(key=lambda i: i[-1], reverse=True)
        else:
            np.random.RandomState(seed).shuffle(self.items)

    def _read_view(self, slice_, idx, common_p3D_idx, is_reference=False):
        prefix = 'ref' if is_reference else 'query'
        path = self.root / f'slice{slice_}/'
        path /= 'database' if is_reference else 'query'
        path /= self.info[slice_][f'{prefix}_image_names'][idx]

        camera = self.cameras[path.name.split('_')[2]]
        R, t = self.info[slice_][f'{prefix}_poses'][idx]
        T = Pose.from_Rt(R, t)
        p3D = self.info[slice_]['points3D']
        data = read_view(self.conf, path, camera, T, p3D, common_p3D_idx,
                         random=(self.split == 'train'))
        data['index'] = idx
        if self.conf.undistort_images:
            data['image'], data['camera'] = self._undistort(data['image'], data['camera'])
        if self.conf.warp_PY_images:
            data['image'], data['camera'] = self._warp_PY(data['image'], data['camera'])
        assert (tuple(data['camera'].size.numpy())
                == data['image'].shape[1:][::-1])

        if self.conf.use_rotational_homography_augmentation:
            assert self.conf.undistort_images
            inplane_angle, tilt_angle, tilt_axis = self._sample_homography_augmentation_parameters()
            if is_reference:
                T_w2cam_old = data['T_w2cam']
            data['image'], data['T_w2cam'] = self._rotational_homography_augmentation(
                data['image'],
                data['T_w2cam'],
                data['camera'],
                inplane_angle,
                tilt_angle,
                tilt_axis,
            )
            data['inplane_angle'] = inplane_angle
            data['tilt_angle'] = tilt_angle
            data['tilt_axis'] = tilt_axis

        if is_reference:
            obs_orig = self.info[slice_]['p3D_observed'][idx]
            if self.conf.crop:
                obs = self._determine_valid_projections(obs_orig, p3D, data['camera'], data['T_w2cam'])
            else:
                obs = obs_orig
            if self.conf.use_rotational_homography_augmentation:
                # If we have performed homography augmentation, we also want to filter points such that black areas are omitted, i.e. regions that are "valid" now, but were not before the augmentation was carried out.
                obs_pre_aug = self._determine_valid_projections(obs_orig, p3D, camera_old, T_w2cam_old)
                obs = np.intersect1d(obs, obs_pre_aug)
            num_diff = self.conf.max_num_points3D - len(obs)
            if num_diff < 0:
                obs = np.random.choice(obs, self.conf.max_num_points3D)
            elif num_diff > 0 and self.conf.force_num_points3D:
                add = np.random.choice(
                    np.delete(np.arange(len(p3D)), obs), num_diff)
                obs = np.r_[obs, add]
            # NOTE: Why would * operation work? p3D[obs] should be Nx3, and T_w2cam should be 4x4.
            data['points3D'] = data['T_w2cam'] * p3D[obs]
        return data

    def _determine_valid_projections(self, obs, p3D, camera, T_w2cam):
        _, valid = camera.world2image(T_w2cam*p3D[obs])
        obs = obs[valid.numpy()]
        return obs

    def _undistort(self, image, camera_intrinsics):
        image = torch_image_to_numpy(image)
        camera_intrinsics_np = copy.deepcopy(camera_intrinsics)
        camera_intrinsics_np._data = camera_intrinsics_np._data.numpy()
        K = np.zeros((3, 3))
        K[0, 0], K[1, 1] = camera_intrinsics_np.fx, camera_intrinsics_np.fy
        K[0, 2], K[1, 2] = camera_intrinsics_np.cx, camera_intrinsics_np.cy
        K[2, 2] = 1
        # TODO-G perhaps change K to a new calibration to get rid of black pixels?
        # See cv2.getOptimalNewCameraMatrix
        image_undist = cv2.undistort(image, K, camera_intrinsics_np.dist, None, K)
        camera_intrinsics_undist = copy.deepcopy(camera_intrinsics)
        camera_intrinsics_undist.dist[:] = 0
        image_undist = numpy_image_to_torch(image_undist)
        return image_undist, camera_intrinsics_undist

    def _warp_PY(self, image, camera_intrinsics):
        # Operates on undistorted images!
        image = torch_image_to_numpy(image)
        h, w = image.shape[:2]
        camera_intrinsics_np = copy.deepcopy(camera_intrinsics)
        camera_intrinsics_np._data = camera_intrinsics_np._data.numpy()
        K = np.zeros((3, 3))
        K[0, 0], K[1, 1] = camera_intrinsics_np.fx, camera_intrinsics_np.fy
        K[0, 2], K[1, 2] = camera_intrinsics_np.cx, camera_intrinsics_np.cy
        K[2, 2] = 1
        camera_intrinsics_warp = copy.deepcopy(camera_intrinsics)  # G-TODO: maybe this needs to be modified (and then the normalize step as well)

        # map_x and map_y will contain the pixel coordinates in the original image
        # corresponding to the pixels in the new image.
        map_x, map_y = np.meshgrid(np.arange(w), np.arange(h))
        # 1. normalize
        map_x, map_y = (map_x - K[0, 2])/K[0, 0], \
                (map_y - K[1, 2])/K[1, 1]
        # 2. tan-warp
        r = np.clip(np.sqrt(map_x**2 + map_y**2), a_min=1.0e-8, a_max=None)
        r = np.tan(r) / r
        map_x, map_y = r * map_x, r * map_y
        # 3. unnormalize
        map_x, map_y = K[0, 0] * map_x + K[0, 2], \
                K[1, 1] * map_y + K[1, 2]

        image_warp = cv2.remap(image, map_x, map_y, cv.INTER_LINEAR)
        image_warp = numpy_image_to_torch(image_warp)
        # TODO-G: Add mask for black pixels?
        
        return image_warp, camera_intrinsics_warp

    def __getitem__(self, idx):
        if self.conf.two_view:
            slice_, idx_q, idx_r, overlap = self.items[idx]
            obs_r = self.info[slice_]['p3D_observed'][idx_r]
            obs_q = self.info[slice_]['p3D_observed'][
                    self.info[slice_]['query_closest_indices'][idx_q]]
            common = np.array(list(set(obs_r) & set(obs_q)))

            data_r = self._read_view(slice_, idx_r, common, is_reference=True)
            data_q = self._read_view(slice_, idx_q, common)
            data = {
                'ref': data_r,
                'query': data_q,
                'overlap': overlap,
                'T_r2q_init': Pose.from_4x4mat(np.eye(4, dtype=np.float32)),
                'T_r2q_gt': data_q['T_w2cam'] @ data_r['T_w2cam'].inv(),
            }
        else:
            slice_, idx = self.items[idx]
            data = self._read_view(slice_, idx, is_reference=True)
        data['scene'] = slice_
        return data

    def __len__(self):
        return len(self.items)

    def _sample_homography_augmentation_parameters(self):
        inplane_angle = np.random.uniform(low=-self.conf.max_inplane_angle[0], high=self.conf.max_inplane_angle[1])
        tilt_angle = np.random.uniform(low=-self.conf.max_tilt_angle[0], high=self.conf.max_tilt_angle[1])
        tmp_inplane_alpha = np.random.uniform(low=0, high=2*math.pi)
        tilt_axis = np.array([np.cos(tmp_inplane_alpha), np.sin(tmp_inplane_alpha)])
        return inplane_angle, tilt_angle, tilt_axis

    def _rotational_homography_augmentation(
        self,
        image,
        T_w2cam,
        camera_intrinsics,
        inplane_angle,
        tilt_angle,
        tilt_axis,
    ):
        """
        Args:
            image: The image to augment
            T_w2cam: R, t annotation as a Pose object
            calibration_intrinsics: The camera calibration parameters. We are assuming there is no distortion.
            inplane_angle: rotate the image with the given angle
            tilt_angle: simulate rotating the camera around an axis in the principal plane with the given angle
            tilt_axis: rotation axis for tilting (should be in principal plane, and provided with x and y components only: shape (2,))
        """
        image = torch_image_to_numpy(image)
        orig_rotation_matrix, orig_translation_vector = T_w2cam.numpy()
        camera_intrinsics_np = copy.deepcopy(camera_intrinsics)
        camera_intrinsics_np._data = camera_intrinsics_np._data.numpy()

        # We are assuming there is no distortion.
        assert np.all(camera_intrinsics_np.dist == 0)

        K = np.zeros((3, 3))
        K[0, 0], K[1, 1] = camera_intrinsics_np.fx.numpy(), camera_intrinsics_np.fy.numpy()
        K[0, 2], K[1, 2] = camera_intrinsics_np.cx.numpy(), camera_intrinsics_np.cy.numpy()
        K[2, 2] = 1

        #get the center point from the intrinsic camera matrix
        cx = K[0, 2]
        cy = K[1, 2]
        # if self.radial_arctan_prewarped_images:
        # # if self.depth_regression_mode == 'cam2obj_dist':
        #     # Transform principal point, according to warping.
        #     cx, cy = radial_arctan_transform(
        #         cx.reshape((1,1)), # x
        #         cy.reshape((1,1)), # y
        #         K[0,0], # fx
        #         K[1,1], # fy
        #         K[0,2], # px
        #         K[1,2], # py
        #         self.one_based_indexing_for_prewarp,
        #         self.original_image_shape,
        #     )
        #     cx = cx.squeeze()
        #     cy = cy.squeeze()
        assert cx.shape == ()
        assert cy.shape == ()

        height, width, _ = img.shape

        # The augmentation is defined as the follow chain:
        # (1) In-plane rotation
        # (2) Tilt rotation (corresponding to homography transformation in image plane)

        # OpenCV yields a single 2x3 similarity matrix to account for both rotation and scaling (here =1):
        # Note: A positive (counter-clockwise) rotation around the z-axis, is a negative (clockwise) rotation in the image plane (which is around the negative z-axis).
        # If the provided angle is positive, the "getRotationMatrix2D" function indeed returns a matrix which yields "counter-clockwise" rotations, but in a negatively oriented coordinate system such as the image plane.
        # The same matrix would result in a clockwise rotation in a positively oriented frame (such as the image plane seen from behind).
        # Consequently, it should actually be the case that [R_inplane_2d_mat; 0 0 1] = K*R_inplane*inv(K).
        try:
            # Note, this should work!
            R_inplane_2d_mat = cv2.getRotationMatrix2D((float(cx), float(cy)), -inplane_angle, 1) # Last argument is the scaling factor of the 2x3 similarity transformation matrix.
        except:
            # While this is unexpected!
            try:
                R_inplane_2d_mat = cv2.getRotationMatrix2D((cx, cy), -inplane_angle, 1) # Last argument is the scaling factor of the 2x3 similarity transformation matrix.
            except:
                R_inplane_2d_mat = cv2.getRotationMatrix2D(np.array([cx, cy]), -inplane_angle, 1) # Last argument is the scaling factor of the 2x3 similarity transformation matrix.

        # Express in-plane rotation with 3D rotation vector / matrix. Rotation is around the z-axis in the camera coordinate system.
        inplane_rotation_vector = np.zeros((3,))
        inplane_rotation_vector[2] = inplane_angle / 180. * math.pi
        R_inplane, _ = cv2.Rodrigues(inplane_rotation_vector)

        # Express tilt rotation with 3D rotation vector / matrix. Rotation is around an axis in the principal plane z = 0.
        assert tilt_axis.shape == (2,)
        tilt_axis = np.concatenate([tilt_axis, np.zeros((1,))], axis=0) # 3D lift to the plane z = 0
        assert tilt_axis.shape == (3,)

        R_tilt, _ = cv2.Rodrigues(tilt_axis * tilt_angle / 180. * math.pi)
        # Sigma = np.diag([scale, scale, 1])
        # Note: One could have considered to undo the effects of scaling, by taking a matrix "Sigma" into account, and include Sigma in K when applying R_tilt.
        # However, it is not entirely obvious how to best handle this part, as Sigma results only in an approximative rigid transformation to start with.
        # Whether to undo Sigma or not before applying R_tilt boils down to whether the scaled or unscaled image is regarded as the observation upon which we would like to apply the tilt augmentation.
        assert K.shape == (3, 3)
        H_tilt = K @ R_tilt @ np.linalg.inv(K)

        # Combine everything into a single homography warping:
        H = H_tilt @ np.concatenate([R_inplane_2d_mat, np.array([[0, 0, 1]])], axis=0)
        assert H.shape == (3, 3)
        augmented_img = cv2.warpPerspective(img, H, (width, height))

        # Initialize the final rotation annotation, as it was before augmentation
        augmented_rotation_matrix = np.copy(orig_rotation_matrix)
        # Also initialize the final translation annotation
        augmented_translation_vector = np.copy(orig_translation_vector)
        # Also initialize the final translation annotation
        augmented_translation_vector = np.copy(orig_translation_vector)

        ##### STEP 1: Apply in-plane rotation on pose annotations #####
        augmented_rotation_matrix = np.dot(R_inplane, augmented_rotation_matrix)
        augmented_translation_vector = np.dot(augmented_translation_vector, R_inplane.T)

        ##### STEP 2: Apply tilt rotation on pose annotations #####
        augmented_rotation_matrix = np.dot(R_tilt, augmented_rotation_matrix)
        augmented_translation_vector = np.dot(augmented_translation_vector, R_tilt.T)

        augmented_T_w2cam = Pose.from_Rt(augmented_rotation_matrix, augmented_translation_vector)
        augmented_img = numpy_image_to_torch(augmented_img)
        return augmented_img, augmented_T_w2cam
