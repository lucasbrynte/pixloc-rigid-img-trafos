import logging
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Union
from omegaconf import DictConfig, OmegaConf as oc
import numpy as np
import torch
import copy
import cv2

from .feature_extractor import FeatureExtractor
from .model3d import Model3D
from .tracker import BaseTracker
from ..pixlib.geometry import Pose, Camera
from ..pixlib.datasets.view import read_image
from ..pixlib.datasets.view import numpy_image_to_torch
from ..pixlib.datasets.view import torch_image_to_numpy
from ..utils.data import Paths

logger = logging.getLogger(__name__)

CAMERAS = '''c0 OPENCV 1024 768 868.993378 866.063001 525.942323 420.042529 -0.399431 0.188924 0.000153 0.000571
c1 OPENCV 1024 768 873.382641 876.489513 529.324138 397.272397 -0.397066 0.181925 0.000176 -0.000579'''

class BaseRefiner:
    base_default_config = dict(
        layer_indices=None,
        min_matches_db=10,
        num_dbs=1,
        min_track_length=3,
        min_points_opt=10,
        point_selection='all',
        average_observations=False,
        normalize_descriptors=True,
        compute_uncertainty=True,
        undistort_images=False,
        warp_PY_images=False,
    )

    default_config = dict()
    tracker: BaseTracker = None

    def __init__(self,
                 device: torch.device,
                 optimizer: torch.nn.Module,
                 model3d: Model3D,
                 feature_extractor: FeatureExtractor,
                 paths: Paths,
                 conf: Union[DictConfig, Dict]):
        self.device = device
        self.optimizer = optimizer
        self.model3d = model3d
        self.feature_extractor = feature_extractor
        self.paths = paths

        self.cameras = {}
        for c in CAMERAS.split('\n'):
            data = c.split()
            name, camera_model, width, height = data[:4]
            params = np.array(data[4:], float)
            camera = Camera.from_colmap(dict(
                    model=camera_model, params=params,
                    width=int(width), height=int(height)))
            self.cameras[name] = camera

        self.conf = oc.merge(
            oc.create(self.base_default_config),
            oc.create(self.default_config),
            oc.create(conf))

    def log_dense(self, **kwargs):
        if self.tracker is not None:
            self.tracker.log_dense(**kwargs)

    def log_optim(self, **kwargs):
        if self.tracker is not None:
            self.tracker.log_optim_done(**kwargs)

    def refine(self, **kwargs):
        ''' Implement this in the child class'''
        raise NotImplementedError

    def refine_pose_using_features(self,
                                   features_query: List[torch.tensor],
                                   scales_query: List[float],
                                   qcamera: Camera,
                                   T_init: Pose,
                                   features_p3d: List[List[torch.Tensor]],
                                   p3dids: List[int]) -> Dict:
        """Perform the pose refinement using given dense query feature-map.
        """
        # decompose descriptors and uncertainities, normalize descriptors
        weights_ref = []
        features_ref = []
        for level in range(len(features_p3d[0])):
            feats = torch.stack([feat[level] for feat in features_p3d], dim=0)
            feats = feats.to(self.device)
            if self.conf.compute_uncertainty:
                feats, weight = feats[:, :-1], feats[:, -1:]
                weights_ref.append(weight)
            if self.conf.normalize_descriptors:
                feats = torch.nn.functional.normalize(feats, dim=1)
            assert not feats.requires_grad
            features_ref.append(feats)

        # query dense features decomposition and normalization
        features_query = [feat.to(self.device) for feat in features_query]
        if self.conf.compute_uncertainty:
            weights_query = [feat[-1:] for feat in features_query]
            features_query = [feat[:-1] for feat in features_query]
        if self.conf.normalize_descriptors:
            features_query = [torch.nn.functional.normalize(feat, dim=0)
                              for feat in features_query]

        p3d = np.stack([self.model3d.points3D[p3did].xyz for p3did in p3dids])

        T_i = T_init
        ret = {'T_init': T_init}
        # We will start with the low res feature map first
        for idx, level in enumerate(reversed(range(len(features_query)))):
            F_q, F_ref = features_query[level], features_ref[level]
            qcamera_feat = qcamera.scale(scales_query[level])

            if self.conf.compute_uncertainty:
                W_ref_query = (weights_ref[level], weights_query[level])
            else:
                W_ref_query = None

            logger.debug(f'Optimizing at level {level}.')
            opt = self.optimizer
            if isinstance(opt, (tuple, list)):
                if self.conf.layer_indices:
                    opt = opt[self.conf.layer_indices[level]]
                else:
                    opt = opt[level]
            T_opt, fail = opt.run(p3d, F_ref, F_q, T_i.to(F_q),
                                  qcamera_feat.to(F_q),
                                  W_ref_query=W_ref_query)

            self.log_optim(i=idx, T_opt=T_opt, fail=fail, level=level,
                           p3d=p3d, p3d_ids=p3dids,
                           T_init=T_init, camera=qcamera_feat)
            if fail:
                return {**ret, 'success': False}
            T_i = T_opt

        # Compute relative pose w.r.t. initilization
        T_opt = T_opt.cpu().double()
        dR, dt = (T_init.inv() @ T_opt).magnitude()
        return {
            **ret,
            'success': True,
            'T_refined': T_opt,
            'diff_R': dR.item(),
            'diff_t': dt.item(),
        }

    def refine_query_pose(self, qname: str, qcamera: Camera, T_init: Pose,
                          p3did_to_dbids: Dict[int, List],
                          multiscales: Optional[List[int]] = None) -> Dict:

        dbid_to_p3dids = self.model3d.get_dbid_to_p3dids(p3did_to_dbids)
        if multiscales is None:
            multiscales = [1]

        rnames = [self.model3d.dbs[i].name for i in dbid_to_p3dids.keys()]
        cameras_ref = [self.cameras[n.split('_')[2]] for n in rnames]
        images_ref = [read_image(self.paths.reference_images / n)
                      for n in rnames]
        image_query = read_image(self.paths.query_images / qname)
        if self.conf.undistort_images:
            tmp_ref = list(zip(*[self._undistort(im, cam)
                                 for (im, cam) in zip(images_ref, cameras_ref)]))
            images_ref = tmp_ref[0]
            cameras_ref = tmp_ref[1]
            image_query, qcamera = self._undistort(image_query, camera_query)
        if self.conf.warp_PY_images:
            if not self.conf.undistort_images:
                raise ValueError()
            tmp_ref = list(zip(*[self._warp_PY(im, cam)
                                 for (im, cam) in zip(images_ref, cameras_ref)]))
            images_ref = tmp_ref[0]
            cameras_ref = tmp_ref[1]
            image_query, qcamera = self._warp_PY(image_query, camera_query)

        for image_scale in multiscales:
            # Compute the reference observations
            # TODO: can we compute this offline before hand?
            dbid_p3did_to_feats = dict()
            for idx, dbid in enumerate(dbid_to_p3dids):
                p3dids = dbid_to_p3dids[dbid]

                features_ref_dense, scales_ref = self.dense_feature_extraction(
                        images_ref[idx], rnames[idx], image_scale)
                dbid_p3did_to_feats[dbid] = self.interp_sparse_observations(
                        features_ref_dense, cameras_ref[idx], scales_ref, dbid, p3dids)
                del features_ref_dense

            p3did_to_feat = self.aggregate_features(
                    p3did_to_dbids, dbid_p3did_to_feats)
            if self.conf.average_observations:
                p3dids = list(p3did_to_feat.keys())
                p3did_to_feat = [p3did_to_feat[p3did] for p3did in p3dids]
            else:  # duplicate the observations
                p3dids, p3did_to_feat = list(zip(*[
                    (p3did, feat) for p3did, feats in p3did_to_feat.items()
                    for feat in zip(*feats)]))

            # Compute dense query feature maps
            features_query, scales_query = self.dense_feature_extraction(
                        image_query, qname, image_scale)

            ret = self.refine_pose_using_features(features_query, scales_query,
                                                  qcamera, T_init,
                                                  p3did_to_feat, p3dids)
            if not ret['success']:
                logger.info(f"Optimization failed for query {qname}")
                break
            else:
                T_init = ret['T_refined']
        return ret

    def dense_feature_extraction(self, image: np.array, name: str,
                                 image_scale: int = 1
                                 ) -> Tuple[List[torch.Tensor], List[int]]:
        features, scales, weight = self.feature_extractor(
                image, image_scale)
        self.log_dense(name=name, image=image, image_scale=image_scale,
                       features=features, scales=scales, weight=weight)

        if self.conf.compute_uncertainty:
            assert weight is not None
            # stack them into a single tensor (makes the bookkeeping easier)
            features = [torch.cat([f, w], 0) for f, w in zip(features, weight)]

        # Filter out some layers or keep them all
        if self.conf.layer_indices is not None:
            features = [features[i] for i in self.conf.layer_indices]
            scales = [scales[i] for i in self.conf.layer_indices]

        return features, scales

    def interp_sparse_observations(self,
                                   feature_maps: List[torch.Tensor],
                                   camera: Camera,
                                   feature_scales: List[float],
                                   image_id: float,
                                   p3dids: List[int],
                                   ) -> Dict[int, torch.Tensor]:
        image = self.model3d.dbs[image_id]
        # camera = Camera.from_colmap(self.model3d.cameras[image.camera_id])
        T_w2cam = Pose.from_colmap(image)
        p3d = np.array([self.model3d.points3D[p3did].xyz for p3did in p3dids])
        p3d_cam = T_w2cam * p3d

        # interpolate sparse descriptors and store
        feature_obs = []
        masks = []
        for i, (feats, sc) in enumerate(zip(feature_maps, feature_scales)):
            p2d_feat, valid = camera.scale(sc).world2image(p3d_cam)
            opt = self.optimizer
            opt = opt[len(opt)-i-1] if isinstance(opt, (tuple, list)) else opt
            obs, mask, _ = opt.interpolator(feats, p2d_feat.to(feats))
            assert not obs.requires_grad
            feature_obs.append(obs)
            masks.append(mask & valid.to(mask))

        mask = torch.all(torch.stack(masks, dim=0), dim=0)

        # We can't stack features because they have different # of channels
        feature_obs = [[feature_obs[i][j] for i in range(len(feature_maps))]
                       for j in range(len(p3dids))]  # N x K x D

        feature_dict = {p3id: feature_obs[i]
                        for i, p3id in enumerate(p3dids) if mask[i]}

        return feature_dict

    def aggregate_features(self,
                           p3did_to_dbids: Dict,
                           dbid_p3did_to_feats: Dict,
                           ) -> Dict[int, List[torch.Tensor]]:
        """Aggregate descriptors from covisible images through averaging.
        """
        p3did_to_feat = defaultdict(list)
        for p3id, obs_dbids in p3did_to_dbids.items():
            features = []
            for obs_imgid in obs_dbids:
                if p3id not in dbid_p3did_to_feats[obs_imgid]:
                    continue
                features.append(dbid_p3did_to_feats[obs_imgid][p3id])
            if len(features) > 0:
                # list with one entry per layer, grouping all 3D observations
                for level in range(len(features[0])):
                    observation = [f[level] for f in features]
                    if self.conf.average_observations:
                        observation = torch.stack(observation, 0)
                        if self.conf.compute_uncertainty:
                            feat, w = observation[:, :-1], observation[:, -1:]
                            feat = (feat * w).sum(0) / w.sum(0)
                            observation = torch.cat([feat, w.mean(0)], -1)
                        else:
                            observation = observation.mean(0)
                    p3did_to_feat[p3id].append(observation)
        return dict(p3did_to_feat)

    # TODO-G factor out this shared code...
    # copied from cmu.py
    def _undistort(self, image, camera):
        h, w = image.shape[:2]
        camera_np = copy.deepcopy(camera)
        camera_np._data = camera_np._data.numpy()
        K = np.zeros((3, 3))
        K[0, 0], K[1, 1] = camera_np.f[0], camera_np.f[1]
        K[0, 2], K[1, 2] = camera_np.c[0], camera_np.c[1]
        K[2, 2] = 1
        # Get calibration matrix that captures the complete undistorted image
        K_new, roi = cv2.getOptimalNewCameraMatrix(K,
                                                   camera_np.dist,
                                                   (w, h),
                                                   1,  # 1: all pixels in orig. image included, 0: no black pixels included
                                                   (w, h))
        image_undist = cv2.undistort(image, K, camera_np.dist, None, K_new)
        camera_undist = copy.deepcopy(camera)
        camera_undist.dist[:] = 0
        camera_undist.f[0] = K_new[0, 0]
        camera_undist.f[1] = K_new[1, 1]
        camera_undist.c[0] = K_new[0, 2]
        camera_undist.c[1] = K_new[1, 2]
        # It is possible to crop out some "black parts" of the image by uncommenting below:
        # x, y, w, h = roi
        # image_undist = image_undist[y:y+h, x:x+w]
        # camera_undist.size[0] = w
        # camera_undist.size[1] = h
        return image_undist, camera_undist

    def _warp_PY(self, image, camera):
        # Operates on undistorted images!
        h, w = image.shape[:2]
        camera_np = copy.deepcopy(camera)
        camera_np._data = camera_np._data.numpy()
        K = np.zeros((3, 3))
        K[0, 0], K[1, 1] = camera_np.f[0], camera_np.f[1]
        K[0, 2], K[1, 2] = camera_np.c[0], camera_np.c[1]
        K[2, 2] = 1

        # Find what normalized range the pixels go to
        extreme_x = np.array([0.0, w])
        extreme_y = np.array([0.0, h])
        extreme_x, extreme_y = (extreme_x - K[0, 2])/K[0, 0], \
            (extreme_y - K[1, 2])/K[1, 1]
        # get bounding box and new camera parameters
        A, B, C, D, K_new = self.PY_bounding_box(
            extreme_x[0], extreme_x[-1], extreme_y[0], extreme_y[-1], w, h)

        # map_x and map_y will contain the pixel coordinates
        # in the original image
        # corresponding to the pixels in the new image.
        # 1. Define a grid in normalized PY-space
        map_x, map_y = np.meshgrid(
            np.linspace(A, B, w, dtype=np.float32),
            np.linspace(C, D, h, dtype=np.float32))
        # 2. tan-warp to normalized P2-space
        r = np.clip(np.sqrt(map_x**2 + map_y**2), a_min=1.0e-8, a_max=None)
        r = np.tan(r) / r
        map_x, map_y = r * map_x, r * map_y
        # 3. unnormalize
        map_x, map_y = K[0, 0] * map_x + K[0, 2], \
            K[1, 1] * map_y + K[1, 2]

        image_warp = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR)

        # fix camera parameters
        camera_warp = copy.deepcopy(camera)
        camera_warp.f[0] = K_new[0, 0]
        camera_warp.f[1] = K_new[1, 1]
        camera_warp.c[0] = K_new[0, 2]
        camera_warp.c[1] = K_new[1, 2]
        camera_warp.dist[:2] = np.nan  # specifies special arctan "distortion"

        return image_warp, camera_warp

    @staticmethod
    def PY_bounding_box(a, b, c, d, w, h):
        # if rho is map P^2 -> PY,
        # finds A,B,C,D so that rho([a,b]x[c,d]) < [A,B]x[C,D]
        # and calibration parameters that maps [A,B]x[C,D] to [0,w]x[0,h]

        # A
        # max arctan(r)/r value will be on x axis,
        # since r will be larger for other pixels on the edge
        r = np.clip(np.abs(a), a_min=1.0e-8, a_max=None)
        A = a * np.arctan(r)/r

        # B
        # max arctan(r)/r value will be on x axis,
        # since r will be larger for other pixels on the edge
        r = np.clip(np.abs(b), a_min=1.0e-8, a_max=None)
        B = b * np.arctan(r)/r

        # C
        # max arctan(r)/r value will be on y axis,
        # since r will be larger for other pixels on the edge
        r = np.clip(np.abs(c), a_min=1.0e-8, a_max=None)
        C = c * np.arctan(r)/r

        # D
        # max arctan(r)/r value will be on y axis,
        # since r will be larger for other pixels on the edge
        r = np.clip(np.abs(d), a_min=1.0e-8, a_max=None)
        D = d * np.arctan(r)/r

        # calculate calibration parameters
        K = np.zeros((3, 3))

        # TODO-G: check these (here I ignored half-pixels etc.):
        K[0, 0] = w/(B-A)
        K[0, 2] = -A * K[0, 0]
        K[1, 1] = h/(D-C)
        K[1, 2] = -C * K[1, 1]
        K[2, 2] = 1
        # earlier values that seem to have incorrect units:
        # K[0, 0] = (B-A)/w
        # K[0, 2] = 0.5*(w-(B+A)/K[0, 0])
        # K[1, 1] = (D-C)/h
        # K[1, 2] = 0.5*(h-(C+D)/K[1, 1])
        # K[2, 2] = 1

        return A, B, C, D, K
