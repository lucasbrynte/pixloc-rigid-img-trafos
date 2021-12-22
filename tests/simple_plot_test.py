from pixloc.pixlib.datasets import get_dataset
from pixloc.pixlib.datasets.view import torch_image_to_numpy
import matplotlib.pyplot as plt
import numpy as np


def plotting_test():

    number_plots = 5
    for img_id in ['rhaug', 'warp', 'orig', 'undist']:
        data_conf = {
            'val_slices': ['9'],
            'val_batch_size': 1,
        }
        if img_id == 'warp':
            data_conf['undistort_images'] = True
            data_conf['warp_PY_images'] = True
        elif img_id == 'rhaug':
            data_conf['undistort_images'] = True
            data_conf['use_rotational_homography_augmentation'] = True
            data_conf['max_inplane_angle'] = 5
            data_conf['max_tilt_angle'] = 30
        elif img_id == 'undist':
            data_conf['undistort_images'] = True
        elif img_id == 'orig':
            pass
        else:
            raise ValueError()

        dataset = get_dataset('cmu')(data_conf)
        val_loader = dataset.get_data_loader('val')
        plot_idx = 0
        for data in val_loader:
            ref_img = data['ref']['image'][0]
            ref_name = data['ref']['name'][0].split('.')[0]
            ref_img = torch_image_to_numpy(ref_img).astype(np.int32)
            plt.imshow(ref_img)
            plt.savefig(f"tests/plots/{img_id}_{ref_name}.png")
            plot_idx += 1
            if plot_idx >= number_plots:
                break


if __name__ == "__main__":
    plotting_test()
