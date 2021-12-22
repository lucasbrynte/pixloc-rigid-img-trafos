from pixloc.pixlib.datasets import get_dataset
from pixloc.pixlib.datasets.view import torch_image_to_numpy
import matplotlib.pyplot as plt
import numpy as np


def plotting_test():

    for img_id in ['orig', 'undist', 'warp']:
        data_conf = {
            'val_slices': ['9'],
            'val_batch_size': 1,
        }
        if img_id == 'orig':
            continue
        elif img_id == 'undist':
            data_conf['undistort_images'] = True
        elif img_id == 'warp':
            data_conf['undistort_images'] = True
            data_conf['warp_PY_images'] = True
        else:
            raise ValueError()

        dataset = get_dataset('cmu')(data_conf)
        val_loader = dataset.get_data_loader('val')
        for data in val_loader:
            ref_img = data['ref']['image'][0]
            ref_name = data['ref']['name'][0].split('.')[0]
            ref_img = torch_image_to_numpy(ref_img).astype(np.int32)
            plt.imshow(ref_img)
            plt.savefig(f"tests/plots/{img_id}_{ref_name}.png")


if __name__ == "__main__":
    plotting_test()
