from pixloc.pixlib.datasets import get_dataset
from pixloc.pixlib.datasets.view import torch_image_to_numpy
import matplotlib.pyplot as plt


def plotting_test():
    data_conf = {
        'val_slices': ['9'],
        'val_number_per_slice': 5,
        'val_batch_size': 1,
    }
    dataset = get_dataset('CMU')(data_conf)
    val_loader = dataset.get_data_loader('val')
    for data in val_loader:
        ref_img = torch_image_to_numpy(data['ref'])
        # query_img = torch_image_to_numpy(data['query'])
        plt.imshow(ref_img)
        plt.savefig("plots/ref1.png")
        break


if __name__ == "__main__":
    plotting_test()
