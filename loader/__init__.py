from torch.utils import data

from loader.synthetic_dataset import SyntheticData
from loader.MNIST_dataset import RotatedShiftedMNIST
from loader.FashionMNIST_dataset import RotatedShiftedFashionMNIST
from loader.CIFAR10_dataset import RotatedShiftedCIFAR10

def get_dataloader(data_dict, **kwargs):
    dataset = get_dataset(data_dict)
    loader = data.DataLoader(
        dataset,
        batch_size=data_dict["batch_size"],
        shuffle=data_dict.get("shuffle", True)
    )
    return loader


def get_dataset(data_dict):
    name = data_dict["dataset"]
    if name == 'synthetic':
        dataset = SyntheticData(**data_dict)
    elif name == 'fashion':
        dataset = RotatedShiftedFashionMNIST(**data_dict)    
    elif name == 'mnist':
        dataset = RotatedShiftedMNIST(**data_dict)
    elif name == 'cifar10':
        dataset = RotatedShiftedCIFAR10(**data_dict)    
    return dataset