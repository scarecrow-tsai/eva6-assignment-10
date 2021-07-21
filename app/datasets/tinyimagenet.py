import numpy as np
from torch.utils import data
from torchvision import datasets


def load_tinyimagenet(
    dataset_path: str, is_train: bool, image_transforms: "Albumentation Transforms"
) -> "PyTorch Dataset":

    """
    Load TinyImageNet dataset using torchvision.
    ---------------------------------------

        - Input: dataset_path, is_train, and image_transforms.
        - Output: PyTorch Dataset object.
    """

    if is_train:
        dataset_path = dataset_path + "train"
    else:
        dataset_path = dataset_path + "test"
    return datasets.ImageFolder(
        root=dataset_path,
        transform=lambda x: image_transforms(image=np.array(x))["image"],
    )
