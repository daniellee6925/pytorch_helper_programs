"""This function creates PyTorch DataLoaders for iamge classification data"""

import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

NUM_WORKERS = os.cpu_count()


def create_dataloaders(
    train_dir: str,
    test_dir: str,
    transforms: transforms.Compose,
    batch_size: int,
    num_workers: int = NUM_WORKERS,
):
    """
    Args:
    train_dir: path to training directory
    test_dir: path to testing directory
    transforms: data augmentation for training and testing data
    batch_size: number of samples per batch in each DataLoader
    num_workers: number of workers in each DataLoader

    Returns:
    Tuple of (train_dataloader, test_dataloader, class_names)
    class_names: list of target classes
    """

    # use ImageFolder to create datasets
    train_data = datasets.ImageFolder(train_dir, transform=transforms)
    test_data = datasets.ImageFolder(test_dir, transform=transforms)

    # get class names
    class_names = train_data.classes

    # Turn Images into data loaders
    train_dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_dataloader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,  # do not shuffle test data
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_dataloader, test_dataloader, class_names
