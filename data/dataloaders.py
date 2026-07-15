import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision import datasets, transforms
from typing import Optional, Tuple

from data.synthetic import make_circles, make_moons, make_spheres, make_spirals


def flatten_tensor(x: torch.Tensor) -> torch.Tensor:
    return x.view(-1)


def get_cifar10_dataloaders(
    batch_size: int,
    data_root: str = "./data",
    seed: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader]:
    """Train/test DataLoaders for CIFAR-10 (D4). ToTensor only (per-image [0,1]); no
    augmentation, to keep the (A)NODE comparison about the flow, not data augmentation.
    Seeded train shuffle for reproducibility (like the MNIST loader)."""
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform)
    generator = torch.Generator().manual_seed(seed) if seed is not None else None
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=0, generator=generator)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, test_loader


def get_mnist_dataloaders(
    batch_size: int,
    data_root: str = "./data",
    flatten: bool = True,
    seed: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader]:
    """Returns train and test DataLoaders for MNIST.

    Args:
        batch_size (int): Batch size for both loaders.
        data_root (str): Directory where MNIST is downloaded.
        flatten (bool): If True, flattens images from [1, 28, 28] to [784].
        seed (Optional[int]): If given, the train-loader shuffle order is made
            reproducible via a seeded generator. The original code left the train
            shuffle on the unseeded global RNG, so MNIST results were not
            reproducible run-to-run.
    """
    transform_list = [transforms.ToTensor()]
    if flatten:
        transform_list.append(transforms.Lambda(flatten_tensor))
    transform = transforms.Compose(transform_list)

    train_dataset = datasets.MNIST(
        root=data_root, train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root=data_root, train=False, download=True, transform=transform
    )

    generator = None
    if seed is not None:
        generator = torch.Generator().manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    return train_loader, test_loader


_DATASET_REGISTRY = {
    "circles": make_circles,
    "spirals": make_spirals,
    "moons": make_moons,
    "spheres": make_spheres,  # Dupont's filled-disk + annulus (App. F.2.1)
}


def get_dataloaders(
    dataset: str = "circles",
    n_samples: int = 1000,
    batch_size: int = 64,
    val_split: float = 0.2,
    noise: float = 0.05,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """Builds train and validation DataLoaders for a synthetic 2D dataset.

    Args:
        dataset (str): One of 'circles', 'spirals', 'moons'.
        n_samples (int): Total dataset size before splitting.
        batch_size (int): Batch size for both loaders.
        val_split (float): Fraction of data reserved for validation.
        noise (float): Noise level forwarded to the generator function.
        seed (int): Controls both data generation and the train/val split.
    """
    if dataset not in _DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset '{dataset}'. Choose from {list(_DATASET_REGISTRY)}."
        )

    X, y = _DATASET_REGISTRY[dataset](n_samples=n_samples, noise=noise, seed=seed)
    full_dataset = TensorDataset(X, y)

    n_val = int(len(full_dataset) * val_split)
    n_train = len(full_dataset) - n_val

    # seeded generator so split is reproducible independently of data generation
    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(full_dataset, [n_train, n_val], generator=generator)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
