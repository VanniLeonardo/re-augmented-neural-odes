"""Manual one-forward-pass smoke check for the MLP Neural-ODE on MNIST.

Renamed from ``test_continuous_mnist.py`` so pytest does not collect it (it is a
script, not a test, and downloading MNIST + a forward pass at collection time
broke ``pytest``). Run directly: ``python -m scripts.smoke_continuous_mnist``.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from data.dataloaders import get_mnist_dataloaders
from models.networks import ODENet


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, _ = get_mnist_dataloaders(batch_size=64, seed=0)
    model = ODENet(
        data_dim=784, hidden_dim=128, num_classes=10, solver_type="dopri5"
    ).to(device)

    x, y = next(iter(train_loader))
    x, y = x.to(device), y.to(device)
    logits = model(x)

    print("input shape:", x.shape)
    print("labels shape:", y.shape)
    print("output shape:", logits.shape)
    print("output dtype:", logits.dtype)
    print("Forward NFE count:", model.ode_func.nfe)


if __name__ == "__main__":
    main()
