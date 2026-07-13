import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
import torch

from data.synthetic import make_circles


def angle_mask(x: torch.Tensor, angle_start: float, angle_width: float) -> torch.Tensor:
    """Return mask for points whose polar angle lies in the removed sector."""
    if angle_width <= 0.0:
        raise ValueError("angle_width must be positive.")

    two_pi = 2.0 * math.pi
    theta = torch.atan2(x[:, 1], x[:, 0])
    theta = torch.remainder(theta, two_pi)

    angle_start = angle_start % two_pi
    if angle_width >= two_pi:
        return torch.ones_like(theta, dtype=torch.bool)

    angle_end = angle_start + angle_width
    if angle_end <= two_pi:
        return (theta >= angle_start) & (theta <= angle_end)

    wrapped_end = angle_end - two_pi
    return (theta >= angle_start) | (theta <= wrapped_end)


def main() -> None:
    """Plot the missing-slice dataset construction used in ANODE experiments."""
    n_samples = 1000
    n_val_samples = 3000
    noise = 0.05
    seed = 0
    missing_angle_start = 0.0
    missing_angle_width = math.pi / 5.0

    x_train_pool, y_train_pool = make_circles(
        n_samples=n_samples,
        noise=noise,
        seed=seed,
    )
    train_slice_mask = angle_mask(
        x_train_pool,
        missing_angle_start,
        missing_angle_width,
    )
    x_train = x_train_pool[~train_slice_mask]
    y_train = y_train_pool[~train_slice_mask]

    x_val, y_val = make_circles(
        n_samples=n_val_samples,
        noise=noise,
        seed=seed + 10_000,
    )
    val_slice_mask = angle_mask(
        x_val,
        missing_angle_start,
        missing_angle_width,
    )
    x_slice = x_val[val_slice_mask]
    y_slice = y_val[val_slice_mask]

    output_dir = Path("figures/anode/corrected")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "slice_dataset_seed0.png"

    fig, ax = plt.subplots(figsize=(6, 6))

    wedge = Wedge(
        center=(0.0, 0.0),
        r=1.7,
        theta1=math.degrees(missing_angle_start),
        theta2=math.degrees(missing_angle_start + missing_angle_width),
        alpha=0.15,
        label="Removed angular sector",
    )
    ax.add_patch(wedge)

    x_train_np = x_train.numpy()
    y_train_np = y_train.numpy()
    x_slice_np = x_slice.numpy()
    y_slice_np = y_slice.numpy()

    for class_id in (0, 1):
        train_mask = y_train_np == class_id
        slice_mask = y_slice_np == class_id

        ax.scatter(
            x_train_np[train_mask, 0],
            x_train_np[train_mask, 1],
            s=12,
            marker="o",
            alpha=0.50,
            label=f"Observed train, class {class_id}",
        )
        ax.scatter(
            x_slice_np[slice_mask, 0],
            x_slice_np[slice_mask, 1],
            s=18,
            marker="x",
            alpha=0.85,
            label=f"Held-out slice, class {class_id}",
        )

    ax.set_title("Missing-slice circles dataset")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-1.6, 1.6)
    ax.legend(loc="upper left", fontsize=7, frameon=True)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

    print(f"Saved {output_path}")
    print(f"Observed train points: {len(x_train)}")
    print(f"Held-out slice points: {len(x_slice)}")


if __name__ == "__main__":
    main()
