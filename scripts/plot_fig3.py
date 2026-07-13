import torch
import torch.nn as nn
import time
import matplotlib.pyplot as plt
from models.networks import ODENet
from data.dataloaders import get_mnist_dataloaders
import os


def evaluate_tolerances(model, x, y):
    tolerances = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]

    model.ode_block.atol = 1e-9
    model.ode_block.rtol = 1e-9
    with torch.no_grad():
        perfect_h_T = model.ode_block(model.downsampling(x))

    results = []

    for tol in tolerances:
        model.ode_block.atol = tol
        model.ode_block.rtol = tol

        model.ode_func.nfe = 0
        start_time = time.time()

        h = model.downsampling(x)
        h_T = model.ode_block(h)
        logits = model.fc(h_T)
        loss = nn.CrossEntropyLoss()(logits, y)

        forward_time = time.time() - start_time
        nfe_forward = model.ode_func.nfe

        num_error = torch.mean((perfect_h_T - h_T) ** 2).item()

        model.ode_func.nfe = 0
        loss.backward()
        nfe_backward = model.ode_func.nfe

        results.append(
            {
                "tol": tol,
                "nfe_forward": nfe_forward,
                "nfe_backward": nfe_backward,
                "time": forward_time,
                "error": num_error,
            }
        )

    return results


def plot_figure_3(results, epoch):
    """Plots the replications of Fig 3a, 3b, and 3c."""
    nfes = [r["nfe_forward"] for r in results]
    errors = [r["error"] for r in results]
    times = [r["time"] for r in results]
    nfes_bw = [r["nfe_backward"] for r in results]

    max_time = max(times)
    rel_times = [t / max_time for t in times]
    colors = nfes  # Map color to NFE like the paper

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # (a) NFE Forward vs Numerical Error
    sc0 = axes[0].scatter(nfes, errors, c=colors, cmap="rainbow", s=100)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("NFE Forward", fontsize=14)
    axes[0].set_ylabel("Numerical Error", fontsize=14)
    axes[0].set_xlim(0, max(nfes) + 20)

    # (b) NFE Forward vs Relative Time
    axes[1].scatter(nfes, rel_times, c=colors, cmap="rainbow", s=100)
    axes[1].set_xlabel("NFE Forward", fontsize=14)
    axes[1].set_ylabel("Relative Time", fontsize=14)
    axes[1].set_xlim(0, max(nfes) + 20)
    axes[1].set_ylim(0, 1.1)

    # (c) NFE Forward vs NFE Backward
    axes[2].scatter(nfes, nfes_bw, c=colors, cmap="rainbow", s=100)
    axes[2].plot([0, max(nfes) + 20], [0, max(nfes) + 20], "k--", alpha=0.5)  # x=y line
    axes[2].set_xlabel("NFE Forward", fontsize=14)
    axes[2].set_ylabel("NFE Backward", fontsize=14)
    axes[2].set_xlim(0, max(nfes) + 20)
    axes[2].set_ylim(0, max(nfes_bw) + 20)

    plt.tight_layout()
    os.makedirs("plots", exist_ok=True)
    plt.savefig(f"plots/fig3_epoch_{epoch}.png")
    plt.close()
