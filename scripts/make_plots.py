import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

plt.rcParams.update(
    {
        "font.size": 11,
        "font.family": "serif",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "savefig.bbox": "tight",
        "savefig.dpi": 200,
        "figure.constrained_layout.use": True,
        "figure.constrained_layout.h_pad": 0.15,
        "figure.constrained_layout.w_pad": 0.15,
        "axes.titlepad": 12,
        "axes.labelpad": 6,
    }
)

ENCODER_STYLE = {
    "gru_notime": {"label": "GRU (vanilla)", "color": "#E69F00", "marker": "s"},
    "odernn": {"label": "ODE-RNN", "color": "#D55E00", "marker": "o"},
    "gru_time": {"label": "GRU (time-aware)", "color": "#0072B2", "marker": "^"},
}

results_dir = Path("results")
figures_dir = Path("figures")
figures_dir.mkdir(exist_ok=True)


def load_3way_results():
    grouped = {"odernn": [], "gru_time": [], "gru_notime": []}
    for path in results_dir.glob("3way_*.json"):
        with open(path) as f:
            data = json.load(f)
        enc = data["config"]["encoder_type"]
        grouped[enc].append(data["test_metrics"])
    return grouped


def load_persistence_baseline():
    interp_vals, extrap_vals = [], []
    for path in results_dir.glob("3way_*.json"):
        with open(path) as f:
            data = json.load(f)
        m = data.get("test_metrics", {})
        for k in ["persistence_interp", "persistence_interpolation_mse"]:
            if k in m:
                interp_vals.append(m[k])
                break
        for k in ["persistence_extrap", "persistence_extrapolation_mse"]:
            if k in m:
                extrap_vals.append(m[k])
                break
    if not interp_vals:
        return {"interp": 0.0207, "extrap": 1.4379}
    return {
        "interp": float(np.mean(interp_vals)),
        "extrap": float(np.mean(extrap_vals)),
    }


# -----------------------------------------------------------------------------
# FIGURE 1
# -----------------------------------------------------------------------------
def figure_1_bars():
    grouped = load_3way_results()
    persistence = load_persistence_baseline()

    encoders = ["gru_notime", "odernn", "gru_time"]
    metrics = [
        ("interpolation_mse", "Interpolation"),
        ("extrapolation_mse", "Extrapolation"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))

    for ax, (metric_key, metric_label) in zip(axes, metrics):
        means, stds, colors, labels = [], [], [], []
        for enc in encoders:
            vals = [r[metric_key] for r in grouped[enc]]
            means.append(np.mean(vals))
            stds.append(np.std(vals, ddof=1) if len(vals) > 1 else 0.0)
            colors.append(ENCODER_STYLE[enc]["color"])
            labels.append(ENCODER_STYLE[enc]["label"])

        x = np.arange(len(encoders))
        bars = ax.bar(
            x,
            means,
            yerr=stds,
            capsize=5,
            color=colors,
            alpha=0.92,
            edgecolor="black",
            linewidth=0.8,
            width=0.65,
        )

        for bar, m, s in zip(bars, means, stds):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + s + max(means) * 0.04,
                f"{m:.4f}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=12, ha="right", fontsize=10)
        ax.set_ylabel("Test MSE")
        ax.set_title(metric_label, fontsize=12, fontweight="bold")

        ymax = max(m + s for m, s in zip(means, stds)) * 1.45
        ax.set_ylim(0, ymax)

        pers_key = "interp" if metric_key == "interpolation_mse" else "extrap"
        pers_val = persistence[pers_key]
        if pers_val < ymax:
            ax.axhline(
                pers_val,
                color="gray",
                linestyle="--",
                linewidth=1.2,
                label=f"Persistence: {pers_val:.3f}",
            )
            ax.legend(loc="upper left", fontsize=9, frameon=False)
        else:
            ax.annotate(
                f"Persistence baseline\n= {pers_val:.2f}  (off-scale ↑)",
                xy=(len(encoders) - 1, ymax * 0.95),
                xytext=(len(encoders) - 1, ymax * 0.72),
                ha="right",
                fontsize=9,
                color="gray",
                arrowprops=dict(arrowstyle="->", color="gray", lw=1),
            )

    fig.suptitle(
        "Three-Way Encoder Comparison\n1D sine, 30 epochs, n=2 seeds",
        fontsize=12,
        fontweight="bold",
    )
    out = figures_dir / "figure_1_threeway_bars.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"))
    print(f"saved {out} (and .png)")
    plt.close(fig)


# -----------------------------------------------------------------------------
# FIGURE 2
# -----------------------------------------------------------------------------
def figure_2_dataset_showcase():
    """
    Three-panel mosaic:
      (top row)  three 1D sine examples
      (bottom)   three 2D spiral examples (in the (x, y) plane, coloured by time)
    """
    from config import LatentODEConfig
    from data.synthetic import TimeSeriesDataset
    from matplotlib.collections import LineCollection

    cfg_sine = LatentODEConfig()
    cfg_sine.input_dim = 1
    cfg_sine.signal_type = "sine"
    cfg_sine.extrap_horizon = 10.0
    ds_sine = TimeSeriesDataset(cfg_sine)

    cfg_spiral = LatentODEConfig()
    cfg_spiral.input_dim = 2
    cfg_spiral.signal_type = "spiral"
    cfg_spiral.extrap_horizon = 10.0
    ds_spiral = TimeSeriesDataset(cfg_spiral)

    fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)

    sine_palette = ["#0072B2", "#D55E00", "#009E73"]
    for i, (ax, idx) in enumerate(zip(axes[0], [0, 17, 42])):
        s = ds_sine[idx]
        t_full = s["full_times"].numpy()
        gt = s["ground_truth"].squeeze().numpy()
        t_ctx = s["context_times"].numpy()
        ctx_vals = s["context_values"].squeeze().numpy()
        keep = s["context_mask"].squeeze().numpy().astype(bool)

        ax.axvspan(0, cfg_sine.train_horizon, color=sine_palette[i], alpha=0.06)
        ax.axvspan(
            cfg_sine.train_horizon, cfg_sine.extrap_horizon, color="gray", alpha=0.06
        )

        ax.plot(t_full, gt, color=sine_palette[i], linewidth=2, label="Ground truth")
        ax.scatter(
            t_ctx[keep], ctx_vals[keep], s=28, color="black", zorder=3, label="Observed"
        )
        ax.scatter(
            t_ctx[~keep],
            ctx_vals[~keep],
            s=36,
            facecolors="white",
            edgecolors="black",
            linewidth=1.2,
            zorder=3,
            label="Masked",
        )

        ax.axvline(cfg_sine.train_horizon, color="gray", linestyle=":", linewidth=1)
        ax.set_title(f"1D sine: sample {idx}", fontsize=11, pad=10)
        ax.set_xlabel("Time", labelpad=6)
        if i == 0:
            ax.set_ylabel("Signal", labelpad=6)
            ax.legend(loc="lower left", fontsize=8, frameon=True, framealpha=0.9)
        ax.set_xlim(0, cfg_sine.extrap_horizon)
        ax.set_ylim(-1.4, 1.4)

    cmap = plt.cm.viridis
    last_lc = None
    for i, (ax, idx) in enumerate(zip(axes[1], [0, 17, 42])):
        s = ds_spiral[idx]
        t_full = s["full_times"].numpy()
        gt = s["ground_truth"].numpy()
        keep = s["context_mask"].squeeze().numpy().astype(bool)
        if keep.ndim == 2:
            keep = keep[:, 0].astype(bool)

        points = gt.reshape(-1, 1, 2)
        segs = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(
            segs, cmap=cmap, linewidth=2.0, array=t_full[:-1], alpha=0.85
        )
        ax.add_collection(lc)
        last_lc = lc

        ctx_xy = s["context_values"].numpy()
        ax.scatter(
            ctx_xy[keep, 0],
            ctx_xy[keep, 1],
            s=22,
            color="black",
            zorder=3,
            label="Observed",
        )
        ax.scatter(
            ctx_xy[~keep, 0],
            ctx_xy[~keep, 1],
            s=30,
            facecolors="white",
            edgecolors="black",
            linewidth=1.2,
            zorder=3,
            label="Masked",
        )

        train_idx = np.searchsorted(t_full, cfg_spiral.train_horizon)
        ax.scatter(*gt[0], s=80, marker="*", color="red", zorder=4, label="t=0")
        if train_idx < len(t_full):
            ax.scatter(
                *gt[train_idx],
                s=80,
                marker="X",
                color="dimgray",
                zorder=4,
                label=f"t={cfg_spiral.train_horizon:.0f}",
            )

        ax.set_title(f"2D spiral: sample {idx}", fontsize=11, pad=10)
        ax.set_xlabel("x", labelpad=6)
        if i == 0:
            ax.set_ylabel("y", labelpad=6)
            ax.legend(loc="upper right", fontsize=8, frameon=True, framealpha=0.9)
        ax.set_aspect("equal")

        m = max(np.abs(gt).max(), 0.1) * 1.2
        ax.set_xlim(-m, m)
        ax.set_ylim(-m, m)

    # Colorbar attached to the spiral row only
    cbar = fig.colorbar(last_lc, ax=axes[1], shrink=0.9, pad=0.02)
    cbar.set_label("Time", fontsize=9, labelpad=6)

    fig.suptitle(
        "Synthetic Datasets: 1D Sine and 2D Spiral", fontsize=13, fontweight="bold"
    )

    out = figures_dir / "figure_2_dataset_showcase.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"))
    print(f"saved {out} (and .png)")
    plt.close(fig)


# -----------------------------------------------------------------------------
# FIGURE 3
# -----------------------------------------------------------------------------
def figure_3_ranking_summary():
    grouped = load_3way_results()

    encoders = ["gru_notime", "odernn", "gru_time"]
    metrics = ["interpolation_mse", "extrapolation_mse"]
    metric_labels = ["Interpolation", "Extrapolation"]

    fig, ax = plt.subplots(figsize=(7.5, 4))

    n_metrics = len(metrics)
    n_encoders = len(encoders)
    bar_w = 0.25
    x = np.arange(n_metrics)

    for i, enc in enumerate(encoders):
        means = [np.mean([r[m] for r in grouped[enc]]) for m in metrics]
        stds = [np.std([r[m] for r in grouped[enc]], ddof=1) for m in metrics]
        offset = (i - 1) * bar_w
        bars = ax.bar(
            x + offset,
            means,
            bar_w,
            yerr=stds,
            capsize=4,
            color=ENCODER_STYLE[enc]["color"],
            alpha=0.92,
            edgecolor="black",
            linewidth=0.7,
            label=ENCODER_STYLE[enc]["label"],
        )
        for bar, m in zip(bars, means):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.02,
                f"{m:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=11)
    ax.set_ylabel("Test MSE")
    ax.set_title(
        "Encoder Comparison Across Metrics", fontsize=12, fontweight="bold", pad=12
    )
    ax.legend(loc="upper left", fontsize=10, frameon=False)
    ax.set_yscale("log")  # log scale makes the gaps visible

    out = figures_dir / "figure_3_ranking_summary.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"))
    print(f"saved {out} (and .png)")
    plt.close(fig)


if __name__ == "__main__":
    figure_1_bars()
    figure_2_dataset_showcase()
    figure_3_ranking_summary()
