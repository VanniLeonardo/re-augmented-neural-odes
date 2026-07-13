from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = BASE_DIR / "figures"

MODELS = [
    {
        "name": "GRU (vanilla)",
        "color": "#EDB120",
        "history_file": "final_gru_notime_seed42_e10.json",
    },
    {
        "name": "ODE-RNN",
        "color": "#D95F02",
        "history_file": "final_odernn_seed42_e10.json",
    },
    {
        "name": "GRU (time-aware)",
        "color": "#1F77B4",
        "history_file": "final_gru_time_seed42_e10.json",
    },
]


def load_json(filename: str) -> dict:
    path = RESULTS_DIR / filename
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def history_curve(filename: str, metric_name: str) -> tuple[list[int], list[float]]:
    payload = load_json(filename)
    history = payload["history"]
    epochs = [int(row["epoch"]) for row in history]
    values = [float(row[metric_name]) for row in history]
    return epochs, values


def make_interpolation_curve_plot() -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.size": 9.5,
            "axes.labelsize": 10.5,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 9,
        }
    )

    fig, ax = plt.subplots(figsize=(5.4, 3.3))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ymin = float("inf")
    ymax = float("-inf")

    for model in MODELS:
        epochs, values = history_curve(model["history_file"], "val_interpolation_mse")
        ymin = min(ymin, min(values))
        ymax = max(ymax, max(values))

        ax.plot(
            epochs,
            values,
            color=model["color"],
            linewidth=2.2,
            marker="o",
            markersize=4.8,
            markerfacecolor=model["color"],
            markeredgecolor="white",
            markeredgewidth=0.8,
            label=model["name"],
        )

    yrange = ymax - ymin
    ax.set_xlim(1, 10)
    ax.set_ylim(ymin - 0.05 * yrange, ymax + 0.08 * yrange)
    ax.set_xticks(list(range(1, 11)))
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation interpolation MSE")

    ax.grid(axis="y", color="#B0B0B0", alpha=0.28, linewidth=0.8)
    ax.grid(axis="x", visible=False)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)
    ax.tick_params(axis="both", width=1.0, length=4)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        frameon=False,
        handlelength=2.0,
        columnspacing=1.2,
        handletextpad=0.5,
    )

    fig.subplots_adjust(left=0.14, right=0.98, bottom=0.18, top=0.86)

    out_path = FIGURES_DIR / "standalone_odernn_interp_curve.png"
    plt.savefig(out_path, dpi=260, bbox_inches="tight")
    plt.close()

    return out_path


def main() -> None:
    png_path = make_interpolation_curve_plot()
    print(f"Saved figure to: {png_path}")


if __name__ == "__main__":
    main()
