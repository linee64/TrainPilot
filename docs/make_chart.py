"""Render the T4 FP32 vs AMP bar chart used in README.md."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).with_name("t4_results.png")

LABELS = ["CIFAR-10 ResNet-18", "DistilBERT IMDB", "nanoGPT Shakespeare"]
FP32_MS = [116.8, 310.4, 182.0]
AMP_MS = [46.8, 96.3, 55.2]
SPEEDUP = ["2.5x", "3.22x", "3.3x"]


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    y = np.arange(len(LABELS))
    bar_h = 0.36
    fig, ax = plt.subplots(figsize=(9.2, 3.8), dpi=160)
    fp32_bars = ax.barh(y + bar_h / 2, FP32_MS, bar_h, label="FP32", color="#4b5563")
    amp_bars = ax.barh(y - bar_h / 2, AMP_MS, bar_h, label="AMP (FP16)", color="#2563eb")
    ax.set_yticks(y)
    ax.set_yticklabels(LABELS)
    ax.invert_yaxis()
    ax.set_xlabel("Median step time (ms)")
    ax.set_xlim(0, max(FP32_MS) * 1.28)
    for bar, value in zip(fp32_bars, FP32_MS):
        ax.text(
            value + 5,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.1f}",
            va="center",
            ha="left",
            fontsize=8,
            color="#374151",
        )
    for bar, value, speedup in zip(amp_bars, AMP_MS, SPEEDUP):
        ax.text(
            value + 5,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.1f}   {speedup}",
            va="center",
            ha="left",
            fontsize=8,
            color="#1d4ed8",
            fontweight="bold",
        )
    ax.legend(frameon=False, loc="lower right")
    ax.set_title("Short-run step time on a Tesla T4 (Colab)")
    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
