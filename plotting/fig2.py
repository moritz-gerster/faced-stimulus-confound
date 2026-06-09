"""Figure 2: Cross-subject baseline (Experiment 2a).

Panels:
  a) Cross-subject CV pipeline: 10 subjects (9 train + 1 test),
     each with 28 videos. Train data faded, test data saturated.
     Classifier box (Train/Test split) -> 9 pie charts.
  b) Result bar (39.4% accuracy)

Run: python -m plotting.fig2
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

from .style import (
    setup, save, chance_line, bar, panel_label,
    DOUBLE_COL, FONT_SIZE,
    COLOR_BAR_A,
)
from .panels import load_scores, panel_cross_subject_pipeline


def _panel_result(ax):
    """Single bar showing cross-subject baseline accuracy."""
    scores = load_scores("baseline")
    bar(ax, 0, scores, COLOR_BAR_A, width=0.45)
    mean_acc = np.mean(scores) * 100

    ax.text(0, mean_acc - 3, f"{mean_acc:.1f}", ha="center", va="top",
            fontsize=FONT_SIZE + 1, color="white", fontweight="bold", zorder=3)

    chance_line(ax, 1 / 9, show_label=False)

    ax.set_xlim(-0.6, 0.6)
    ax.set_xticks([])


def plot():
    setup()

    fig = plt.figure(figsize=(DOUBLE_COL, 2.5))
    gs = GridSpec(1, 3, figure=fig,
                 width_ratios=[6, 0.2, 1.5],
                 wspace=0.05)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 2])

    panel_cross_subject_pipeline(ax_a)
    _panel_result(ax_b)

    panel_label(ax_a, "a", title="Subject-wise Cross-Validation")
    panel_label(ax_b, "b", title="Mean accuracy (%)", x=-0.23)

    fig.tight_layout()
    save(fig, "fig2")
    plt.close(fig)
    return fig


if __name__ == "__main__":
    plot()
