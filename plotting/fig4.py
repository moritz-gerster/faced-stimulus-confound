"""Figure 4: Single video per emotion, cross-subject (Experiment 2d).

Panels:
  a) Cross-subject CV pipeline with 2nd/3rd videos grayed out
  b) Training data comparison: 2a vs 2d (hours)
  c) Accuracy comparison: 2a vs 2d

Run: python -m plotting.fig4
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

from .style import (
    setup, save, chance_line, bar, panel_label,
    DOUBLE_COL, FONT_SIZE,
    COLOR_BAR_A, COLOR_BAR_B,
)
from .panels import (
    load_scores, panel_cross_subject_pipeline, KEPT_MASK_SINGLE_VIDEO,
)

N_TRAIN_SUBJECTS = 111  # 9 folds of ~12 subjects

TRAIN_2A = N_TRAIN_SUBJECTS * 28 * 30  # seconds
TRAIN_2D = N_TRAIN_SUBJECTS * 9 * 30


def _panel_training_data(ax):
    """Comparison bar chart: training data in hours."""
    w = 0.3
    x1, x2 = 0, 0.345
    train_2a_h = TRAIN_2A / 3600
    train_2d_h = TRAIN_2D / 3600

    ax.bar(x1, train_2a_h, width=w, color=COLOR_BAR_A, edgecolor="white",
           linewidth=0.5, zorder=2)
    ax.bar(x2, train_2d_h, width=w, color=COLOR_BAR_B, edgecolor="white",
           linewidth=0.5, zorder=2)

    ax.text(x1, 3.6, f"{train_2a_h:.1f}", ha="center", va="center",
            fontsize=FONT_SIZE, color="white", fontweight="bold", zorder=3)
    ax.text(x2, 3.6, f"{train_2d_h:.1f}", ha="center", va="center",
            fontsize=FONT_SIZE, color="white", fontweight="bold", zorder=3)

    ax.set_ylim(0, train_2a_h * 1.15)
    ax.set_xlim(-0.25, 0.55)
    ax.set_xticks([x1, x2])
    ax.set_xticklabels(["2a", "2d"])


def _panel_accuracy(ax):
    """Comparison bar chart: 2a vs 2d accuracy with SEM."""
    scores_2a = load_scores("baseline")
    scores_2d = load_scores("single_video")

    w = 0.3
    x1, x2 = 0, 0.345

    bar(ax, x1, scores_2a, COLOR_BAR_A, width=w)
    acc_2a = np.mean(scores_2a) * 100

    bar(ax, x2, scores_2d, COLOR_BAR_B, width=w)
    acc_2d = np.mean(scores_2d) * 100

    ax.text(x1, 6, f"{acc_2a:.1f}", ha="center", va="center",
            fontsize=FONT_SIZE, color="white", fontweight="bold", zorder=3)
    ax.text(x2, 6, f"{acc_2d:.1f}", ha="center", va="center",
            fontsize=FONT_SIZE, color="white", fontweight="bold", zorder=3)

    chance_line(ax, 1 / 9, show_label=False)

    ax.set_xlim(-0.25, 0.55)
    ax.set_xticks([x1, x2])
    ax.set_xticklabels(["2a", "2d"])


def plot():
    setup()

    fig = plt.figure(figsize=(DOUBLE_COL, 2.5))
    gs = GridSpec(1, 5, figure=fig,
                 width_ratios=[8, 0.3, 2, 0.3, 2],
                 wspace=0.05)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 2])
    ax_c = fig.add_subplot(gs[0, 4])

    panel_cross_subject_pipeline(ax_a, kept_mask=KEPT_MASK_SINGLE_VIDEO)
    _panel_training_data(ax_b)
    _panel_accuracy(ax_c)

    panel_label(ax_a, "a", title="1 video per emotion")
    panel_label(ax_b, "b", title="Train data (hours)")
    panel_label(ax_c, "c", title="Mean accuracy (%)")

    fig.tight_layout()
    save(fig, "fig4")
    plt.close(fig)
    return fig


if __name__ == "__main__":
    plot()
