"""Figure 1: Intra-subject decoding pipeline (Experiments 1a and 1b).

Panels:
  a) Full pipeline: 28 videos, 10-fold CV, classifier, pie charts
  b) Single-video pipeline: 2nd/3rd videos grayed out
  c) Training data comparison: 1a vs 1b (minutes)
  d) Accuracy comparison: 1a vs 1b

Run: python -m plotting.fig1
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
    panel_cv_classify, load_scores, KEPT_MASK_SINGLE_VIDEO,
)

TRAIN_1A = 27 * 28  # seconds
TRAIN_1B = 27 * 9


def _panel_training_data(ax):
    """Comparison bar chart: training data in minutes."""
    w = 0.33
    x1, x2 = 0, 0.345
    train_1a_min = TRAIN_1A / 60
    train_1b_min = TRAIN_1B / 60

    ax.bar(x1, train_1a_min, width=w, color=COLOR_BAR_A, edgecolor="white",
           linewidth=0.5, zorder=2)
    ax.bar(x2, train_1b_min, width=w, color=COLOR_BAR_B, edgecolor="white",
           linewidth=0.5, zorder=2)

    ax.text(x1, 3.5, f"{train_1a_min:.1f}", ha="center", va="top",
            fontsize=FONT_SIZE, color="white", fontweight="bold", zorder=3)
    ax.text(x2, 3.5, f"{train_1b_min:.1f}", ha="center", va="top",
            fontsize=FONT_SIZE, color="white", fontweight="bold", zorder=3)

    ax.set_ylim(0, train_1a_min * 1.15)
    ax.set_xlim(-0.25, 0.55)
    ax.set_xticks([x1, x2])
    ax.set_xticklabels(["1a", "1b"])


def _panel_accuracy(ax):
    """Comparison bar chart: 1a vs 1b accuracy with SEM."""
    scores_1a = load_scores("intra_subject")
    scores_1b = load_scores("intra_single_video")

    w = 0.33
    x1, x2 = 0, 0.345

    bar(ax, x1, scores_1a, COLOR_BAR_A, width=w)
    acc_1a = np.mean(scores_1a) * 100

    bar(ax, x2, scores_1b, COLOR_BAR_B, width=w)
    acc_1b = np.mean(scores_1b) * 100

    ax.text(x1, 16.65, f"{acc_1a:.1f}", ha="center", va="center",
            fontsize=FONT_SIZE, color="white", fontweight="bold", zorder=3)
    ax.text(x2, 16.65, f"{acc_1b:.1f}", ha="center", va="center",
            fontsize=FONT_SIZE, color="white", fontweight="bold", zorder=3)

    chance_line(ax, 1 / 9, show_label=False)

    ax.set_xlim(-0.25, 0.55)
    ax.set_xticks([x1, x2])
    ax.set_xticklabels(["1a", "1b"])


def plot():
    setup()

    fig = plt.figure(figsize=(DOUBLE_COL, 2.5))
    gs = GridSpec(1, 6, figure=fig,
                 width_ratios=[4, 3, 0.3, 1.5, 0.5, 1.5],
                 wspace=0.05)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 3])
    ax_d = fig.add_subplot(gs[0, 5])

    dummy_fracs_a = [0.65, 0.55, 0.60, 0.58, 0.70, 0.62, 0.50, 0.55, 0.60]
    dummy_fracs_b = [0.75, 0.70, 0.72, 0.68, 0.80, 0.74, 0.65, 0.70, 0.72]

    panel_cv_classify(ax_a, show_emotion_labels=True, dummy_fracs=dummy_fracs_a)
    panel_cv_classify(ax_b, kept_mask=KEPT_MASK_SINGLE_VIDEO, dummy_fracs=dummy_fracs_b)
    _panel_training_data(ax_c)
    _panel_accuracy(ax_d)

    panel_label(ax_a, "a", title="3 videos per emotion")
    panel_label(ax_b, "b", title="1 video per emotion", x=0.0)
    panel_label(ax_c, "c", title="Train data (min)", x=-0.3)
    panel_label(ax_d, "d", title="Accuracy (%)", x=-0.3)

    fig.tight_layout()
    save(fig, "fig1")
    plt.close(fig)
    return fig


if __name__ == "__main__":
    plot()
