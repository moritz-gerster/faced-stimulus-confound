"""Figure 5: CLISA deep learning results (Experiment 2e).

Compact summary showing that CLISA reproduces the same confound pattern as SVM.

Panels:
  a) Baseline: 34.2%
  b) Concordance: concordant (27.7%) vs discordant (31.4%)
  c) Subjective concordance: concordant (26.3%) vs discordant (17.1%)
  d) Single video: baseline (34.2%) vs single (50.6%)

Run: python -m plotting.fig5
"""
import matplotlib.pyplot as plt
import numpy as np

from .style import (
    setup, save, bar, panel_label,
    FONT_SIZE,
    COLOR_BAR_A, COLOR_BAR_B, COLOR_CONC, COLOR_DISC,
)


def _panel_baseline(ax):
    """Single bar: CLISA baseline accuracy."""
    from .data import load_cross_subject_data
    data = load_cross_subject_data()
    scores = data["clisa_baseline"]
    bar(ax, 0, scores, COLOR_BAR_A, width=0.7)
    mean_acc = np.mean(scores) * 100

    ax.text(0, 6, f"{mean_acc:.1f}", ha="center", va="center", 
            fontsize=FONT_SIZE, color="white", fontweight="bold", zorder=3)

    ax.set_ylim(0, 55)
    ax.set_xlim(-0.6, 0.6)
    ax.set_xticks([])


def _panel_concordance(ax):
    """Two bars: concordant vs discordant from crowd-label classifier."""
    from .data import load_cross_subject_data
    data = load_cross_subject_data()

    w = 0.32
    x1, x2 = 0, 0.345

    conc = data["clisa_bl_conc"]
    disc = data["clisa_bl_disc"]

    bar(ax, x1, conc, COLOR_CONC, width=w, alpha=0.6)
    val_conc = np.mean(conc) * 100

    bar(ax, x2, disc, COLOR_DISC, width=w, alpha=0.6)
    val_disc = np.mean(disc) * 100

    ax.text(x1, 6, f"{val_conc:.1f}", ha="center", va="center",
            fontsize=FONT_SIZE, color="white", fontweight="bold", zorder=3)
    ax.text(x2, 6, f"{val_disc:.1f}", ha="center", va="center",
            fontsize=FONT_SIZE, color="white", fontweight="bold", zorder=3)

    ax.set_xlim(-0.25, 0.55)
    ax.set_xticks([x1, x2])
    ax.set_xticklabels(["Conc.", "Disc."])


def _panel_subjective_concordance(ax):
    """Two bars: concordant vs discordant from subjective-label classifier."""
    from .data import load_cross_subject_data
    data = load_cross_subject_data()

    w = 0.32
    x1, x2 = 0, 0.345

    conc = data["clisa_subj_conc"]
    disc = data["clisa_subj_disc"]

    bar(ax, x1, conc, COLOR_CONC, width=w, alpha=0.6)
    val_conc = np.mean(conc) * 100

    bar(ax, x2, disc, COLOR_DISC, width=w, alpha=0.6)
    val_disc = np.mean(disc) * 100

    ax.text(x1, 6, f"{val_conc:.1f}", ha="center", va="center",
            fontsize=FONT_SIZE, color="white", fontweight="bold", zorder=3)
    ax.text(x2, 6, f"{val_disc:.1f}", ha="center", va="center",
            fontsize=FONT_SIZE, color="white", fontweight="bold", zorder=3)

    ax.set_xlim(-0.25, 0.55)
    ax.set_xticks([x1, x2])
    ax.set_xticklabels(["Conc.", "Disc."])


def _panel_single_video(ax):
    """Single bar: CLISA single-video accuracy."""
    from .data import load_cross_subject_data
    data = load_cross_subject_data()

    single = data["clisa_single"]

    bar(ax, 0, single, COLOR_BAR_B, width=0.7)
    val_single = np.mean(single) * 100

    ax.text(0, 6, f"{val_single:.1f}", ha="center", va="center",
            fontsize=FONT_SIZE, color="white", fontweight="bold", zorder=3)

    ax.set_xlim(-0.6, 0.6)
    ax.set_xticks([])


def plot():
    setup()

    fig, (ax_a, ax_b, ax_c, ax_d) = plt.subplots(
        1, 4, figsize=(4, 2.5), sharey=True,
        width_ratios=[1, 1.4, 1.4, 1])

    _panel_baseline(ax_a)
    _panel_concordance(ax_b)
    _panel_subjective_concordance(ax_c)
    _panel_single_video(ax_d)

    fig.tight_layout()

    # Single chance line spanning all panels (drawn in figure coords)
    import matplotlib.lines as mlines
    chance_y = 100 / 9
    renderer = fig.canvas.get_renderer()
    _, fig_y = fig.transFigure.inverted().transform(
        ax_a.transData.transform((0, chance_y)))
    x_left = ax_a.get_position().x0
    x_right = ax_d.get_position().x1
    line = mlines.Line2D(
        [x_left, x_right], [fig_y, fig_y],
        transform=fig.transFigure, color="#333333",
        linewidth=.6, linestyle="--", zorder=5, clip_on=False)
    fig.add_artist(line)


    panel_label(ax_a, "a", title="Baseline", x=-0.3)
    panel_label(ax_b, "b", title="Concordance", x=-0.1)
    panel_label(ax_c, "c", title="Subjective labels", x=-0.1)
    panel_label(ax_d, "d", title="Single video")

    save(fig, "fig5")
    plt.close(fig)
    return fig


if __name__ == "__main__":
    plot()
