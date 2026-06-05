"""Figure 3: Concordance & subjective labels (Experiments 2b + 2c).

Panels:
  a) Concordance definition schematic
  b) Accuracy by concordance (crowd labels): concordant vs discordant
  c) Label replacement schematic: video labels -> subject reports
  d) Accuracy (subjective labels): overall, concordant, discordant

Run: python -m plotting.fig3
"""
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.gridspec import GridSpec

from .style import (
    setup, save, chance_line, bar, panel_label,
    DOUBLE_COL, FONT_SIZE, COLOR_TEXT,
    COLOR_BAR_A, COLOR_CONC, COLOR_DISC,
)
from .panels import EMOTION_COLORS


def _panel_concordance_concept(ax):
    """Schematic showing concordant vs discordant trial definition."""
    fear_color = EMOTION_COLORS[2]
    amusement_color = EMOTION_COLORS[5]

    cx = 2.8
    subj_offset = 2.5
    sa_x = cx - subj_offset
    sb_x = cx + subj_offset

    # Video box centered at cx
    vid_w, vid_h = 3.0, 0.8
    vid_x = cx - vid_w / 2
    vid_y = 0.2
    rect = mpatches.FancyBboxPatch(
        (vid_x, vid_y), vid_w, vid_h,
        boxstyle="round,pad=0.05",
        facecolor=fear_color, edgecolor="none", linewidth=0.4,
    )
    ax.add_patch(rect)
    ax.text(cx, vid_y + vid_h / 2, "Fear", ha="center", va="center",
            fontsize=FONT_SIZE, fontweight="bold", color="white")

    ax.text(cx, vid_y - 0.45, "Stimulus label", ha="center", va="top",
            fontsize=FONT_SIZE - 0.5, style="italic", color=COLOR_TEXT)

    # Two subjects (symmetric around cx)
    subj_y = 2.7
    for sx, name, report, report_color, label, label_color, bg, ec in [
        (sa_x, "Subject A", '"Fear"', fear_color,
         "Concordant", COLOR_CONC, "#e8f5e9", COLOR_CONC),
        (sb_x, "Subject B", '"Amusement"', amusement_color,
         "Discordant", COLOR_DISC, "#ffebee", COLOR_DISC),
    ]:
        ax.text(sx, subj_y, name, ha="center", va="center",
                fontsize=FONT_SIZE - 0.5, fontweight="bold", color=COLOR_TEXT)
        ax.text(sx, subj_y + 0.5, "reports:", ha="center", va="center",
                fontsize=FONT_SIZE - 0.5, color=COLOR_TEXT)
        ax.text(sx, subj_y + 1, report, ha="center", va="center",
                fontsize=FONT_SIZE - 0.5, color=report_color, fontweight="bold")
        r, g, b = mcolors.to_rgb(label_color)
        ax.text(sx, subj_y + 1.8, label, ha="center", va="center",
                fontsize=FONT_SIZE, fontweight="bold",
                color=(r, g, b, 0.75),
                bbox=dict(boxstyle="round,pad=0.3", facecolor=bg,
                          edgecolor=ec, linewidth=0.6))

    # Arrows from video to subjects (symmetric)
    arrow_kw = dict(arrowstyle="-|>", color="grey", lw=0.8)
    ax.annotate("", xy=(sa_x, subj_y - 0.3), xytext=(cx - .5, vid_y + vid_h + .25),
                arrowprops=arrow_kw)
    ax.annotate("", xy=(sb_x, subj_y - 0.3), xytext=(cx + .5, vid_y + vid_h + .25),
                arrowprops=arrow_kw)

    # Concordance ratio pie chart (centered at cx)
    from decoding.labels import load_subjective_labels
    from decoding.config import VIDEO_LABELS_9CLASS, NEUTRAL_CLASS
    subj_labels = load_subjective_labels()
    crowd = np.array(VIDEO_LABELS_9CLASS)
    non_neutral_mask = crowd != NEUTRAL_CLASS
    concordant = (subj_labels[:, non_neutral_mask] == crowd[non_neutral_mask])
    n_conc = concordant.sum()
    n_disc = (~concordant).sum()
    frac_conc = n_conc / (n_conc + n_disc)

    pie_y = subj_y + 4.5
    pie_size = 0.55

    xlim = (-0.5, 7.5)
    ylim_range = (-0.5, pie_y + 1.0)

    frac_x = (cx - xlim[0]) / (xlim[1] - xlim[0])
    frac_y_inv = 1.0 - (pie_y - ylim_range[0]) / (ylim_range[1] - ylim_range[0])

    inset = ax.inset_axes(
        [frac_x - pie_size / 2, frac_y_inv - pie_size / 2,
         pie_size, pie_size],
        transform=ax.transAxes,
    )
    inset.set_aspect("equal")
    wedges, _ = inset.pie(
        [frac_conc, 1 - frac_conc],
        colors=[COLOR_CONC, COLOR_DISC],
        startangle=90, counterclock=True,
        wedgeprops=dict(edgecolor="white", linewidth=0.8, alpha=0.6),
    )
    for wedge, pct in zip(wedges, [frac_conc, 1 - frac_conc]):
        ang = (wedge.theta2 + wedge.theta1) / 2
        x = 0.55 * np.cos(np.radians(ang))
        y = 0.55 * np.sin(np.radians(ang))
        inset.text(x, y, f"{pct:.0%}", ha="center", va="center",
                   fontsize=FONT_SIZE - 1, color="white", fontweight="bold")
    inset.axis("off")

    # Arrows from labels to pie (symmetric offsets)
    arrow_dx = 0.9
    arrow_kw2 = dict(arrowstyle="-|>", lw=0.8)
    ax.annotate("", xy=(cx - arrow_dx, pie_y - 1),
                xytext=(sa_x, subj_y + 2.3),
                arrowprops=dict(**arrow_kw2, color=COLOR_CONC))
    ax.annotate("", xy=(cx + arrow_dx, pie_y - 1),
                xytext=(sb_x, subj_y + 2.3),
                arrowprops=dict(**arrow_kw2, color=COLOR_DISC))

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim_range)
    ax.invert_yaxis()
    ax.axis("off")


def _panel_concordance_result(ax):
    """Two bars: concordant vs discordant (crowd-label classifier)."""
    from .data import load_cross_subject_data
    data = load_cross_subject_data()

    w = 0.3
    x1, x2 = 0, 0.345

    conc = data["svm_bl_conc"]
    disc = data["svm_bl_disc"]

    bar(ax, x1, conc, COLOR_CONC, width=w, alpha=0.6)
    val_conc = np.mean(conc) * 100

    bar(ax, x2, disc, COLOR_DISC, width=w, alpha=0.6)
    val_disc = np.mean(disc) * 100

    ax.text(x1, 6, f"{val_conc:.1f}", ha="center", va="center",
            fontsize=FONT_SIZE, color="white", fontweight="bold", zorder=3)
    ax.text(x2, 6, f"{val_disc:.1f}", ha="center", va="center",
            fontsize=FONT_SIZE, color="white", fontweight="bold", zorder=3)

    chance_line(ax, 1 / 9, show_label=False)

    ax.set_ylim(0, 40)
    ax.set_xlim(-0.25, 0.55)
    ax.set_xticks([x1, x2])
    ax.set_xticklabels(["Conc.", "Disc."])


def _panel_label_replacement(ax):
    """Schematic: crowd labels replaced by subjective labels."""
    left_x = 0.5
    right_x = 3.5
    header_y = 0.5

    ax.text(left_x, header_y, "Stimulus\nlabel", ha="center", va="center",
            fontsize=FONT_SIZE, fontweight="bold", color=COLOR_TEXT)
    ax.text(right_x + .5, header_y, "Subject\nreport", ha="center", va="center",
            fontsize=FONT_SIZE, fontweight="bold", color=COLOR_TEXT)

    examples = [
        ("Fear", 2, "Fear", 2),
        ("Fear", 2, "Amusement", 5),
        ("Joy", 7, "Joy", 7),
        ("Joy", 7, "Inspiration", 6),
    ]

    row_h = 1.3
    for i, (crowd_name, crowd_idx, subj_name, subj_idx) in enumerate(examples):
        y = 1.2 + i * row_h
        crowd_color = EMOTION_COLORS[crowd_idx]
        subj_color = EMOTION_COLORS[subj_idx]

        rect = mpatches.FancyBboxPatch(
            (left_x - 0.8, y - 0.3), 1.6, 0.6,
            boxstyle="round,pad=0.05",
            facecolor=crowd_color, edgecolor="none", linewidth=0.4,
        )
        ax.add_patch(rect)
        ax.text(left_x, y, crowd_name, ha="center", va="center",
                fontsize=FONT_SIZE - 0.5, color="white")

        ax.annotate("", xy=(right_x - 0.9, y), xytext=(left_x + 0.9, y),
                    arrowprops=dict(arrowstyle="-|>", color="grey", lw=0.8))

        rect2 = mpatches.FancyBboxPatch(
            (right_x - 0.8, y - 0.3), 2.7, 0.6,
            boxstyle="round,pad=0.05",
            facecolor=subj_color, edgecolor="none", linewidth=0.4,
        )
        ax.add_patch(rect2)
        ax.text(right_x + .5, y, subj_name, ha="center", va="center",
                fontsize=FONT_SIZE - 0.5, color="white")

    ax.set_xlim(-0.5, 6.5)
    ax.set_ylim(-0.3, 5.5)
    ax.invert_yaxis()
    ax.axis("off")


def _panel_subjective_result(ax):
    """Three bars: overall, concordant, discordant (subjective-label classifier)."""
    from .data import load_cross_subject_data
    data = load_cross_subject_data()

    scores = data["svm_subjective"]
    conc = data["svm_subj_conc"]
    disc = data["svm_subj_disc"]

    w = 0.35
    x0, x1, x2 = 0, 0.4, 0.8

    bar(ax, x0, scores, COLOR_BAR_A, width=w)
    bar(ax, x1, conc, COLOR_CONC, width=w, alpha=0.6)
    bar(ax, x2, disc, COLOR_DISC, width=w, alpha=0.6)

    val_overall = np.mean(scores) * 100
    val_conc = np.mean(conc) * 100
    val_disc = np.mean(disc) * 100

    ax.text(x0, 6, f"{val_overall:.1f}", ha="center", va="center",
            fontsize=FONT_SIZE, color="white", fontweight="bold", zorder=3)
    ax.text(x1, 6, f"{val_conc:.1f}", ha="center", va="center",
            fontsize=FONT_SIZE, color="white", fontweight="bold", zorder=3)
    ax.text(x2, 6, f"{val_disc:.1f}", ha="center", va="center",
            fontsize=FONT_SIZE, color="white", fontweight="bold", zorder=3)

    chance_line(ax, 1 / 9, show_label=False)

    ax.set_ylim(0, 40)
    ax.set_xlim(-0.25, 1.05)
    ax.set_xticks([x0, x1, x2])
    ax.set_xticklabels(["Overall", "Conc.", "Disc."])


def plot():
    setup()

    fig = plt.figure(figsize=(DOUBLE_COL, 2.5))
    gs = GridSpec(1, 7, figure=fig,
                 width_ratios=[1, .3, 0.8, 0.3, 1.3, 0.1, 1.2],
                 wspace=0.05)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 2])
    ax_c = fig.add_subplot(gs[0, 4])
    ax_d = fig.add_subplot(gs[0, 6])

    _panel_concordance_concept(ax_a)
    _panel_concordance_result(ax_b)
    _panel_label_replacement(ax_c)
    _panel_subjective_result(ax_d)

    panel_label(ax_a, "a", title="Concordance")
    ax_b.set_title("Accuracy (%)", fontsize=FONT_SIZE, pad=6, y=1.0)
    panel_label(ax_c, "b", title="Label replacement")
    ax_d.set_title("Accuracy (%)", fontsize=FONT_SIZE, pad=6, y=1.0)

    fig.tight_layout()
    save(fig, "fig3")
    plt.close(fig)
    return fig


if __name__ == "__main__":
    plot()
