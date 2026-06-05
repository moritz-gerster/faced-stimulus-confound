"""Shared style constants and helpers for publication figures.

Follows Nature Scientific Data guidelines:
- Clear sans-serif typeface (Helvetica)
- White background, no chartjunk
- Error bars with statistical treatment described in legend
- Bold lowercase panel labels (a, b, c, ...)
"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.patheffects as pe
import numpy as np
from matplotlib.transforms import blended_transform_factory, offset_copy
from scipy.stats import sem

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"

# Typography
FONT_FAMILY = "Helvetica"
FONT_SIZE = 7
LABEL_SIZE = 8
TITLE_SIZE = 8
PANEL_LABEL_SIZE = 11

# Colors
COLOR_SVM = "#4878CF"
COLOR_CLISA = "#D4642C"
COLOR_CHANCE = "#333333"
COLOR_BAR_A = (0, 0, 0, 0.3)
COLOR_BAR_B = (0, 0, 0, 0.5)
COLOR_TEXT = (0, 0, 0, 0.65)
COLOR_CONC = "#2d8632"
COLOR_DISC = "#c62828"

# Figure width (Nature Sci Data: single col ~88 mm, double col ~180 mm)
DOUBLE_COL = 7.09  # inches ≈ 180 mm

# Bar plot defaults
BAR_WIDTH = 0.3
ERROR_KW = dict(capsize=3, elinewidth=0.8, capthick=0.8)


def setup():
    """Apply publication-ready matplotlib defaults."""
    mpl.rcParams.update({
        "font.family": FONT_FAMILY,
        "font.size": FONT_SIZE,
        "axes.labelsize": LABEL_SIZE,
        "axes.titlesize": TITLE_SIZE,
        "xtick.labelsize": FONT_SIZE,
        "ytick.labelsize": FONT_SIZE,
        "legend.fontsize": FONT_SIZE,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "figure.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    })


def panel_label(ax, label, title=None, x=-0.18, y=1.092):
    """Add bold lowercase panel label (a, b, c, ...) and optional title."""
    ax.text(
        x, y, label,
        transform=ax.transAxes,
        fontsize=PANEL_LABEL_SIZE,
        fontweight="bold",
        va="baseline",
        ha="left",
        path_effects=[pe.withStroke(linewidth=0.6, foreground="black")],
    )
    if title:
        title_tf = offset_copy(
            ax.transAxes, fig=ax.figure,
            x=PANEL_LABEL_SIZE + 5, y=-0.3, units="points",
        )
        ax.text(
            x, y, title,
            transform=title_tf,
            fontsize=FONT_SIZE,
            va="baseline",
            ha="left",
        )


def chance_line(ax, level, show_label=True, label_x=0.5, label_ha="center"):
    """Add horizontal dashed chance-level line, optionally with text label."""
    ax.axhline(level * 100, color=COLOR_CHANCE, lw=.6, zorder=10,
               dashes=(4, 3), dash_capstyle="butt")
    if show_label:
        trans = blended_transform_factory(ax.transAxes, ax.transData)
        ax.text(
            label_x, level * 100 + 1.2,
            f"Chance ({level * 100:.1f}%)",
            transform=trans,
            fontsize=FONT_SIZE - 1,
            color=COLOR_CHANCE,
            ha=label_ha,
            va="bottom",
            style="italic",
        )


def bar(ax, x, scores, color, width=BAR_WIDTH, hatch=None, alpha=None):
    """Plot a single bar with SEM error bar (upper only, top cap only)."""
    mean = np.mean(scores) * 100
    err = sem(scores) * 100
    ax.bar(
        x, mean,
        width=width, color=color, edgecolor="white", linewidth=0.5,
        hatch=hatch, alpha=alpha, zorder=2,
    )
    _, caps, _ = ax.errorbar(
        x, mean, yerr=[[0], [err]],
        fmt="none", ecolor="grey", lw=ERROR_KW["elinewidth"],
        capsize=ERROR_KW["capsize"], capthick=ERROR_KW["capthick"],
        zorder=3,
    )
    caps[0].set_visible(False)




def save(fig, name):
    """Save figure as PDF and PNG."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUTPUT_DIR / f"{name}.{ext}")
    print(f"Saved {name}.pdf/.png → {OUTPUT_DIR}")
