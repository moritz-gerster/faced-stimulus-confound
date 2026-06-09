"""Shared panel building blocks for pipeline figures.

All figure scripts import from here to avoid duplication.
"""
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import numpy as np

from .style import FONT_SIZE, COLOR_TEXT, RESULTS_DIR

EMOTION_NAMES = [
    "Anger", "Disgust", "Fear", "Sadness", "Neutral",
    "Amusement", "Inspiration", "Joy", "Tenderness",
]
VIDEOS_PER_EMOTION = [3, 3, 3, 3, 4, 3, 3, 3, 3]
EMOTION_COLORS = [
    '#e05549', '#e6d44d', '#9b8ec4', '#5a9abf', '#a0a0a0',
    '#e8943a', '#5dbdaa', '#8dc450', '#f0a0c0',
]
# Weaker variant:
# EMOTION_COLORS = [
#     '#fb8072', '#ffffb3', '#bebada', '#80b1d3', '#bfbfbf',
#     '#fdb462', '#8dd3c7', '#b3de69', '#fccde5',
# ]
COLOR_REMOVED = (0, 0, 0, 0.04)

KEPT_MASK_SINGLE_VIDEO = []
for _n_vids in VIDEOS_PER_EMOTION:
    KEPT_MASK_SINGLE_VIDEO.extend([True] + [False] * (_n_vids - 1))


def load_scores(name):
    """Load scores from scores.npy or predictions.npz."""
    npy = RESULTS_DIR / name / "scores.npy"
    if npy.exists():
        return np.load(npy)
    npz = RESULTS_DIR / name / "predictions.npz"
    if npz.exists():
        return np.load(str(npz))["scores"]
    raise FileNotFoundError(
        f"No scores found in {RESULTS_DIR / name} "
        f"(checked scores.npy and predictions.npz)"
    )


def panel_cv_classify(ax, kept_mask=None, dummy_fracs=None,
                      show_emotion_labels=False):
    """Classification pipeline: CV split bars -> classifier -> pie charts.

    Parameters
    ----------
    kept_mask : list of bool (length 28), optional
        Which videos are active (colored + connected). None = all kept.
    dummy_fracs : list of float (length 9), optional
        Illustrative per-emotion fractions for pie charts (not real data).
    show_emotion_labels : bool, optional
        If True, draw emotion names to the left of the CV bars.
    """
    if dummy_fracs is None:
        dummy_fracs = [0.65, 0.55, 0.60, 0.58, 0.70, 0.62, 0.50, 0.55, 0.60]

    total_videos = sum(VIDEOS_PER_EMOTION)
    bar_h = 1.0
    total_h = total_videos * bar_h

    # Layout x-coordinates
    n_folds = 10
    seg_w = 0.28
    cv_x = 0
    cv_w = n_folds * seg_w
    test_fold = 7
    gap = 0.5
    test_col_x = cv_x + cv_w + gap
    test_col_w = seg_w
    box_cx = test_col_x + test_col_w + 5.0
    box_w = 4
    box_h = 6
    pie_x = box_cx + box_w / 2 + 3.5
    pie_r = 1.0

    # --- Left column: all 28 videos as CV split bars ---
    video_y_centers = []
    vid_idx = 0
    y = 0
    for emo_idx, n_vids in enumerate(VIDEOS_PER_EMOTION):
        color = EMOTION_COLORS[emo_idx]
        base_rgb = mcolors.to_rgb(color)
        for v in range(n_vids):
            kept = kept_mask[vid_idx] if kept_mask is not None else True
            for s in range(n_folds):
                is_test = (s == test_fold)
                if kept:
                    fc = color if is_test else tuple(c * 0.25 + 0.75 for c in base_rgb)
                else:
                    fc = COLOR_REMOVED
                rect = mpatches.Rectangle(
                    (cv_x + s * seg_w, y), seg_w - 0.02, bar_h * 0.85,
                    facecolor=fc, edgecolor="white", linewidth=0.15,
                )
                ax.add_patch(rect)
            # Test column
            tc = color if kept else COLOR_REMOVED
            rect_test = mpatches.Rectangle(
                (test_col_x, y), test_col_w - 0.02, bar_h * 0.85,
                facecolor=tc, edgecolor="white", linewidth=0.15,
            )
            ax.add_patch(rect_test)
            if kept:
                video_y_centers.append(y + bar_h * 0.85 / 2)
            y += bar_h
            vid_idx += 1

    # --- Emotion labels (left of CV bars) ---
    if show_emotion_labels:
        vid_idx_lbl = 0
        y_lbl = 0
        for emo_idx, (name, n_vids) in enumerate(
            zip(EMOTION_NAMES, VIDEOS_PER_EMOTION)
        ):
            group_center_y = y_lbl + n_vids * bar_h / 2
            ax.text(cv_x - 0.3, group_center_y, name, ha="right", va="center",
                    fontsize=FONT_SIZE - 0.5, color="k")
            y_lbl += n_vids * bar_h
            vid_idx_lbl += n_vids

    # --- Center: Classifier box ---
    box_cy = total_h / 2
    box_x0 = box_cx - box_w / 2
    box_y0 = box_cy - box_h / 2
    rect = mpatches.FancyBboxPatch(
        (box_x0, box_y0), box_w, box_h,
        boxstyle="round,pad=0.2",
        facecolor="white", edgecolor="grey", linewidth=1.2,
    )
    ax.add_patch(rect)
    ax.text(box_cx, box_cy, "Classifier", ha="center", va="center",
            fontsize=FONT_SIZE - 1, color=COLOR_TEXT)

    # --- Right: 9 pie charts (inset axes) ---
    n_emotions = 9
    pie_spacing = total_h / n_emotions
    pie_y_centers = []
    pie_size = 0.085

    xlim = (-2.0 if show_emotion_labels else -0.5, pie_x + pie_r + 1.5)
    ylim = (-1.0, total_h + 0.5)

    for emo_idx in range(n_emotions):
        cy = emo_idx * pie_spacing + pie_spacing / 2
        pie_y_centers.append(cy)
        color = EMOTION_COLORS[emo_idx]
        frac = dummy_fracs[emo_idx]

        frac_x = (pie_x - xlim[0]) / (xlim[1] - xlim[0])
        frac_y = 1.0 - (cy - ylim[0]) / (ylim[1] - ylim[0])

        inset = ax.inset_axes(
            [frac_x - pie_size / 2, frac_y - pie_size / 2,
             pie_size, pie_size],
            transform=ax.transAxes,
        )
        inset.set_aspect("equal")
        inset.pie(
            [frac, 1 - frac], colors=[color, "#e0e0e0"],
            startangle=90, counterclock=False,
            wedgeprops=dict(edgecolor="white", linewidth=0.4),
        )
        inset.axis("off")

    # --- Connecting lines: test column -> box ---
    line_kw = dict(color="grey", linewidth=0.15, zorder=0)
    box_left = box_cx - box_w / 2
    for vy in video_y_centers:
        ax.plot([test_col_x + test_col_w, box_left], [vy, box_cy], **line_kw)

    # --- Connecting lines: box -> pies ---
    box_right = box_cx + box_w / 2
    for emo_idx in range(n_emotions):
        pie_cy = pie_y_centers[emo_idx]
        # One line per kept video in this emotion
        n_kept = sum(1 for i in range(sum(VIDEOS_PER_EMOTION[:emo_idx]),
                                       sum(VIDEOS_PER_EMOTION[:emo_idx + 1]))
                     if (kept_mask is None or kept_mask[i]))
        for _ in range(n_kept):
            ax.plot([box_right, pie_x], [box_cy, pie_cy], **line_kw)

    # Duration annotations below video bars (tick-style distance markers)
    if show_emotion_labels:
        ann_y = total_h + 0.3
        tick_h = 0.5
        lw = 0.4
        # 30 s: tick-style distance marker
        x_left, x_right = cv_x, cv_x + cv_w - 0.02
        mid = (x_left + x_right) / 2
        ax.plot([x_left, x_left], [ann_y, ann_y + tick_h],
                color="black", lw=lw, clip_on=False)
        ax.plot([x_right, x_right], [ann_y, ann_y + tick_h],
                color="black", lw=lw, clip_on=False)
        ax.plot([x_left, x_right], [ann_y + tick_h / 2, ann_y + tick_h / 2],
                color="black", lw=lw, clip_on=False)
        ax.text(mid, ann_y + tick_h + 0.2, "30 s", ha="center", va="top",
                    fontsize=5, color=COLOR_TEXT)

        # 3 s: inward-pointing arrows
        x_left, x_right = test_col_x + 0.04, test_col_x + test_col_w - 0.04
        mid = (x_left + x_right) / 2
        arrow_y = ann_y + tick_h / 2
        arrow_kw = dict(arrowstyle="->,head_length=0.15,head_width=0.1",
                        color="black", lw=lw,
                        shrinkA=0, shrinkB=0)
        ax.annotate("", xy=(x_right, arrow_y),
                    xytext=(x_right + 0.3, arrow_y), arrowprops=arrow_kw)
        ax.annotate("", xy=(x_left, arrow_y),
                    xytext=(x_left - 0.3, arrow_y), arrowprops=arrow_kw)
        ax.text(mid, ann_y + tick_h + 0.2, "3 s", ha="center", va="top",
                    fontsize=5, color=COLOR_TEXT)

    # Axis setup
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.invert_yaxis()
    ax.axis("off")


N_SUBJECTS_SHOWN = 10
TEST_SUBJECT_IDX = 9


def panel_cross_subject_pipeline(ax, kept_mask=None, dummy_fracs=None):
    """Cross-subject CV pipeline: subjects -> Train/Test box -> pie charts.

    Parameters
    ----------
    kept_mask : list of bool (length 28), optional
        Which videos to show in full color. None = all kept.
    dummy_fracs : list of float (length 9), optional
        Illustrative per-emotion fractions for pie charts (not real data).
    """
    if dummy_fracs is None:
        dummy_fracs = [0.25, 0.6, 0.42, 0.28, 0.50, 0.44, 0.25, 0.3, 0.42]

    n_videos = sum(VIDEOS_PER_EMOTION)
    vid_w = 0.35
    vid_h = 1.6
    subj_gap = 0.6
    subj_row_w = n_videos * vid_w

    subj_y_starts = []
    y = 0
    for s in range(N_SUBJECTS_SHOWN):
        subj_y_starts.append(y)
        y += vid_h + subj_gap
    total_h = y - subj_gap

    subj_y_centers = []
    for s in range(N_SUBJECTS_SHOWN):
        is_test = (s == TEST_SUBJECT_IDX)
        sy = subj_y_starts[s]
        vid_idx = 0
        for emo_idx, n_vids in enumerate(VIDEOS_PER_EMOTION):
            color = EMOTION_COLORS[emo_idx]
            base_rgb = mcolors.to_rgb(color)
            for v in range(n_vids):
                vx = vid_idx * vid_w
                if kept_mask is not None and not kept_mask[vid_idx]:
                    fc = COLOR_REMOVED
                elif is_test:
                    fc = color
                else:
                    fc = tuple(c * 0.35 + 0.65 for c in base_rgb)
                rect = mpatches.FancyBboxPatch(
                    (vx, sy), vid_w * 0.85, vid_h * 0.9,
                    boxstyle="round,pad=0.01",
                    facecolor=fc, edgecolor="white", linewidth=0.3,
                )
                ax.add_patch(rect)
                vid_idx += 1
        subj_y_centers.append(sy + vid_h / 2)

    for s in range(N_SUBJECTS_SHOWN):
        ax.text(-3.0, subj_y_centers[s], f"Subject {s+1}", ha="left",
                va="center", fontsize=FONT_SIZE - 1, color="k")

    # Classifier box with horizontal divider
    box_x = subj_row_w + 2.5
    box_w = 2.5
    box_h = 6
    mid_y = total_h / 2.119
    box_y = mid_y - box_h / 2

    rect_box = mpatches.FancyBboxPatch(
        (box_x, box_y), box_w, box_h,
        boxstyle="round,pad=0",
        facecolor="white", edgecolor="grey", linewidth=1.2,
    )
    ax.add_patch(rect_box)

    ax.plot([box_x, box_x + box_w], [mid_y, mid_y],
            color="grey", linewidth=0.8)

    ax.text(box_x + box_w / 2, mid_y - box_h * 0.25,
            "Train", ha="center", va="center", fontsize=FONT_SIZE - 1,
            color=COLOR_TEXT)
    ax.text(box_x + box_w / 2, mid_y + box_h * 0.25,
            "Test", ha="center", va="center", fontsize=FONT_SIZE - 1,
            color=COLOR_TEXT)

    # Connecting lines
    train_cy = mid_y - box_h * 0.25
    test_cy = mid_y + box_h * 0.25
    box_bottom = box_y + box_h
    box_cx = box_x + box_w / 2

    # Set to True to connect test subject to left edge of Test box;
    # set to False to connect to center-bottom of box.
    TEST_LINE_TO_LEFT = True

    for s in range(N_SUBJECTS_SHOWN):
        is_test = (s == TEST_SUBJECT_IDX)
        if is_test:
            if TEST_LINE_TO_LEFT:
                target_x, target_y = box_x, test_cy
            else:
                target_x, target_y = box_cx, box_bottom
            linewidth = 0.4
            alpha = 1.0
        else:
            target_x, target_y = box_x, train_cy
            linewidth = 0.25
            alpha = 0.2
        ax.plot([subj_row_w, target_x], [subj_y_centers[s], target_y],
                color="k", linewidth=linewidth, alpha=alpha, zorder=0)

    # Pie charts
    pie_x = box_x + box_w + 4.5
    n_emotions = 9
    pie_spacing = total_h / n_emotions
    pie_y_centers = []
    pie_size = 0.075

    xlim = (-1.4, pie_x + 1.5)
    ylim = (-1.5, total_h + 1.5)

    for emo_idx in range(n_emotions):
        cy = emo_idx * pie_spacing + pie_spacing / 2
        pie_y_centers.append(cy)
        color = EMOTION_COLORS[emo_idx]
        frac = dummy_fracs[emo_idx]

        frac_x = (pie_x - xlim[0]) / (xlim[1] - xlim[0])
        frac_y = 1.0 - (cy - ylim[0]) / (ylim[1] - ylim[0])

        inset = ax.inset_axes(
            [frac_x - pie_size / 2, frac_y - pie_size / 2,
             pie_size, pie_size],
            transform=ax.transAxes,
        )
        inset.set_aspect("equal")
        inset.pie(
            [frac, 1 - frac], colors=[color, "#e0e0e0"],
            startangle=90, counterclock=False,
            wedgeprops=dict(edgecolor="white", linewidth=0.4),
        )
        inset.axis("off")

    # Lines: Test half of box -> pies
    box_right = box_x + box_w
    for cy in pie_y_centers:
        ax.plot([box_right, pie_x], [test_cy, cy],
                color="k", linewidth=0.4, zorder=0)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.invert_yaxis()
    ax.axis("off")


