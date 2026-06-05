"""Label loading for FACED dataset.

Provides crowd (video-assigned) labels and subjective (self-report argmax) labels.
Also loads per-subject video presentation orders for normalization.
"""
import os

import numpy as np
import scipy.io as sio

from decoding.config import (
    REMARKS_DIR, N_SUBJECTS, N_VIDEOS,
    SCORE_INDEX_TO_CLASS,
)


def _load_all_remarks():
    """Load After_remark arrays for all subjects."""
    subject_dirs = sorted(
        d for d in os.listdir(REMARKS_DIR)
        if os.path.isdir(REMARKS_DIR / d)
    )
    return [
        sio.loadmat(str(REMARKS_DIR / d / "After_remarks.mat"))["After_remark"]
        for d in subject_dirs
    ]


def load_subjective_labels() -> np.ndarray:
    """Per-subject labels from argmax of self-report emotion ratings.

    Returns:
        Array of shape (123, 28) with class IDs 0-8 per subject/video.
    """
    labels = np.zeros((N_SUBJECTS, N_VIDEOS), dtype=int)
    for sub_idx, ar in enumerate(_load_all_remarks()):
        for i in range(ar.shape[0]):
            row = ar[i][0]
            scores = row[0].flatten()[:8]
            vid = int(row[2].item()) - 1  # 1-based → 0-based
            peak_idx = int(np.argmax(scores))
            labels[sub_idx, vid] = SCORE_INDEX_TO_CLASS[peak_idx]
    return labels


def load_video_orders() -> np.ndarray:
    """Load per-subject video presentation orders from After_remarks.mat.

    Returns:
        Array of shape (123, 28) with 1-based video IDs in presentation order.
    """
    remarks = _load_all_remarks()
    vid_orders = np.zeros((len(remarks), N_VIDEOS))
    for idx, ar in enumerate(remarks):
        vid_orders[idx, :] = [ar[vid][0][2].item() for vid in range(N_VIDEOS)]
    return vid_orders
