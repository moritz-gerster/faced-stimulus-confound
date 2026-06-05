"""Running normalization and LDS smoothing.

Implements the normalization pipeline from Chen et al. (2023):
1. Running normalization with exponential decay (per fold)
2. LDS (Kalman) smoothing per subject
"""
import os
import random

import numpy as np
import scipy.io as sio
from tqdm import trange

from decoding.config import (
    N_VIDEOS, N_FOLDS, WINDOWS_PER_TRIAL, N_FEATURES,
    DECAY_RATE, RANDOM_SEED, RESULTS_DIR,
    LDS_V0, LDS_A, LDS_T, LDS_C, LDS_SIGMA,
    fold_split,
)
from decoding.labels import load_video_orders


def _reorder_to_presentation(
    data: np.ndarray, vid_orders: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Reorder from canonical video order to per-subject presentation order."""
    n_subs, _, n_feat = data.shape
    wpt = WINDOWS_PER_TRIAL
    n_vids = N_VIDEOS
    reordered = np.zeros_like(data)
    order_0based = np.zeros((n_subs, n_vids), dtype=np.int32)

    for sub in range(n_subs):
        perm = (vid_orders[sub, :] - 1).astype(np.int32)
        order_0based[sub] = perm
        data_sub = data[sub].reshape(n_vids, wpt, n_feat)
        reordered[sub] = data_sub[perm].reshape(-1, n_feat)

    return reordered, order_0based


def _reorder_back(data: np.ndarray, order_0based: np.ndarray) -> np.ndarray:
    """Reorder from presentation order back to canonical video order."""
    n_subs = data.shape[0]
    n_feat = data.shape[2] if data.ndim == 3 else data.shape[-1]
    wpt = WINDOWS_PER_TRIAL
    n_vids = N_VIDEOS
    result = np.zeros_like(data)

    for sub in range(n_subs):
        data_sub = data[sub].reshape(n_vids, wpt, n_feat)
        out = np.zeros_like(data_sub)
        out[order_0based[sub]] = data_sub
        result[sub] = out.reshape(-1, n_feat)

    return result


def _lds_smooth(sequence: np.ndarray) -> np.ndarray:
    """Scalar Kalman-style LDS smoothing along the time axis.

    Args:
        sequence: shape (n_time, n_features)

    Returns:
        Smoothed array of same shape.
    """
    u0 = np.mean(sequence, axis=0)
    X = sequence.T  # (n_features, n_time)
    m, n = X.shape

    V0 = LDS_V0
    A = LDS_A
    T = LDS_T
    C = LDS_C
    sigma = LDS_SIGMA

    P = np.zeros((m, n))
    u = np.zeros((m, n))
    V = np.zeros((m, n))
    K = np.zeros((m, n))

    K[:, 0] = (V0 * C / (C * V0 * C + sigma)) * np.ones(m)
    u[:, 0] = u0 + K[:, 0] * (X[:, 0] - C * u0)
    V[:, 0] = (np.ones(m) - K[:, 0] * C) * V0

    for i in range(1, n):
        P[:, i - 1] = A * V[:, i - 1] * A + T
        K[:, i] = P[:, i - 1] * C / (C * P[:, i - 1] * C + sigma)
        u[:, i] = A * u[:, i - 1] + K[:, i] * (X[:, i] - C * A * u[:, i - 1])
        V[:, i] = (np.ones(m) - K[:, i] * C) * P[:, i - 1]

    return u.T


def run_normalization(de_features: np.ndarray) -> None:
    """Run running normalization + LDS smoothing, saving per-fold .mat files.

    Args:
        de_features: shape (123, 840, 150) from features.load_de_features()
    """
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    out_dir = RESULTS_DIR / "smooth"
    os.makedirs(out_dir, exist_ok=True)

    # Check if already computed
    if all((out_dir / f"de_lds_fold{f}.mat").exists() for f in range(N_FOLDS)):
        return

    n_subs = de_features.shape[0]
    n_total = N_VIDEOS * WINDOWS_PER_TRIAL

    vid_orders = load_video_orders()

    for fold in trange(N_FOLDS, desc="Normalization folds"):
        lds_path = out_dir / f"de_lds_fold{fold}.mat"
        if lds_path.exists():
            continue

        data = de_features.copy()

        train_sub, val_sub = fold_split(fold, n_subs)

        # Reorder to presentation order
        data, order_0based = _reorder_to_presentation(data, vid_orders)
        data[~np.isfinite(data)] = -30

        # Global stats from training subjects
        data_mean = np.mean(np.mean(data[train_sub], axis=1), axis=0)
        data_var = np.mean(np.var(data[train_sub], axis=1), axis=0)

        # Running normalization (vectorized across subjects)
        data_norm = np.zeros_like(data)
        running_sum = np.zeros((n_subs, N_FEATURES))
        running_square = np.zeros((n_subs, N_FEATURES))
        decay_factor = 1.0

        for t in range(n_total):
            x = data[:, t, :]
            running_sum += x
            running_mean = running_sum / (t + 1)
            running_square += x ** 2
            running_var = (
                (running_square - 2 * running_mean * running_sum) / (t + 1)
                + running_mean ** 2
            )
            curr_mean = decay_factor * data_mean + (1 - decay_factor) * running_mean
            curr_var = decay_factor * data_var + (1 - decay_factor) * running_var
            decay_factor *= DECAY_RATE
            data_norm[:, t, :] = (x - curr_mean) / np.sqrt(curr_var + 1e-5)

        # Reorder back to canonical video order
        data_norm = _reorder_back(data_norm, order_0based)

        # LDS smoothing
        data_lds = np.zeros_like(data_norm)
        for sub in range(n_subs):
            data_lds[sub] = _lds_smooth(data_norm[sub])

        sio.savemat(str(lds_path), {"de_lds": data_lds})
