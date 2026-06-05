"""CLISA LDS smoothing (adapted from Chen et al. 2023).

Same Kalman filter as normalization.py but operates on 256-dim CNN features.
"""
import os
import scipy.io as sio
import numpy as np
import random

from decoding.config import N_SUBJECTS, N_FOLDS, RANDOM_SEED, RESULTS_DIR


def _LDS(sequence):
    ave = np.mean(sequence, axis=0)
    u0 = ave
    X = sequence.transpose((1, 0))

    V0 = 0.01
    A = 1
    T = 0.0001
    C = 1
    sigma = 1

    [m, n] = X.shape
    P = np.zeros((m, n))
    u = np.zeros((m, n))
    V = np.zeros((m, n))
    K = np.zeros((m, n))

    K[:, 0] = (V0 * C / (C * V0 * C + sigma)) * np.ones((m,))
    u[:, 0] = u0 + K[:, 0] * (X[:, 0] - C * u0)
    V[:, 0] = (np.ones((m,)) - K[:, 0] * C) * V0

    for i in range(1, n):
        P[:, i - 1] = A * V[:, i - 1] * A + T
        K[:, i] = P[:, i - 1] * C / (C * P[:, i - 1] * C + sigma)
        u[:, i] = A * u[:, i - 1] + K[:, i] * (X[:, i] - C * A * u[:, i - 1])
        V[:, i] = (np.ones((m,)) - K[:, i] * C) * P[:, i - 1]

    return u.transpose((1, 0))


def run_lds_smoothing():
    """Apply LDS smoothing to running-normalized CLISA features."""
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    n_subs = N_SUBJECTS
    n_vids = 28
    n_length = 30
    n_spatial = 16
    n_time = 16

    save_dir = str(RESULTS_DIR / "clisa" / "runs_srt")

    for fold in range(N_FOLDS):
        print(f'=== LDS smoothing fold {fold} ===')
        fold_dir = os.path.join(save_dir, str(fold))

        out_path = os.path.join(fold_dir, 'features1_de_1s_lds.mat')
        if os.path.exists(out_path):
            print(f'  Already exists, skipping')
            continue

        data_dir = os.path.join(fold_dir, 'features1_de_1s_normTrain_rnPreWeighted0.990_play_order.mat')
        feature_de_norm = sio.loadmat(data_dir)['de']

        subs_feature_lds = np.ones((n_subs, n_vids * n_length, n_spatial * n_time))
        for sub in range(n_subs):
            for i, step in enumerate(range(0, feature_de_norm.shape[1], 30)):
                subs_feature_lds[sub, step:step+30, :] = _LDS(feature_de_norm[sub, step:step+30, :])

        sio.savemat(out_path, {'de_lds': subs_feature_lds})
        print(f'  Saved {out_path}')
