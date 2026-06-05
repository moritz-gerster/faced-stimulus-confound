"""CLISA running normalization (adapted from Chen et al. 2023).

Same concept as normalization.py but operates on 256-dim CNN features.
"""
import os
import scipy.io as sio
import numpy as np
import random

from decoding.config import N_SUBJECTS, N_FOLDS, RANDOM_SEED, RESULTS_DIR, fold_split
from .reorder_vids import video_order_load, reorder_vids, reorder_vids_back


def run_running_norm():
    """Apply running normalization to extracted CLISA features."""
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    n_vids = 28
    n_subs = N_SUBJECTS
    n_total = 30 * n_vids
    decay_rate = 0.990

    save_dir = str(RESULTS_DIR / "clisa" / "runs_srt")
    vid_order = video_order_load(n_vids)

    for fold in range(N_FOLDS):
        print(f'=== Running norm fold {fold} ===')
        fold_dir = os.path.join(save_dir, str(fold))

        out_path = os.path.join(fold_dir, 'features1_de_1s_normTrain_rnPreWeighted0.990_play_order.mat')
        if os.path.exists(out_path):
            print(f'  Already exists, skipping')
            continue

        data = sio.loadmat(os.path.join(fold_dir, 'features1_de_1s_normTrain.mat'))['de']

        train_sub, val_sub = fold_split(fold)

        data, vid_play_order_new = reorder_vids(data, vid_order)
        data[np.isnan(data)] = -30

        data_mean = np.mean(np.mean(data[train_sub, :, :], axis=1), axis=0)
        data_var = np.mean(np.var(data[train_sub, :, :], axis=1), axis=0)

        data_norm = np.zeros_like(data)
        for sub in range(data.shape[0]):
            running_sum = np.zeros(data.shape[-1])
            running_square = np.zeros(data.shape[-1])
            decay_factor = 1.
            for counter in range(n_total):
                data_one = data[sub, counter:counter+1, :]
                running_sum = running_sum + data_one
                running_mean = running_sum / (counter+1)
                running_square = running_square + data_one**2
                running_var = (running_square - 2 * running_mean * running_sum) / (counter+1) + running_mean**2

                curr_mean = decay_factor*data_mean + (1-decay_factor)*running_mean
                curr_var = decay_factor*data_var + (1-decay_factor)*running_var
                decay_factor = decay_factor*decay_rate

                data_one = (data_one - curr_mean) / np.sqrt(curr_var + 1e-5)
                data_norm[sub, counter:counter+1, :] = data_one

        data_norm = reorder_vids_back(data_norm, vid_play_order_new)
        sio.savemat(out_path, {'de': data_norm})
        print(f'  Saved {out_path}')
