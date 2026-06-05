"""CLISA data loading (adapted from Chen et al. 2023).

Path changes: uses DATA_DIR from config instead of hardcoded './Clisa_data'.
Only cls9 (9-class) label type is supported (cls2/cls3 branches removed).
"""
import numpy as np
import scipy.io as sio
import os
import pickle

from decoding.config import (
    DATA_DIR, N_SUBJECTS, N_VIDEOS, N_CHANNELS, CHANNELS_TO_DROP,
)


CLISA_DIR = DATA_DIR / "Clisa_data"


def _make_label_cls9(n_vids=28):
    """9-class label list: 3,3,3,3,4,3,3,3,3 videos per class."""
    label = [0] * 3
    for i in range(1, 4):
        label.extend([i] * 3)
    label.extend([4] * 4)
    for i in range(5, 9):
        label.extend([i] * 3)
    return label[:n_vids]


def load_srt_raw_newPre(timeLen, timeStep, fs, channel_norm, time_norm):
    n_channs = N_CHANNELS
    n_points = 7500
    n_segs = int((n_points/fs - timeLen) / timeStep + 1)

    data_paths = sorted(f for f in os.listdir(CLISA_DIR) if f.endswith('.pkl'))
    if len(data_paths) != N_SUBJECTS:
        raise FileNotFoundError(
            f"Expected {N_SUBJECTS} .pkl files in {CLISA_DIR}, "
            f"found {len(data_paths)}"
        )
    n_vids = N_VIDEOS
    chn = N_CHANNELS
    fs = 250
    sec = 30

    data = np.zeros((len(data_paths), n_vids, chn, fs * sec))

    for idx, path in enumerate(data_paths):
        with open(os.path.join(CLISA_DIR, path), 'rb') as f:
            data_sub = pickle.load(f)
        data[idx,:,:,:] = data_sub[:, :-CHANNELS_TO_DROP, :]

    n_subs = data.shape[0]
    n_videos = N_VIDEOS

    data = np.transpose(data, (0,1,3,2)).reshape(n_subs, -1, n_channs)

    if channel_norm:
        for i in range(data.shape[0]):
            data[i,:,:] = (data[i,:,:] - np.mean(data[i,:,:], axis=0)) / np.std(data[i,:,:], axis=0)

    if time_norm:
        data = (data - np.tile(np.expand_dims(np.mean(data, axis=2), 2), (1, 1, data.shape[2]))) / np.tile(
            np.expand_dims(np.std(data, axis=2), 2), (1, 1, data.shape[2])
        )

    n_samples = np.ones(n_videos) * n_segs

    label = _make_label_cls9()
    label_repeat = []
    for i in range(len(label)):
        label_repeat = label_repeat + [label[i]]*n_segs

    return data, label_repeat, n_samples, n_segs


def load_srt_pretrainFeat(datadir, channel_norm, timeLen, timeStep, isFilt, filtLen, label_type):
    n_samples = np.ones(N_VIDEOS).astype(np.int32) * 30

    for i in range(len(n_samples)):
        n_samples[i] = int((n_samples[i] - timeLen) / timeStep + 1)

    if datadir[-4:] == '.npy':
        data = np.load(datadir)
        data[data < -10] = -5
    elif datadir[-4:] == '.mat':
        data = sio.loadmat(datadir)['de_lds']
        data[np.isnan(data)] = -8

    if channel_norm:
        for i in range(data.shape[0]):
            data[i,:,:] = (data[i,:,:] - np.mean(data[i,:,:], axis=0)) / (np.std(data[i,:,:], axis=0) + 1e-3)

    label = _make_label_cls9()
    label_repeat = []
    for i in range(len(label)):
        label_repeat = label_repeat + [label[i]]*n_samples[i]
    return data, label_repeat, n_samples
