"""Video reordering utilities (adapted from Chen et al. 2023).

Path changes: uses REMARKS_DIR from config instead of hardcoded './After_remarks'.
Replaced hdf5storage with scipy.io (already used elsewhere in this project).
"""
import scipy.io as sio
import numpy as np
import os

from decoding.config import REMARKS_DIR


def video_order_load(n_vids=28):
    filesPath = sorted(
        d for d in os.listdir(REMARKS_DIR)
        if os.path.isdir(REMARKS_DIR / d)
    )
    vid_orders = np.zeros((len(filesPath), n_vids))
    for idx, file in enumerate(filesPath):
        remark_file = os.path.join(REMARKS_DIR, file, 'After_remarks.mat')
        subject_remark = sio.loadmat(remark_file)['After_remark']
        vid_orders[idx, :] = [subject_remark[vid][0][2].item() for vid in range(0, n_vids)]
    return vid_orders


def reorder_vids(data, vid_play_order):
    n_vids = int(data.shape[1] / 30)
    n_subs = data.shape[0]
    vid_play_order_copy = vid_play_order.copy()

    if n_vids == 24:
        vid_play_order_new = np.zeros((n_subs, n_vids)).astype(np.int32)
        data_reorder = np.zeros_like(data)
        for sub in range(n_subs):
            tmp = vid_play_order_copy[sub,:]
            tmp = tmp[(tmp<13)|(tmp>16)]
            tmp[tmp>=17] = tmp[tmp>=17] - 4
            tmp = tmp - 1
            vid_play_order_new[sub, :] = tmp
            data_sub = data[sub, :, :].reshape(n_vids, 30, data.shape[-1])
            tmp = [int(i) for i in tmp]
            data_sub = data_sub[tmp, :, :]
            data_reorder[sub, :, :] = data_sub.reshape(n_vids*30, data.shape[-1])

    elif n_vids == 28:
        vid_play_order_new = np.zeros((n_subs, n_vids)).astype(np.int32)
        data_reorder = np.zeros_like(data)
        for sub in range(n_subs):
            tmp = vid_play_order_copy[sub,:] - 1
            vid_play_order_new[sub, :] = tmp
            data_sub = data[sub, :, :].reshape(n_vids, 30, data.shape[-1])
            tmp = [int(i) for i in tmp]
            data_sub = data_sub[tmp, :, :]
            data_reorder[sub, :, :] = data_sub.reshape(n_vids*30, data.shape[-1])

    return data_reorder, vid_play_order_new


def reorder_vids_back(data, vid_play_order_new):
    n_vids = int(data.shape[1] / 30)
    n_subs = data.shape[0]
    data_back = np.zeros((n_subs, n_vids, 30, data.shape[-1]))
    for sub in range(n_subs):
        data_sub = data[sub, :, :].reshape(n_vids, 30, data.shape[-1])
        data_back[sub, vid_play_order_new[sub, :], :, :] = data_sub
    data_back = data_back.reshape(n_subs, n_vids*30, data.shape[-1])
    return data_back
