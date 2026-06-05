"""CLISA feature extraction (adapted from Chen et al. 2023 extract_pretrainFeat.py).

Passes data through pretrained ConvNet to extract 256-dim DE features per fold.
"""
import numpy as np
import torch
import os
import scipy.io as sio
import pickle
from torch.utils.data import DataLoader

import torch.nn as nn
import torch.nn.functional as F

from decoding.config import N_SUBJECTS, N_FOLDS, RESULTS_DIR, fold_split
from . import _init_torch
from .io_utils import EmotionDataset
from .load_data import load_srt_raw_newPre
from .model import stratified_layerNorm


class _ConvNet_extract(nn.Module):
    """ConvNet variant for feature extraction (returns intermediate output)."""
    def __init__(self, n_spatialFilters, n_timeFilters, timeFilterLen, n_channs, stratified, multiFact):
        super(_ConvNet_extract, self).__init__()
        self.spatialConv = nn.Conv2d(1, n_spatialFilters, (n_channs, 1))
        self.timeConv = nn.Conv2d(1, n_timeFilters, (1, timeFilterLen), padding=(0, (timeFilterLen-1)//2))
        self.avgpool = nn.AvgPool2d((1, 30))
        self.spatialConv2 = nn.Conv2d(n_timeFilters, n_timeFilters*multiFact, (n_spatialFilters, 1), groups=n_timeFilters)
        self.timeConv2 = nn.Conv2d(n_timeFilters*multiFact, n_timeFilters*multiFact*multiFact, (1, 6), groups=n_timeFilters*multiFact)
        self.n_spatialFilters = n_spatialFilters
        self.n_timeFilters = n_timeFilters
        self.stratified = stratified

    def forward(self, input):
        if 'initial' in self.stratified:
            input = stratified_layerNorm(input, input.shape[0])
        out = self.spatialConv(input)
        out = out.permute(0,2,1,3)
        out = self.timeConv(out)
        out1 = out.clone()
        out = F.elu(out)
        out = self.avgpool(out)
        if 'middle1' in self.stratified:
            out = stratified_layerNorm(out, out.shape[0])
        out = F.elu(self.spatialConv2(out))
        out = F.elu(self.timeConv2(out))
        if 'middle2' in self.stratified:
            out = stratified_layerNorm(out, out.shape[0])
        return out, out1


def run_extract():
    """Extract 256-dim DE features from pretrained ConvNet for all folds."""
    device = _init_torch()

    n_spatialFilters = 16
    n_timeFilters = 16
    timeFilterLen = 60
    n_channs = 30
    multiFact = 2
    timeLen = 1
    timeStep = 1
    fs = 250

    data, label_repeat, n_samples, n_segs = load_srt_raw_newPre(
        timeLen, timeStep, fs, channel_norm=False, time_norm=False
    )

    n_subs = N_SUBJECTS
    n_total = int(np.sum(n_samples))
    save_dir = str(RESULTS_DIR / "clisa" / "runs_srt")

    for fold in range(N_FOLDS):
        print(f'=== Extract features fold {fold} ===')

        model = _ConvNet_extract(
            n_spatialFilters, n_timeFilters, timeFilterLen,
            n_channs, stratified=[], multiFact=multiFact
        ).to(device)

        fold_dir = os.path.join(save_dir, str(fold))
        results_path = os.path.join(save_dir, f'folds_{fold}_dataset_both_results_pretrain.pkl')
        if not os.path.exists(results_path):
            results_path = os.path.join(save_dir, 'folds_all_dataset_both_results_pretrain.pkl')

        with open(results_path, 'rb') as f:
            results_pretrain = pickle.load(f)
        best_pretrain_epoch = int(results_pretrain['best_epoch'][fold])
        checkpoint_name = 'checkpoint_{:04d}.pth.tar'.format(best_pretrain_epoch)
        checkpoint = torch.load(os.path.join(fold_dir, checkpoint_name), map_location=device)
        model.load_state_dict(checkpoint['state_dict'], strict=False)

        train_sub, val_sub = fold_split(fold)

        # Normalize all data using training statistics
        data_fold = data.copy()
        data_mean = np.mean(np.mean(data_fold[train_sub, :, :], axis=1), axis=0)
        data_var = np.mean(np.var(data_fold[train_sub, :, :], axis=1), axis=0)
        for i in range(n_subs):
            data_fold[i,:,:] = (data_fold[i,:,:] - data_mean) / np.sqrt(data_var + 1e-5)

        features1_de = np.zeros((n_subs, n_total, n_timeFilters, n_spatialFilters))
        for sub in range(n_subs):
            data_val = data_fold[sub, :, :]
            label_val_arr = np.array(label_repeat)
            valset = EmotionDataset(data_val, label_val_arr, timeLen, timeStep, n_segs, fs)
            val_loader = DataLoader(dataset=valset, batch_size=1, pin_memory=True, num_workers=8, shuffle=False)

            for counter, (x_batch, y_batch) in enumerate(val_loader):
                x_batch = x_batch.to(device)
                _, out = model(x_batch)
                out = out.detach().cpu().numpy()
                de = 0.5*np.log(2*np.pi*np.exp(1)*(np.var(out, 3)))
                if (counter + 1) < n_total:
                    features1_de[sub, counter, :, :] = de
                else:
                    features1_de[sub, counter, :, :] = de

        features1_de = features1_de.reshape(n_subs, n_total, 256)
        de = {'de': features1_de}
        out_path = os.path.join(fold_dir, 'features1_de_1s_normTrain.mat')
        sio.savemat(out_path, de)
        print(f'Saved {out_path}')
