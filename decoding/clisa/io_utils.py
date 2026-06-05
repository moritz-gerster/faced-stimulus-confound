"""CLISA I/O utilities (from Chen et al. 2023, unchanged)."""
import os
import numpy as np
from torch.utils.data import Dataset
import torch
import shutil
import random


class DEDataset(Dataset):
    def __init__(self, data, label):
        self.data = torch.FloatTensor(data)
        self.label = torch.from_numpy(label)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]


class EmotionDataset(Dataset):
    def __init__(self, data, label, timeLen, timeStep, n_segs, fs, transform=None):
        self.data = data.transpose()
        self.timeLen = timeLen
        self.timeStep = timeStep
        self.n_segs = n_segs
        self.fs = fs
        self.transform = transform
        self.label = torch.from_numpy(label)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        n_samples_remain_each = 30 - self.n_segs * self.timeStep
        one_seq = self.data[:, int((idx * self.timeStep + n_samples_remain_each * np.floor(idx / self.n_segs)) * self.fs):
                            int((idx * self.timeStep + self.timeLen + n_samples_remain_each * np.floor(idx / self.n_segs)) * self.fs)]
        one_label = self.label[idx]
        if self.transform:
            one_seq = self.transform(one_seq)
        one_seq = torch.FloatTensor(one_seq).unsqueeze(0)
        return one_seq, one_label


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def save_config_file(model_checkpoints_folder, args):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)


class TrainSampler():
    def __init__(self, n_subs, n_times, batch_size, n_samples):
        self.n_per = int(np.sum(n_samples))
        self.n_subs = n_subs
        self.batch_size = batch_size
        self.n_samples_cum = np.concatenate((np.array([0]), np.cumsum(n_samples)))
        self.n_samples_per_trial = int(batch_size / len(n_samples))

        self.sub_pairs = []
        for i in range(self.n_subs):
            for j in range(i+1, self.n_subs):
                self.sub_pairs.append([i, j])
        random.shuffle(self.sub_pairs)
        self.n_times = n_times

    def __len__(self):
        return self.n_times * len(self.sub_pairs)

    def __iter__(self):
        for s in range(len(self.sub_pairs)):
            for t in range(self.n_times):
                [sub1, sub2] = self.sub_pairs[s]

                ind_abs = np.zeros(0)
                if self.batch_size < len(self.n_samples_cum)-1:
                    sel_vids = np.random.choice(np.arange(len(self.n_samples_cum)-1), self.batch_size)
                    for i in sel_vids:
                        ind_one = np.random.choice(np.arange(self.n_samples_cum[i], self.n_samples_cum[i+1]), 1, replace=False)
                        ind_abs = np.concatenate((ind_abs, ind_one))
                else:
                    for i in range(len(self.n_samples_cum)-2):
                        ind_one = np.random.choice(np.arange(self.n_samples_cum[i], self.n_samples_cum[i+1]),
                                                   self.n_samples_per_trial, replace=False)
                        ind_abs = np.concatenate((ind_abs, ind_one))

                    i = len(self.n_samples_cum) - 2
                    ind_one = np.random.choice(np.arange(self.n_samples_cum[i], self.n_samples_cum[i + 1]),
                                               int(self.batch_size - len(ind_abs)), replace=False)
                    ind_abs = np.concatenate((ind_abs, ind_one))

                assert len(ind_abs) == self.batch_size

                ind_this1 = ind_abs + self.n_per*sub1
                ind_this2 = ind_abs + self.n_per*sub2

                batch = torch.LongTensor(np.concatenate((ind_this1, ind_this2)))
                yield batch
