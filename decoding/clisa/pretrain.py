"""CLISA SimCLR pretraining (adapted from Chen et al. 2023 main_pretrain.py).

Adapted: removed argparse, paths use RESULTS_DIR, callable as function.
"""
import numpy as np
import torch
import os
import pickle
from torch.utils.data import DataLoader

from decoding.config import N_SUBJECTS, N_FOLDS, RANDOM_SEED, RESULTS_DIR, fold_split
from . import _init_torch
from .io_utils import EmotionDataset, TrainSampler
from .load_data import load_srt_raw_newPre
from .model import ConvNet_baseNonlinearHead
from .simCLR import SimCLR


class _Args:
    """Mimics argparse namespace with CLISA default hyperparameters."""
    epochs_pretrain = 80
    restart_times = 3
    max_tol_pretrain = 30
    n_views = 2
    batch_size_pretrain = 28
    learning_rate = 0.0007
    weight_decay = 0.015
    temperature = 0.07
    n_times = 1
    fp16_precision = False
    timeLen = 5
    randSeed = RANDOM_SEED
    n_spatialFilters = 16
    n_timeFilters = 16
    timeFilterLen = 60
    multiFact = 2
    hidden_dim = 30
    cls = 9
    epochs_finetune = 100
    max_tol = 50
    batch_size_finetune = 270
    learning_rate_finetune = 0.0005
    gpu_index = 0
    device = None


def run_pretrain(fold=None):
    """Run SimCLR pretraining for one or all folds.

    Args:
        fold: If int, run single fold. If None, run all 10 folds.
    """
    args = _Args()
    args.device = _init_torch()

    stratified = ['initial', 'middle1', 'middle2']
    n_channs = 30
    timeLen = args.timeLen
    timeStep = 2
    fs = 250

    data, label_repeat, n_samples, n_segs = load_srt_raw_newPre(
        timeLen, timeStep, fs, channel_norm=False, time_norm=False
    )

    n_subs = N_SUBJECTS
    save_dir = str(RESULTS_DIR / "clisa" / "runs_srt")
    os.makedirs(save_dir, exist_ok=True)

    folds_list = range(N_FOLDS) if fold is None else [fold]

    results_pretrain = {
        'train_top1_history': np.zeros((N_FOLDS, args.epochs_pretrain)),
        'val_top1_history': np.zeros((N_FOLDS, args.epochs_pretrain)),
        'train_top5_history': np.zeros((N_FOLDS, args.epochs_pretrain)),
        'val_top5_history': np.zeros((N_FOLDS, args.epochs_pretrain)),
        'train_loss_history': np.zeros((N_FOLDS, args.epochs_pretrain)),
        'val_loss_history': np.zeros((N_FOLDS, args.epochs_pretrain)),
        'best_val_top1': np.zeros(N_FOLDS),
        'best_val_top5': np.zeros(N_FOLDS),
        'best_val_loss': np.zeros(N_FOLDS),
        'best_epoch': np.zeros(N_FOLDS),
    }

    for f in folds_list:
        print(f'=== Pretrain fold {f} ===')
        fold_dir = os.path.join(save_dir, str(f))
        os.makedirs(fold_dir, exist_ok=True)

        model_pre = ConvNet_baseNonlinearHead(
            args.n_spatialFilters, args.n_timeFilters, args.timeFilterLen,
            n_channs, stratified=stratified, multiFact=args.multiFact,
            isMaxPool=False, args=args
        ).to(args.device)

        train_sub, val_sub = fold_split(f)
        val_sub = [int(v) for v in val_sub]
        train_sub_list = list(train_sub)

        data_train = data[train_sub_list, :, :].reshape(-1, data.shape[-1])
        label_train = np.tile(label_repeat, len(train_sub_list))
        data_val = data[val_sub, :, :].reshape(-1, data.shape[-1])
        label_val = np.tile(label_repeat, len(val_sub))

        trainset = EmotionDataset(data_train, label_train, timeLen, timeStep, n_segs, fs)
        valset = EmotionDataset(data_val, label_val, timeLen, timeStep, n_segs, fs)

        train_sampler = TrainSampler(len(train_sub_list), n_times=args.n_times,
                                     batch_size=args.batch_size_pretrain, n_samples=n_samples)
        val_sampler = TrainSampler(len(val_sub), n_times=args.n_times,
                                   batch_size=args.batch_size_pretrain, n_samples=n_samples)

        train_loader = DataLoader(dataset=trainset, batch_sampler=train_sampler, pin_memory=True, num_workers=8)
        val_loader = DataLoader(dataset=valset, batch_sampler=val_sampler, pin_memory=True, num_workers=8)

        optimizer = torch.optim.Adam(model_pre.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=args.epochs_pretrain // args.restart_times, eta_min=0, last_epoch=-1
        )

        with torch.cuda.device(args.gpu_index):
            simclr = SimCLR(args=args, model=model_pre, optimizer=optimizer,
                            scheduler=scheduler, log_dir=fold_dir, stratified='no')
            model_pre, best_epoch, train_top1, val_top1, train_top5, val_top5, train_loss, val_loss = simclr.train(
                train_loader, val_loader, args.n_times
            )

        results_pretrain['train_top1_history'][f,:] = train_top1
        results_pretrain['val_top1_history'][f,:] = val_top1
        results_pretrain['train_top5_history'][f,:] = train_top5
        results_pretrain['val_top5_history'][f,:] = val_top5
        results_pretrain['train_loss_history'][f,:] = train_loss
        results_pretrain['val_loss_history'][f,:] = val_loss
        results_pretrain['best_val_top1'][f] = val_top1[best_epoch]
        results_pretrain['best_val_top5'][f] = val_top5[best_epoch]
        results_pretrain['best_val_loss'][f] = val_loss[best_epoch]
        results_pretrain['best_epoch'][f] = best_epoch

        np.save(os.path.join(fold_dir, 'train_top1_history.npy'), train_top1)
        np.save(os.path.join(fold_dir, 'val_top1_history.npy'), val_top1)

    results_path = os.path.join(save_dir, 'folds_all_dataset_both_results_pretrain.pkl')
    with open(results_path, 'wb') as fp:
        pickle.dump(results_pretrain, fp)
    print(f'Pretrain results saved to {results_path}')
