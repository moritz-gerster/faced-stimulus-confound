"""CLISA MLP classification (adapted from Chen et al. 2023 main_classify.py).

Supports all cross-subject experiment variants:
  2a: baseline (crowd labels, 28 videos) — exports per-window predictions
  2b: concordance split (reuses 2a predictions)
  2c: subjective labels (retrain MLP with per-subject labels)
  2d: single video per emotion (restrict to 9 videos from same LDS features)
"""
import numpy as np
import torch
import os
from torch.utils.data import DataLoader

from decoding.config import (
    N_SUBJECTS, N_FOLDS, RESULTS_DIR,
    VIDEO_LABELS_9CLASS, WINDOWS_PER_TRIAL, SINGLE_VIDEO_INDICES,
    fold_split,
)
from . import _init_torch
from .io_utils import DEDataset
from .load_data import load_srt_pretrainFeat
from .model import simpleNN3
from .train_utils import train_earlyStopping


class _ClassifyArgs:
    """Mimics argparse namespace for classification."""
    epochs_finetune = 100
    max_tol = 50
    batch_size_finetune = 270
    learning_rate_finetune = 0.0005

    def __init__(self, device, save_dir_ft):
        self.device = device
        self.save_dir_ft = save_dir_ft


def _run_classify(labels, video_indices=None, analysis_name="baseline"):
    """Core classification loop.

    Args:
        labels: shape (n_subjects, n_windows) with class IDs per subject/window.
        video_indices: if not None, restrict to these video indices (0-based).
        analysis_name: name for result directory.

    Returns:
        dict with 'scores' and 'predictions' arrays.
    """
    device = _init_torch()

    sec = WINDOWS_PER_TRIAL
    save_dir = str(RESULTS_DIR / "clisa" / "runs_srt")
    out_dir = str(RESULTS_DIR / "clisa" / analysis_name)
    os.makedirs(out_dir, exist_ok=True)

    n_windows = labels.shape[1]
    subjects_score = np.zeros(N_SUBJECTS)
    all_predictions = np.full((N_SUBJECTS, n_windows), -1, dtype=np.int64)

    for fold in range(N_FOLDS):
        print(f'=== Classify {analysis_name} fold {fold} ===')
        fold_dir = os.path.join(save_dir, str(fold))
        args = _ClassifyArgs(device, fold_dir)

        use_features = os.path.join(fold_dir, 'features1_de_1s_lds.mat')
        data, _, _ = load_srt_pretrainFeat(
            use_features, channel_norm=True, timeLen=1, timeStep=1,
            isFilt=False, filtLen=1, label_type='cls9'
        )

        if video_indices is not None:
            window_idx = np.concatenate([
                np.arange(v * sec, v * sec + sec) for v in video_indices
            ])
            data = data[:, window_idx, :]

        train_sub, val_sub = fold_split(fold)
        val_sub_list = [int(v) for v in val_sub]
        train_sub_list = list(train_sub)

        data_train = data[train_sub_list].reshape(-1, data.shape[-1])
        lab_train = labels[train_sub_list].reshape(-1)
        data_val = data[val_sub_list].reshape(-1, data.shape[-1])
        lab_val = labels[val_sub_list].reshape(-1)

        trainset = DEDataset(data_train, lab_train)
        valset = DEDataset(data_val, lab_val)
        train_loader = DataLoader(dataset=trainset, batch_size=args.batch_size_finetune,
                                  shuffle=True, num_workers=8)
        val_loader = DataLoader(dataset=valset, batch_size=args.batch_size_finetune,
                                shuffle=False, num_workers=8)

        n_classes_out = int(labels.max()) + 1
        model = simpleNN3(data.shape[-1], 30, n_classes_out, 30, False).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate_finetune,
                                     weight_decay=0.05)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.epochs_finetune,
                                                     gamma=0.8, last_epoch=-1, verbose=False)
        criterion = torch.nn.CrossEntropyLoss().to(device)

        # Chen et al.'s CLISA pipeline uses the held-out fold for early
        # stopping/model selection and then reports that fold. We keep that
        # behavior so these outputs are replication-comparable, not nested-CV
        # generalization estimates.
        best_epoch, _, _, _, _, _ = train_earlyStopping(
            args, train_loader, val_loader, model, criterion, optimizer, scheduler, True
        )

        best_ckpt = os.path.join(fold_dir, f'finetune_checkpoint_{best_epoch:04d}.pth.tar')
        model.load_state_dict(torch.load(best_ckpt, map_location=device)['state_dict'])
        model.eval()

        results = []
        for counter, (x_batch, y_batch) in enumerate(val_loader):
            x_batch = x_batch.to(device)
            logits = model(x_batch)
            _, result = torch.max(logits, dim=1)
            results.extend(list(result.cpu().numpy()))

        results_arr = np.array(results).reshape(len(val_sub_list), -1)
        for i, s in enumerate(val_sub_list):
            all_predictions[s] = results_arr[i]
            subjects_score[s] = np.mean(labels[s] == results_arr[i])

    np.savez(
        os.path.join(out_dir, 'predictions.npz'),
        predictions=all_predictions,
        labels=labels,
        scores=subjects_score,
    )
    print(f'Results saved to {out_dir}')
    return {
        'scores': subjects_score,
        'predictions': all_predictions,
        'labels': labels,
    }


def run_clisa_baseline():
    """2a: Cross-subject baseline with crowd labels, 28 videos."""
    labels = np.tile(
        np.repeat(VIDEO_LABELS_9CLASS, WINDOWS_PER_TRIAL), (N_SUBJECTS, 1)
    )
    return _run_classify(labels, analysis_name="baseline")


def run_clisa_subjective(subjective_labels):
    """2c: Cross-subject with per-subject subjective labels.

    Args:
        subjective_labels: shape (123, 28) with class IDs per subject/video.
    """
    labels = np.repeat(subjective_labels, WINDOWS_PER_TRIAL, axis=1)
    return _run_classify(labels, analysis_name="subjective")


def run_clisa_single_video():
    """2d: Cross-subject with single video per emotion (9 videos)."""
    n_classes = len(SINGLE_VIDEO_INDICES)
    label_vec = np.repeat(np.arange(n_classes), WINDOWS_PER_TRIAL)
    labels = np.tile(label_vec, (N_SUBJECTS, 1))
    return _run_classify(labels, video_indices=SINGLE_VIDEO_INDICES,
                         analysis_name="single_video")
