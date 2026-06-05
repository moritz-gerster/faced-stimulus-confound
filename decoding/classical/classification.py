"""Classification analyses for the FACED stimulus-identity confound study.

Part 1 — Intra-subject analyses:
  1a. Intra-subject baseline (within-video time split, replication of Chen et al. 2023)
  1b. Single video per emotion (intra-subject temporal CV)

Part 2 — Cross-subject analyses:
  2a. Cross-subject baseline (replication of Chen et al. 2023)
  2b. Concordance split (see concordance.py)
  2c. Subjective-label variant
  2d. Single video per emotion
"""
import json
import os
import random
import warnings

import numpy as np
import scipy.io as sio
from sklearn.exceptions import ConvergenceWarning
from sklearn.svm import LinearSVC
from tqdm import trange

warnings.filterwarnings("ignore", category=ConvergenceWarning)

from decoding.config import (
    N_SUBJECTS, N_VIDEOS, N_FOLDS, N_FEATURES, WINDOWS_PER_TRIAL,
    C_CANDIDATES, RANDOM_SEED, RESULTS_DIR, SINGLE_VIDEO_INDICES,
    VIDEO_LABELS_9CLASS, fold_split,
)


def _zscore_per_subject(data: np.ndarray) -> np.ndarray:
    """Per-subject z-normalization over time axis."""
    mu = np.mean(data, axis=1, keepdims=True)
    std = np.std(data, axis=1, keepdims=True)
    std[std == 0] = 1.0
    return (data - mu) / std


def _grid_search_C(data_train, label_train, data_val, label_val, **svc_kwargs):
    """Find best C on the held-out fold, matching Chen et al. (2023).

    This intentionally uses the same held-out subjects/windows for model
    selection and reporting so the output is replication-comparable. It is not
    a nested-CV estimate of generalization performance.
    """
    best_C, best_acc, best_clf = C_CANDIDATES[0], 0.0, None
    for C in C_CANDIDATES:
        clf = LinearSVC(random_state=RANDOM_SEED, C=C, **svc_kwargs)
        clf.fit(data_train, label_train)
        acc = np.mean(label_val == clf.predict(data_val))
        if acc > best_acc:
            best_acc, best_C, best_clf = acc, C, clf
    return best_clf, best_C


def _run_fold_cv(labels, desc, track_predictions=False):
    """Shared 10-fold cross-subject SVM with grid-searched C.

    Args:
        labels: shape (n_subjects, n_windows) with class IDs.
        desc: tqdm progress bar description.
        track_predictions: if True, also return predictions and best_C per fold.
    """
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    smooth_dir = RESULTS_DIR / "smooth"
    n_subs = labels.shape[0]

    subjects_score = np.zeros(n_subs)
    all_preds = np.full_like(labels, -1) if track_predictions else None
    best_C_per_fold = [] if track_predictions else None

    for fold in trange(N_FOLDS, desc=desc):
        data = sio.loadmat(str(smooth_dir / f"de_lds_fold{fold}.mat"))["de_lds"]
        data = _zscore_per_subject(data)

        train_sub, val_sub = fold_split(fold)

        data_train = data[train_sub].reshape(-1, N_FEATURES)
        label_train = labels[train_sub].reshape(-1)
        data_val = data[val_sub].reshape(-1, N_FEATURES)
        label_val = labels[val_sub].reshape(-1)

        best_clf, best_C = _grid_search_C(
            data_train, label_train, data_val, label_val
        )

        if track_predictions:
            best_C_per_fold.append(float(best_C))

        preds = best_clf.predict(data_val)
        preds_reshaped = preds.reshape(len(val_sub), -1)
        for i, s in enumerate(val_sub):
            subjects_score[s] = np.mean(labels[s] == preds_reshaped[i])
            if track_predictions:
                all_preds[s] = preds_reshaped[i]

    result = {"scores": subjects_score}
    if track_predictions:
        result["predictions"] = all_preds
        result["labels"] = labels
        result["best_C"] = best_C_per_fold
    return result


# ==================================================================
# Part 1 — Intra-subject analyses
# ==================================================================

# ------------------------------------------------------------------
# 1a. Intra-subject baseline (within-video time split)
# ------------------------------------------------------------------

def run_intra_subject() -> dict:
    """10-fold intra-subject SVM replicating Chen et al. (2023).

    Split: within each 30-second video, hold out 3 consecutive seconds
    per fold. All subjects are in both train and test.
    """
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    smooth_dir = RESULTS_DIR / "smooth"
    wpt = WINDOWS_PER_TRIAL
    n_total = N_VIDEOS * wpt

    label_vec = np.repeat(VIDEO_LABELS_9CLASS, wpt)

    sample = sio.loadmat(str(smooth_dir / "de_lds_fold0.mat"))["de_lds"]
    n_subs = sample.shape[0]
    subjects_score = np.zeros(n_subs)
    best_C_per_fold = []

    val_seconds = wpt // N_FOLDS  # 3 seconds per fold

    for fold in trange(N_FOLDS, desc="Intra-subject folds"):
        data = sio.loadmat(
            str(smooth_dir / f"de_lds_fold{fold}.mat")
        )["de_lds"]
        data = _zscore_per_subject(data)

        # Hold out 3 consecutive seconds from each 30-s trial
        val_starts = np.arange(0, n_total, wpt) + val_seconds * fold
        val_idx = np.concatenate([
            np.arange(s, s + val_seconds) for s in val_starts
        ])
        train_idx = np.setdiff1d(np.arange(n_total), val_idx)

        data_train = data[:, train_idx, :].reshape(-1, N_FEATURES)
        label_train = np.tile(label_vec[train_idx], n_subs)
        data_val = data[:, val_idx, :].reshape(-1, N_FEATURES)
        label_val = np.tile(label_vec[val_idx], n_subs)

        best_clf, best_C = _grid_search_C(
            data_train, label_train, data_val, label_val
        )
        best_C_per_fold.append(float(best_C))
        preds = best_clf.predict(data_val)

        preds_by_sub = preds.reshape(n_subs, -1)
        labels_by_sub = label_val.reshape(n_subs, -1)
        for s in range(n_subs):
            subjects_score[s] += np.sum(preds_by_sub[s] == labels_by_sub[s])

    subjects_score /= n_total

    cls_dir = RESULTS_DIR / "intra_subject"
    os.makedirs(cls_dir, exist_ok=True)
    np.save(cls_dir / "scores.npy", subjects_score)

    return {"scores": subjects_score, "best_C": best_C_per_fold}


# ------------------------------------------------------------------
# 1b. Single video per emotion (intra-subject temporal CV)
# ------------------------------------------------------------------

def run_intra_single_video() -> dict:
    """Intra-subject SVM with one video per emotion (9 videos total).

    Uses the same temporal CV as 1a (27s train, 3s test) but restricts
    to a single video per emotion category.  Each class is a single video,
    so high accuracy is consistent with temporal autocorrelation contributing
    to classification; within-class stimulus similarity cannot contribute.
    """
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    smooth_dir = RESULTS_DIR / "smooth"
    wpt = WINDOWS_PER_TRIAL
    n_classes = len(SINGLE_VIDEO_INDICES)
    n_total = n_classes * wpt  # 9 * 30 = 270
    window_indices = (
        np.array(SINGLE_VIDEO_INDICES)[:, None] * wpt + np.arange(wpt)
    ).ravel()

    label_vec = np.repeat(np.arange(n_classes), wpt)

    sample = sio.loadmat(str(smooth_dir / "de_lds_fold0.mat"))["de_lds"]
    n_subs = sample.shape[0]
    subjects_score = np.zeros(n_subs)

    val_seconds = wpt // N_FOLDS

    for fold in trange(N_FOLDS, desc="Intra single-video folds"):
        data_full = sio.loadmat(
            str(smooth_dir / f"de_lds_fold{fold}.mat")
        )["de_lds"]
        data_full = _zscore_per_subject(data_full)
        data = data_full[:, window_indices, :]

        val_starts = np.arange(0, n_total, wpt) + val_seconds * fold
        val_idx = np.concatenate([
            np.arange(s, s + val_seconds) for s in val_starts
        ])
        train_idx = np.setdiff1d(np.arange(n_total), val_idx)

        data_train = data[:, train_idx, :].reshape(-1, N_FEATURES)
        label_train = np.tile(label_vec[train_idx], n_subs)
        data_val = data[:, val_idx, :].reshape(-1, N_FEATURES)
        label_val = np.tile(label_vec[val_idx], n_subs)

        best_clf, _ = _grid_search_C(
            data_train, label_train, data_val, label_val
        )
        preds = best_clf.predict(data_val)

        preds_by_sub = preds.reshape(n_subs, -1)
        labels_by_sub = label_val.reshape(n_subs, -1)
        for s in range(n_subs):
            subjects_score[s] += np.sum(preds_by_sub[s] == labels_by_sub[s])

    subjects_score /= n_total

    cls_dir = RESULTS_DIR / "intra_single_video"
    os.makedirs(cls_dir, exist_ok=True)
    np.save(cls_dir / "scores.npy", subjects_score)

    return {"scores": subjects_score}


# ==================================================================
# Part 2 — Cross-subject analyses
# ==================================================================

# ------------------------------------------------------------------
# 2a. Cross-subject baseline
# ------------------------------------------------------------------

def run_baseline() -> dict:
    """10-fold cross-subject SVM with smoothed features and crowd labels.

    Returns dict with 'scores' (per-subject accuracy), 'predictions', 'labels',
    and 'best_C' list.
    """
    labels = np.tile(
        np.repeat(VIDEO_LABELS_9CLASS, WINDOWS_PER_TRIAL), (N_SUBJECTS, 1)
    )
    result = _run_fold_cv(labels, "Baseline folds", track_predictions=True)

    cls_dir = RESULTS_DIR / "baseline"
    os.makedirs(cls_dir, exist_ok=True)
    np.savez(
        str(cls_dir / "predictions.npz"),
        predictions=result["predictions"], labels=result["labels"],
        scores=result["scores"],
    )
    with open(cls_dir / "best_C.json", "w") as f:
        json.dump(result["best_C"], f)

    return result


# ------------------------------------------------------------------
# 2c. Subjective labels
# ------------------------------------------------------------------

def run_subjective(subjective_labels: np.ndarray) -> dict:
    """10-fold cross-subject SVM with per-subject subjective labels.

    Args:
        subjective_labels: shape (123, 28) with class IDs per subject/video.

    Returns dict with 'scores' array.
    """
    labels = np.repeat(subjective_labels, WINDOWS_PER_TRIAL, axis=1)
    result = _run_fold_cv(labels, "Subjective folds", track_predictions=True)

    cls_dir = RESULTS_DIR / "subjective"
    os.makedirs(cls_dir, exist_ok=True)
    np.savez(
        str(cls_dir / "predictions.npz"),
        predictions=result["predictions"], labels=result["labels"],
        scores=result["scores"],
    )

    return result



# ------------------------------------------------------------------
# 2d. Single video per emotion
# ------------------------------------------------------------------

def run_single_video() -> dict:
    """9-class SVM using only 1 video per emotion (9 total).

    Uses smoothed features from baseline normalization.
    """
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    smooth_dir = RESULTS_DIR / "smooth"
    n_classes = len(SINGLE_VIDEO_INDICES)
    wpt = WINDOWS_PER_TRIAL
    n_windows = n_classes * wpt  # 270
    window_indices = (
        np.array(SINGLE_VIDEO_INDICES)[:, None] * wpt + np.arange(wpt)
    ).ravel()

    # Labels: 0..8, each repeated 30 times
    sample = sio.loadmat(str(smooth_dir / "de_lds_fold0.mat"))["de_lds"]
    n_subs = sample.shape[0]
    label_vec = np.repeat(np.arange(n_classes), wpt)
    labels = np.tile(label_vec, (n_subs, 1))

    subjects_score = np.zeros(n_subs)

    for fold in trange(N_FOLDS, desc="Single-video folds"):
        data_full = sio.loadmat(
            str(smooth_dir / f"de_lds_fold{fold}.mat")
        )["de_lds"]
        data_full = _zscore_per_subject(data_full)
        data = data_full[:, window_indices, :]

        train_sub, val_sub = fold_split(fold)

        data_train = data[train_sub].reshape(-1, N_FEATURES)
        label_train = labels[train_sub].reshape(-1)
        data_val = data[val_sub].reshape(-1, N_FEATURES)
        label_val = labels[val_sub].reshape(-1)

        # Nested 5-fold inner C search
        best_C, best_acc = C_CANDIDATES[0], 0.0
        for C in C_CANDIDATES:
            accs = []
            for inner_fold in range(5):
                it, iv = fold_split(inner_fold, len(train_sub))
                clf = LinearSVC(random_state=RANDOM_SEED, C=C, max_iter=5000)
                clf.fit(
                    data[train_sub[it]].reshape(-1, N_FEATURES),
                    labels[train_sub[it]].reshape(-1),
                )
                preds_inner = clf.predict(data[train_sub[iv]].reshape(-1, N_FEATURES))
                accs.append(np.mean(labels[train_sub[iv]].reshape(-1) == preds_inner))
            if np.mean(accs) > best_acc:
                best_acc = np.mean(accs)
                best_C = C

        clf = LinearSVC(random_state=RANDOM_SEED, C=best_C, max_iter=5000)
        clf.fit(data_train, label_train)
        preds = clf.predict(data_val)

        preds_reshaped = preds.reshape(len(val_sub), n_windows)
        for i, s in enumerate(val_sub):
            subjects_score[s] = np.mean(labels[s] == preds_reshaped[i])

    cls_dir = RESULTS_DIR / "single_video"
    os.makedirs(cls_dir, exist_ok=True)
    np.save(cls_dir / "scores.npy", subjects_score)

    return {"scores": subjects_score}
