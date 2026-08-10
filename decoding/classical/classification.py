"""Classification analyses for the FACED stimulus-identity confound study.

Part 1 — Intra-subject analyses:
  1a. Intra-subject baseline (within-video time split, replication of Chen et al. 2023)
  1b. Single video per emotion (intra-subject temporal CV)
  1c. Cross-video leave-one-out (28-fold, supplementary)
  1d. Balanced cross-video generalization (3-fold, class-balanced test sets)
  1d-control. Matched temporal control (same volume/folds, temporal split)
  1d-perm. Video-level permutation test (cross-video split)
  1d-control-perm. Video-level permutation test (temporal split)

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
    VIDEO_LABELS_9CLASS, N_PERMUTATIONS, fold_split, class_video_indices,
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


# ------------------------------------------------------------------
# 1c. Cross-video leave-one-out
# ------------------------------------------------------------------

def run_cross_video() -> dict:
    """Leave-one-video-out: train on 27 videos, test on held-out video.

    Uses smoothed features (same as intra-subject baseline) so the only
    difference from 1a is the split strategy. C is tuned on the held-out
    video (same optimistic test-set selection as 1a).
    """
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    smooth_dir = RESULTS_DIR / "smooth"
    data = sio.loadmat(str(smooth_dir / "de_lds_fold0.mat"))["de_lds"]
    data = _zscore_per_subject(data)

    n_subs = data.shape[0]
    wpt = WINDOWS_PER_TRIAL
    n_vids = N_VIDEOS

    vid_labels = np.array(VIDEO_LABELS_9CLASS)

    subjects_correct = np.zeros(n_subs)
    subjects_total = np.zeros(n_subs)

    for held_out in trange(n_vids, desc="Cross-video LOO"):
        train_vids = [v for v in range(n_vids) if v != held_out]

        train_data_list = []
        train_label_list = []
        for v in train_vids:
            start = v * wpt
            vid_data = data[:, start:start + wpt, :].reshape(-1, N_FEATURES)
            vid_label = np.full(n_subs * wpt, vid_labels[v])
            train_data_list.append(vid_data)
            train_label_list.append(vid_label)

        data_train = np.concatenate(train_data_list)
        label_train = np.concatenate(train_label_list)

        start = held_out * wpt
        data_test = data[:, start:start + wpt, :].reshape(-1, N_FEATURES)
        label_test = np.full(n_subs * wpt, vid_labels[held_out])

        clf, _ = _grid_search_C(data_train, label_train, data_test, label_test)
        preds = clf.predict(data_test)

        correct = (preds == label_test).reshape(n_subs, wpt)
        subjects_correct += correct.sum(axis=1)
        subjects_total += wpt

    subjects_score = subjects_correct / subjects_total

    cls_dir = RESULTS_DIR / "cross_video"
    os.makedirs(cls_dir, exist_ok=True)
    np.savez(
        str(cls_dir / "predictions.npz"),
        scores=subjects_score,
    )

    return {"scores": subjects_score}


# ------------------------------------------------------------------
# 1d. Balanced cross-video generalization & matched temporal control
# ------------------------------------------------------------------
#
# Design rationale (preserved here so it survives code changes):
#
# The two designs are identical in every respect — 27 videos (4th neutral
# clip dropped), 3 outer folds, same training/test volume per class per
# fold (60 train, 30 test windows per class) — differing only in the axis
# along which train and test are separated:
#   1d:         hold out 1 video per class (cross-video generalization)
#   1d-control: hold out 10 consecutive seconds of every video (temporal)
#
# Both test sets are class-balanced by construction, so plain accuracy
# equals balanced accuracy and chance is exactly 1/9 = 11.1%.
#
# The matched control (1d-control) is NOT identical to 1a: holding out
# 10-second blocks puts test windows further from training windows on
# average than 1a's 3-second blocks, so 1d-control carries less temporal
# autocorrelation leakage than 1a. This is conservative for the argument
# (biases against finding a difference between the two axes) but should
# not be described as matched to 1a.
#
# Normalization (z-score, LDS smoothing) is still computed over the full
# session including held-out windows, a pre-existing leakage retained for
# comparability with 1a.
#
# The full 2×2 over {split axis} × {C selection} separates the effect of
# fixing the hyperparameter tuning from the effect of the split axis.
# ------------------------------------------------------------------

N_MATCHED_FOLDS = 3  # 3 videos per class → 3-fold CV
_VIDEOS_PER_CLASS = 3
_TEST_SECONDS = WINDOWS_PER_TRIAL // N_MATCHED_FOLDS  # 10


def _cross_video_folds(groups, label_vec, wpt):
    """Yield (train_idx, test_idx, inner_splits) for cross-video 3-fold CV.

    Each fold holds out the k-th video of each class. Inner splits for
    nested C selection split the 2 training videos per class into two
    halves (one video each), matching the outer objective: cross-video
    generalization.
    """
    for k in range(N_MATCHED_FOLDS):
        test_vids = [g[k] for g in groups]
        train_vids = [v for g in groups for i, v in enumerate(g) if i != k]

        test_idx = np.concatenate([np.arange(v * wpt, (v + 1) * wpt)
                                   for v in test_vids])
        train_idx = np.concatenate([np.arange(v * wpt, (v + 1) * wpt)
                                    for v in train_vids])

        # Inner: leave-one-training-video-out (2-fold)
        inner_splits = []
        for ik in range(N_MATCHED_FOLDS - 1):
            inner_val_vids = [g[(k + 1 + ik) % N_MATCHED_FOLDS]
                              for g in groups]
            inner_val = np.concatenate([np.arange(v * wpt, (v + 1) * wpt)
                                        for v in inner_val_vids])
            inner_train = np.setdiff1d(train_idx, inner_val)
            inner_splits.append((inner_train, inner_val))

        yield train_idx, test_idx, inner_splits


def _temporal_folds(groups, label_vec, wpt):
    """Yield (train_idx, test_idx, inner_splits) for temporal 3-fold CV.

    Each fold holds out seconds [10k, 10k+10) of every retained video.
    Inner splits for nested C selection split the 20-second training
    portion into two 10-second halves, matching the outer objective:
    temporal generalization within a video.
    """
    all_vids = [v for g in groups for v in g]
    ts = _TEST_SECONDS

    for k in range(N_MATCHED_FOLDS):
        test_idx_parts = []
        train_idx_parts = []
        for v in all_vids:
            base = v * wpt
            test_start = base + ts * k
            test_idx_parts.append(np.arange(test_start, test_start + ts))
            full = np.arange(base, base + wpt)
            train_idx_parts.append(
                np.setdiff1d(full, np.arange(test_start, test_start + ts)))

        test_idx = np.concatenate(test_idx_parts)
        train_idx = np.concatenate(train_idx_parts)

        # Inner: split the 20-second training portion into two 10-second
        # halves (2-fold), matching the outer temporal objective.
        inner_splits = []
        remaining_folds = [f for f in range(N_MATCHED_FOLDS) if f != k]
        for ik in remaining_folds:
            inner_val_parts = []
            inner_train_parts = []
            for v in all_vids:
                base = v * wpt
                iv_start = base + ts * ik
                inner_val_parts.append(np.arange(iv_start, iv_start + ts))
                inner_train_parts.append(
                    np.setdiff1d(train_idx[
                        (train_idx >= base) & (train_idx < base + wpt)],
                        np.arange(iv_start, iv_start + ts)))
            inner_splits.append((np.concatenate(inner_train_parts),
                                 np.concatenate(inner_val_parts)))

        yield train_idx, test_idx, inner_splits


def _run_matched_split(fold_generator, c_selection, desc, out_dir):
    """Shared 3-fold engine for 1d and 1d-control.

    The two designs are identical except for how each fold's train/test
    window indices are built, so they share this engine deliberately —
    any accuracy difference between axes cannot come from implementation
    drift.

    Args:
        fold_generator: iterable of (train_idx, test_idx, inner_splits)
        c_selection: "optimistic" (test-set C, matching 1a) or "nested"
            (inner leave-one-out along the same axis as the outer split)
        desc: tqdm description
        out_dir: directory under results/ for saving outputs

    Returns:
        dict with 'scores', 'predictions', 'labels', 'best_C'
    """
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    smooth_dir = RESULTS_DIR / "smooth"
    data = sio.loadmat(str(smooth_dir / "de_lds_fold0.mat"))["de_lds"]
    data = _zscore_per_subject(data)

    groups = class_video_indices(_VIDEOS_PER_CLASS)
    all_vids = [v for g in groups for v in g]
    n_classes = len(groups)
    wpt = WINDOWS_PER_TRIAL

    # Build label vector for the 27 retained videos
    full_label_vec = np.repeat(VIDEO_LABELS_9CLASS, wpt)
    # Window indices for retained videos only
    retained_windows = np.concatenate([np.arange(v * wpt, (v + 1) * wpt)
                                       for v in all_vids])
    n_retained = len(retained_windows)

    n_subs = data.shape[0]
    subjects_correct = np.zeros(n_subs)
    all_preds = np.full((n_subs, n_retained), -1, dtype=int)
    all_labels = np.tile(full_label_vec[retained_windows], (n_subs, 1))
    best_C_list = []

    folds = list(fold_generator(groups, full_label_vec, wpt))
    test_offset = 0

    for fold_i, (train_idx, test_idx, inner_splits) in enumerate(folds):
        print(f"  {desc} fold {fold_i + 1}/{N_MATCHED_FOLDS}")

        data_train = data[:, train_idx, :].reshape(-1, N_FEATURES)
        label_train = np.tile(full_label_vec[train_idx], n_subs)
        data_test = data[:, test_idx, :].reshape(-1, N_FEATURES)
        label_test = np.tile(full_label_vec[test_idx], n_subs)

        if c_selection == "optimistic":
            best_clf, best_C = _grid_search_C(
                data_train, label_train, data_test, label_test)
        else:
            # Nested C: inner CV along the same axis as the outer split
            best_C, best_inner_acc = C_CANDIDATES[0], 0.0
            for C in C_CANDIDATES:
                accs = []
                for it_idx, iv_idx in inner_splits:
                    d_it = data[:, it_idx, :].reshape(-1, N_FEATURES)
                    l_it = np.tile(full_label_vec[it_idx], n_subs)
                    d_iv = data[:, iv_idx, :].reshape(-1, N_FEATURES)
                    l_iv = np.tile(full_label_vec[iv_idx], n_subs)
                    clf = LinearSVC(random_state=RANDOM_SEED, C=C)
                    clf.fit(d_it, l_it)
                    accs.append(np.mean(l_iv == clf.predict(d_iv)))
                if np.mean(accs) > best_inner_acc:
                    best_inner_acc = np.mean(accs)
                    best_C = C
            best_clf = LinearSVC(random_state=RANDOM_SEED, C=best_C)
            best_clf.fit(data_train, label_train)

        best_C_list.append(float(best_C))
        preds = best_clf.predict(data_test)

        n_test = len(test_idx)
        preds_by_sub = preds.reshape(n_subs, n_test)
        labels_by_sub = label_test.reshape(n_subs, n_test)
        for s in range(n_subs):
            subjects_correct[s] += np.sum(preds_by_sub[s] == labels_by_sub[s])
            all_preds[s, test_offset:test_offset + n_test] = preds_by_sub[s]

        test_offset += n_test

    subjects_score = subjects_correct / n_retained

    cls_dir = RESULTS_DIR / out_dir
    os.makedirs(cls_dir, exist_ok=True)
    np.savez(
        str(cls_dir / f"predictions_{c_selection}.npz"),
        scores=subjects_score,
        predictions=all_preds,
        labels=all_labels,
        best_C=np.array(best_C_list),
    )

    return {"scores": subjects_score, "best_C": best_C_list}


def run_cross_video_balanced(c_selection="nested") -> dict:
    """Balanced cross-video generalization (1d).

    Hold out the k-th video of each emotion class (k=0,1,2). Tests
    cross-video generalization with class-balanced test sets.
    """
    return _run_matched_split(
        _cross_video_folds, c_selection,
        f"Cross-video ({c_selection})", "cross_video_balanced")


def run_temporal_control(c_selection="nested") -> dict:
    """Matched temporal control (1d-control).

    Hold out 10 consecutive seconds of every video (k=0,1,2). Same
    training volume and class balance as 1d, differing only in split axis.
    """
    return _run_matched_split(
        _temporal_folds, c_selection,
        f"Temporal control ({c_selection})", "temporal_control")


# ------------------------------------------------------------------
# 1d-perm / 1d-control-perm. Video-level permutation tests
# ------------------------------------------------------------------
#
# Permutes video-to-emotion assignment (27 videos reshuffled into
# 9 groups of 3) and rebuilds folds. Permuting at video level rather
# than window level is the whole point: the independent unit is the
# stimulus, not the 123×30 = 3690 windows.
#
# C is frozen at the value the nested procedure selected on real data.
# Re-running the 12-value grid per permutation would multiply cost by
# 12 for no inferential gain.
#
# The contrast between the two permutation tests is the argument:
#   - Cross-video null should FAIL (low accuracy) — the classifier
#     cannot match unseen videos to arbitrary labels.
#   - Temporal null should SUCCEED (high accuracy) — temporal
#     autocorrelation lets the classifier decode arbitrary video
#     groupings from within-video time splits, proving that the
#     true emotion labels contribute almost nothing.
# ------------------------------------------------------------------

def _run_permutation(fold_fn, best_C_list, desc, out_dir,
                     n_permutations=N_PERMUTATIONS):
    """Shared permutation engine for both split axes.

    Shuffles 27 videos into 9 arbitrary groups of 3, rebuilds the folds
    from the shuffled grouping, and classifies with frozen C.
    Reports p = (1 + #{perm >= observed}) / (n + 1).
    """
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    smooth_dir = RESULTS_DIR / "smooth"
    data = sio.loadmat(str(smooth_dir / "de_lds_fold0.mat"))["de_lds"]
    data = _zscore_per_subject(data)

    real_groups = class_video_indices(_VIDEOS_PER_CLASS)
    all_vids = [v for g in real_groups for v in g]
    wpt = WINDOWS_PER_TRIAL
    n_subs = data.shape[0]
    n_retained = len(all_vids) * wpt

    null_accuracies = np.zeros(n_permutations)

    for pi in range(n_permutations):
        if (pi + 1) % 100 == 0:
            print(f"  {desc}: permutation {pi + 1}/{n_permutations}")

        shuffled_vids = list(all_vids)
        random.shuffle(shuffled_vids)
        perm_groups = [shuffled_vids[i * _VIDEOS_PER_CLASS:
                                     (i + 1) * _VIDEOS_PER_CLASS]
                       for i in range(len(real_groups))]

        # Build a permuted label vector: group index becomes the label
        perm_label_vec = np.zeros(N_VIDEOS * wpt, dtype=int)
        for cls_i, grp in enumerate(perm_groups):
            for v in grp:
                perm_label_vec[v * wpt:(v + 1) * wpt] = cls_i

        folds = list(fold_fn(perm_groups, perm_label_vec, wpt))
        subjects_correct = np.zeros(n_subs)

        for fold_i, (train_idx, test_idx, _) in enumerate(folds):
            data_train = data[:, train_idx, :].reshape(-1, N_FEATURES)
            label_train = np.tile(perm_label_vec[train_idx], n_subs)
            data_test = data[:, test_idx, :].reshape(-1, N_FEATURES)
            label_test = np.tile(perm_label_vec[test_idx], n_subs)

            C = best_C_list[fold_i]
            clf = LinearSVC(random_state=RANDOM_SEED, C=C)
            clf.fit(data_train, label_train)
            preds = clf.predict(data_test)

            correct = (preds == label_test).reshape(n_subs, -1)
            subjects_correct += correct.sum(axis=1)

        null_accuracies[pi] = np.mean(subjects_correct / n_retained)

    cls_dir = RESULTS_DIR / out_dir
    os.makedirs(cls_dir, exist_ok=True)
    np.save(str(cls_dir / "null_distribution.npy"), null_accuracies)

    return {"null_distribution": null_accuracies}


def run_cross_video_permutation(n_permutations=N_PERMUTATIONS) -> dict:
    """Video-level permutation test under cross-video split (1d-perm).

    Uses C frozen from the nested run on real data. Expected result:
    accuracy near chance, confirming that cross-video generalization
    requires the true emotion grouping.
    """
    saved = np.load(str(RESULTS_DIR / "cross_video_balanced"
                        / "predictions_nested.npz"))
    observed = float(np.mean(saved["scores"]))
    best_C_list = list(saved["best_C"])

    perm = _run_permutation(
        _cross_video_folds, best_C_list,
        "Cross-video perm", "cross_video_balanced", n_permutations)

    null = perm["null_distribution"]
    p_value = (1 + np.sum(null >= observed)) / (n_permutations + 1)
    print(f"  Cross-video permutation: observed={observed:.3f}, "
          f"p={p_value:.4f} (n={n_permutations})")

    np.savez(str(RESULTS_DIR / "cross_video_balanced" / "permutation.npz"),
             null_distribution=null, observed=observed, p_value=p_value)

    return {"null_distribution": null, "observed": observed,
            "p_value": p_value}


def run_temporal_permutation(n_permutations=N_PERMUTATIONS) -> dict:
    """Video-level permutation test under temporal split (1d-control-perm).

    Uses C frozen from the nested run on real data. Expected result:
    accuracy comparable to the real temporal-control accuracy, because
    temporal autocorrelation lets the classifier decode arbitrary video
    groupings. If confirmed, this proves the true emotion labels
    contribute almost nothing to temporal-split performance.
    """
    saved = np.load(str(RESULTS_DIR / "temporal_control"
                        / "predictions_nested.npz"))
    observed = float(np.mean(saved["scores"]))
    best_C_list = list(saved["best_C"])

    perm = _run_permutation(
        _temporal_folds, best_C_list,
        "Temporal perm", "temporal_control", n_permutations)

    null = perm["null_distribution"]
    p_value = (1 + np.sum(null >= observed)) / (n_permutations + 1)
    print(f"  Temporal permutation: observed={observed:.3f}, "
          f"p={p_value:.4f} (n={n_permutations})")

    np.savez(str(RESULTS_DIR / "temporal_control" / "permutation.npz"),
             null_distribution=null, observed=observed, p_value=p_value)

    return {"null_distribution": null, "observed": observed,
            "p_value": p_value}


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
