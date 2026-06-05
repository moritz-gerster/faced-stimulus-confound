"""Shared data loading for cross-subject figures (fig2, fig3).

Loads SVM and CLISA results from results/ .npz/.npy files.
Concordance is recomputed from saved predictions + subjective labels.
"""
import numpy as np

from decoding.config import N_SUBJECTS, VIDEO_LABELS_9CLASS, WINDOWS_PER_TRIAL
from decoding.labels import load_subjective_labels
from .style import RESULTS_DIR


def _load_scores(directory):
    """Load scores from predictions.npz or scores.npy in directory."""
    npz = directory / "predictions.npz"
    npy = directory / "scores.npy"
    if npz.exists():
        return np.load(str(npz))["scores"]
    if npy.exists():
        return np.load(npy)
    raise FileNotFoundError(
        f"No scores found in {directory} (checked predictions.npz and scores.npy)"
    )


def _load_predictions(path):
    """Load (predictions, labels) from .npz."""
    if not path.exists():
        raise FileNotFoundError(f"Predictions file not found: {path}")
    d = np.load(str(path))
    labels = d["labels"] if "labels" in d else None
    return d["predictions"], labels


def _concordance(predictions, labels, subj_labels):
    """Compute concordance split. Returns (concordant_acc, discordant_acc)."""
    import warnings
    from decoding.concordance import run_concordance
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="y_pred contains classes not in y_true")
        conc = run_concordance(predictions, labels, subj_labels)
    return conc["concordant_acc"], conc["discordant_acc"]


def _crowd_labels() -> np.ndarray:
    """Return crowd labels in subject x window shape."""
    labels = np.repeat(VIDEO_LABELS_9CLASS, WINDOWS_PER_TRIAL)
    return np.tile(labels, (N_SUBJECTS, 1))


def _subjective_window_labels(subj_labels):
    """Return subjective labels in subject x window shape."""
    return np.repeat(subj_labels, WINDOWS_PER_TRIAL, axis=1)


def load_cross_subject_data():
    """Load all cross-subject results into a flat dict."""
    data = {}
    subj_labels = load_subjective_labels()

    # --- SVM ---
    bl_pred, bl_labels = _load_predictions(
        RESULTS_DIR / "baseline" / "predictions.npz"
    )
    data["svm_baseline"] = _load_scores(RESULTS_DIR / "baseline")

    subj_pred, subj_pred_labels = _load_predictions(
        RESULTS_DIR / "subjective" / "predictions.npz"
    )
    data["svm_subjective"] = _load_scores(RESULTS_DIR / "subjective")
    data["svm_single"] = _load_scores(RESULTS_DIR / "single_video")

    data["svm_bl_conc"], data["svm_bl_disc"] = _concordance(
        bl_pred, bl_labels, subj_labels
    )
    data["svm_subj_conc"], data["svm_subj_disc"] = _concordance(
        subj_pred, subj_pred_labels, subj_labels
    )

    # --- CLISA ---
    clisa = RESULTS_DIR / "clisa"
    data["clisa_baseline"] = _load_scores(clisa / "baseline")
    data["clisa_subjective"] = _load_scores(clisa / "subjective")
    data["clisa_single"] = _load_scores(clisa / "single_video")

    cl_bl_pred, cl_bl_labels = _load_predictions(
        clisa / "baseline" / "predictions.npz"
    )
    if cl_bl_labels is None:
        cl_bl_labels = _crowd_labels()
    data["clisa_bl_conc"], data["clisa_bl_disc"] = _concordance(
        cl_bl_pred, cl_bl_labels, subj_labels
    )

    cl_subj_pred, cl_subj_labels = _load_predictions(
        clisa / "subjective" / "predictions.npz"
    )
    if cl_subj_labels is None:
        cl_subj_labels = _subjective_window_labels(subj_labels)
    data["clisa_subj_conc"], data["clisa_subj_disc"] = _concordance(
        cl_subj_pred, cl_subj_labels, subj_labels
    )

    return data
