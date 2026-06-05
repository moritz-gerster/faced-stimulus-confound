"""Concordance analysis: split baseline accuracy by subjective agreement.

For each subject and video, determine whether the subject's self-reported
emotion (argmax of ratings) matches the crowd/video-assigned label.
Then compute baseline SVM accuracy separately for concordant and discordant
trials, assessing whether performance is similar regardless of subjective
experience.
"""
import numpy as np
from sklearn.metrics import balanced_accuracy_score

from decoding.config import NEUTRAL_CLASS, WINDOWS_PER_TRIAL, VIDEO_LABELS_9CLASS


def run_concordance(
    baseline_predictions: np.ndarray,
    baseline_labels: np.ndarray,
    subjective_labels: np.ndarray,
) -> dict:
    """Split baseline predictions by concordance with subjective labels.

    Args:
        baseline_predictions: shape (123, 840) from baseline SVM
        baseline_labels: shape (123, 840) crowd labels
        subjective_labels: shape (123, 28) per-subject argmax of self-report

    Returns:
        Dict with 'concordant_acc', 'discordant_acc', 'overall_acc' arrays.
    """
    wpt = WINDOWS_PER_TRIAL
    crowd_per_video = np.array(VIDEO_LABELS_9CLASS)

    concordance = np.repeat(
        subjective_labels == crowd_per_video, wpt, axis=1
    )  # (123, 840)
    non_neutral = np.repeat(crowd_per_video != NEUTRAL_CLASS, wpt)  # (840,)

    conc_nn = concordance & non_neutral
    disc_nn = ~concordance & non_neutral

    n_subjects = baseline_labels.shape[0]
    concordant_accs = np.zeros(n_subjects)
    discordant_accs = np.zeros(n_subjects)
    overall_accs = np.zeros(n_subjects)

    for s in range(n_subjects):
        if conc_nn[s].sum() > 0:
            concordant_accs[s] = balanced_accuracy_score(
                baseline_labels[s, conc_nn[s]],
                baseline_predictions[s, conc_nn[s]],
            )
        if disc_nn[s].sum() > 0:
            discordant_accs[s] = balanced_accuracy_score(
                baseline_labels[s, disc_nn[s]],
                baseline_predictions[s, disc_nn[s]],
            )
        if non_neutral.sum() > 0:
            overall_accs[s] = balanced_accuracy_score(
                baseline_labels[s, non_neutral],
                baseline_predictions[s, non_neutral],
            )

    return {
        "concordant_acc": concordant_accs,
        "discordant_acc": discordant_accs,
        "overall_acc": overall_accs,
    }
