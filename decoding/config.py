"""Configuration for FACED stimulus-identity confound analyses.

All parameters follow Chen et al. (2023), Scientific Data.
"""
import os
from pathlib import Path

import numpy as np

# --- Paths ---
DATA_DIR = Path(os.environ.get("DATA_DIR", "data"))
RESULTS_DIR = Path("results")

DE_DIR = DATA_DIR / "EEG_Features" / "DE"
REMARKS_DIR = DATA_DIR / "Data"

# --- Dataset ---
N_SUBJECTS = 123
N_VIDEOS = 28
N_CHANNELS = 30  # 32 raw minus 2 mastoids
CHANNELS_TO_DROP = 2
N_BANDS = 5
WINDOWS_PER_TRIAL = 30  # 30s epoch / 1s window
N_FEATURES = N_CHANNELS * N_BANDS  # 150

# --- Classification ---
N_FOLDS = 10
RANDOM_SEED = 7
C_CANDIDATES = 10.0 ** np.arange(-5, 1, 0.5)  # 12 values
DECAY_RATE = 0.990

# --- LDS smoothing ---
LDS_V0 = 0.01
LDS_A = 1.0
LDS_T = 0.0001
LDS_C = 1.0
LDS_SIGMA = 1.0

# --- Labels ---
VIDEO_LABELS_9CLASS = (
    [0] * 3 + [1] * 3 + [2] * 3 + [3] * 3
    + [4] * 4
    + [5] * 3 + [6] * 3 + [7] * 3 + [8] * 3
)
CLASS_NAMES = [
    "Anger", "Disgust", "Fear", "Sadness", "Neutral",
    "Amusement", "Inspiration", "Joy", "Tenderness",
]
NEUTRAL_CLASS = CLASS_NAMES.index("Neutral")

# Single-video indices: one video per emotion (0-based)
SINGLE_VIDEO_INDICES = [0, 3, 6, 9, 12, 16, 19, 22, 25]

# Subjective label score order in After_remarks.mat (first 8 scores)
# Maps score index → 9-class emotion ID
SCORE_NAMES = ["Joy", "Tenderness", "Inspiration", "Amusement",
               "Anger", "Disgust", "Fear", "Sadness"]
SCORE_INDEX_TO_CLASS = [CLASS_NAMES.index(n) for n in SCORE_NAMES]


def fold_split(fold, n_total=N_SUBJECTS):
    """Return (train_indices, val_indices) for a given fold."""
    n_per = round(n_total / N_FOLDS)
    if fold < N_FOLDS - 1:
        val = np.arange(n_per * fold, n_per * (fold + 1))
    else:
        val = np.arange(n_per * fold, n_total)
    train = np.setdiff1d(np.arange(n_total), val)
    return train, val
