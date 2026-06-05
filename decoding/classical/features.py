"""Load official pre-computed DE features from the FACED dataset.

Features are per-subject .pkl files with shape (28, 32, 30, 5).
We concatenate all subjects, drop mastoid channels, and reshape for
classification: (n_subjects, n_total_windows, n_features).
"""
import os
import pickle

import numpy as np

from decoding.config import (
    DE_DIR, N_SUBJECTS, N_VIDEOS, N_CHANNELS, CHANNELS_TO_DROP,
    N_BANDS, WINDOWS_PER_TRIAL, N_FEATURES, RESULTS_DIR,
)


def load_de_features() -> np.ndarray:
    """Load and reshape DE features from official FACED .pkl files.

    Returns:
        Array of shape (123, 840, 150) = (subjects, total_windows, features).
        Features are bands×channels flattened.
    """
    cache_path = RESULTS_DIR / "de_features.npy"
    if cache_path.exists():
        return np.load(cache_path)

    pkl_files = sorted(f for f in os.listdir(DE_DIR) if f.endswith(".pkl"))
    if len(pkl_files) != N_SUBJECTS:
        raise FileNotFoundError(
            f"Expected {N_SUBJECTS} .pkl files in {DE_DIR}, found {len(pkl_files)}"
        )

    # Each file: (28, 32, 30, 5) → (videos, channels, windows, bands)
    de_all = np.zeros((N_SUBJECTS, N_VIDEOS, N_CHANNELS, WINDOWS_PER_TRIAL, N_BANDS))
    for idx, fname in enumerate(pkl_files):
        with open(DE_DIR / fname, "rb") as f:
            de_sub = pickle.load(f)  # (28, 32, 30, 5)
        # Drop last 2 channels (mastoids)
        de_all[idx] = de_sub[:, :-CHANNELS_TO_DROP, :, :]

    # Reshape: (subs, videos, channels, windows, bands)
    #        → (subs, videos*windows, bands*channels)
    de_all = de_all.transpose(0, 1, 3, 4, 2)  # (subs, vids, windows, bands, ch)
    de_all = de_all.reshape(N_SUBJECTS, N_VIDEOS * WINDOWS_PER_TRIAL, N_FEATURES)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    np.save(cache_path, de_all)
    return de_all
