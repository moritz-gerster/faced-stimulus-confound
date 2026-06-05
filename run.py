"""Main entry point: run all analyses and print summary table.

Usage:
    DATA_DIR=/path/to/FACED python run.py              # run all
    DATA_DIR=/path/to/FACED python run.py --analysis 1b # run single analysis
"""
import argparse
import time

import numpy as np
from scipy.stats import sem

from decoding.classical.classification import (
    run_intra_subject, run_intra_single_video,
    run_baseline, run_subjective, run_single_video,
)
from decoding.concordance import run_concordance
from decoding.config import RESULTS_DIR
from decoding.classical.features import load_de_features
from decoding.labels import load_subjective_labels
from decoding.classical.normalization import run_normalization

VALID_ANALYSES = ["1a", "1b", "2a", "2b", "2c", "2d"]


def _mean_pm_sem(scores):
    """Format mean +/- SEM as percentage string, e.g. '35.2% +/- 1.0%'."""
    return f"{np.mean(scores):.1%} +/- {sem(scores):.1%}"


def _run_single(analysis_id: str) -> None:
    """Run a single analysis (smoothed features must already exist)."""
    t = time.time()

    if analysis_id == "1a":
        result = run_intra_subject()
        print(f"1a. Intra-subject baseline — {_mean_pm_sem(result['scores'])}")
    elif analysis_id == "1b":
        result = run_intra_single_video()
        print(f"1b. Single video per emotion (intra) — {_mean_pm_sem(result['scores'])}")
    elif analysis_id == "2a":
        result = run_baseline()
        print(f"2a. Cross-subject baseline — {_mean_pm_sem(result['scores'])}")
    elif analysis_id == "2b":
        saved = np.load(str(RESULTS_DIR / "baseline" / "predictions.npz"))
        subjective_labels = load_subjective_labels()
        conc = run_concordance(saved["predictions"], saved["labels"],
                               subjective_labels)
        print(f"2b. Concordant: {_mean_pm_sem(conc['concordant_acc'])}, "
              f"Discordant: {_mean_pm_sem(conc['discordant_acc'])}")
    elif analysis_id == "2c":
        subjective_labels = load_subjective_labels()
        result = run_subjective(subjective_labels)
        print(f"2c. Subjective labels — {_mean_pm_sem(result['scores'])}")
        conc_subj = run_concordance(
            result["predictions"], result["labels"], subjective_labels
        )
        print(f"  Concordant: {_mean_pm_sem(conc_subj['concordant_acc'])}, "
              f"Discordant: {_mean_pm_sem(conc_subj['discordant_acc'])}")
    elif analysis_id == "2d":
        result = run_single_video()
        print(f"2d. Single video per emotion — {_mean_pm_sem(result['scores'])}")
    print(f"Runtime: {(time.time() - t) / 60:.1f} min")


def _run_all() -> None:
    """Run preprocessing + all analyses, print summary table and report."""
    start_time = time.time()
    t = time.time()

    # --- Step 1: Load DE features ---
    print("Loading DE features...")
    de_features = load_de_features()
    print(f"  Shape: {de_features.shape} ({(time.time() - t) / 60:.1f} min)")

    # --- Step 2: Run normalization + LDS smoothing ---
    t = time.time()
    print("Running normalization and LDS smoothing...")
    run_normalization(de_features)
    print(f"  Done. ({(time.time() - t) / 60:.1f} min)")

    # ==================================================================
    # Part 1 — Intra-subject analyses
    # ==================================================================

    # --- 1a. Intra-subject baseline (within-video time split) ---
    t = time.time()
    print("1a. Intra-subject baseline (within-video time split)...")
    intra = run_intra_subject()
    print(f"  {_mean_pm_sem(intra['scores'])} ({(time.time() - t) / 60:.1f} min)")

    # --- 1b. Single video per emotion (intra-subject) ---
    t = time.time()
    print("1b. Single video per emotion (intra-subject)...")
    single_intra = run_intra_single_video()
    print(f"  {_mean_pm_sem(single_intra['scores'])} ({(time.time() - t) / 60:.1f} min)")

    # ==================================================================
    # Part 2 — Cross-subject analyses
    # ==================================================================

    # --- 2a. Cross-subject baseline ---
    t = time.time()
    print("2a. Cross-subject baseline classification...")
    baseline = run_baseline()
    print(f"  {_mean_pm_sem(baseline['scores'])} ({(time.time() - t) / 60:.1f} min)")

    # --- 2b. Concordance ---
    t = time.time()
    print("2b. Concordance/discordance split...")
    subjective_labels = load_subjective_labels()
    conc = run_concordance(
        baseline["predictions"], baseline["labels"], subjective_labels
    )
    print(f"  Concordant: {_mean_pm_sem(conc['concordant_acc'])}, "
          f"Discordant: {_mean_pm_sem(conc['discordant_acc'])} ({(time.time() - t) / 60:.1f} min)")

    # --- 2c. Subjective labels ---
    t = time.time()
    print("2c. Subjective labels classification...")
    subj = run_subjective(subjective_labels)
    conc_subj = run_concordance(
        subj["predictions"], subj["labels"], subjective_labels
    )
    print(f"  {_mean_pm_sem(subj['scores'])} "
          f"(concordant {_mean_pm_sem(conc_subj['concordant_acc'])}, "
          f"discordant {_mean_pm_sem(conc_subj['discordant_acc'])}) "
          f"({(time.time() - t) / 60:.1f} min)")

    # --- 2d. Single video per emotion ---
    t = time.time()
    print("2d. Single video per emotion (9 videos)...")
    single_vid = run_single_video()
    print(f"  {_mean_pm_sem(single_vid['scores'])} ({(time.time() - t) / 60:.1f} min)")

    elapsed = time.time() - start_time

    # --- Summary table ---
    print()
    print("=" * 78)
    print("FACED Stimulus-Identity Confound Analysis — Results")
    print("=" * 78)
    print()
    print(f"{'Analysis':<42} {'Accuracy (mean +/- SEM)':>20} {'Chance':>8}")
    print("-" * 78)
    print("Part 1 — Intra-subject")
    print(f"  {'1a. Intra-subject baseline':<40} {_mean_pm_sem(intra['scores']):>20} {'11.1%':>8}")
    print(f"  {'1b. Single video per emotion (intra)':<40} {_mean_pm_sem(single_intra['scores']):>20} {'11.1%':>8}")
    print("Part 2 — Cross-subject")
    print(f"  {'2a. Cross-subject baseline':<40} {_mean_pm_sem(baseline['scores']):>20} {'11.1%':>8}")
    print(f"  {'2b-i. Concordant trials':<40} {_mean_pm_sem(conc['concordant_acc']):>20} {'11.1%':>8}")
    print(f"  {'2b-ii. Discordant trials':<40} {_mean_pm_sem(conc['discordant_acc']):>20} {'11.1%':>8}")
    print(f"  {'2c. Subjective labels':<40} {_mean_pm_sem(subj['scores']):>20} {'11.1%':>8}")
    print(f"  {'2c-i. Subjective concordant':<40} {_mean_pm_sem(conc_subj['concordant_acc']):>20} {'11.1%':>8}")
    print(f"  {'2c-ii. Subjective discordant':<40} {_mean_pm_sem(conc_subj['discordant_acc']):>20} {'11.1%':>8}")
    print(f"  {'2d. Single video per emotion':<40} {_mean_pm_sem(single_vid['scores']):>20} {'11.1%':>8}")
    print("=" * 78)
    print(f"\nTotal runtime: {elapsed / 60:.1f} min")

    # --- Write report ---
    report = (
        "# Results\n\n"
        "| Analysis | 9-class Accuracy (mean +/- SEM) | Chance |\n"
        "|----------|:---:|:---:|\n"
        "| **Part 1 — Intra-subject** | | |\n"
        f"| 1a. Intra-subject baseline | {_mean_pm_sem(intra['scores'])} | 11.1% |\n"
        f"| 1b. Single video per emotion (intra) | {_mean_pm_sem(single_intra['scores'])} | 11.1% |\n"
        "| **Part 2 — Cross-subject** | | |\n"
        f"| 2a. Cross-subject baseline | {_mean_pm_sem(baseline['scores'])} | 11.1% |\n"
        f"| 2b-i. Concordant trials | {_mean_pm_sem(conc['concordant_acc'])} | 11.1% |\n"
        f"| 2b-ii. Discordant trials | {_mean_pm_sem(conc['discordant_acc'])} | 11.1% |\n"
        f"| 2c. Subjective labels | {_mean_pm_sem(subj['scores'])} | 11.1% |\n"
        f"| 2c-i. Subjective concordant | {_mean_pm_sem(conc_subj['concordant_acc'])} | 11.1% |\n"
        f"| 2c-ii. Subjective discordant | {_mean_pm_sem(conc_subj['discordant_acc'])} | 11.1% |\n"
        f"| 2d. Single video per emotion | {_mean_pm_sem(single_vid['scores'])} | 11.1% |\n"
    )
    report_path = RESULTS_DIR / "report.md"
    report_path.write_text(report)
    print(f"Report saved to: {report_path.resolve()}")


def main():
    parser = argparse.ArgumentParser(
        description="FACED stimulus-identity confound analysis"
    )
    parser.add_argument(
        "--analysis", choices=VALID_ANALYSES, default=None,
        help="Run a single analysis instead of the full pipeline",
    )
    args = parser.parse_args()

    if args.analysis:
        _run_single(args.analysis)
    else:
        _run_all()


if __name__ == "__main__":
    main()
