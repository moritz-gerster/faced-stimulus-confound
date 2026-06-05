"""Run the CLISA deep learning pipeline for cross-subject confound analyses.

Usage:
    DATA_DIR=/path/to/FACED python run_clisa.py                    # full pipeline
    DATA_DIR=/path/to/FACED python run_clisa.py --stage pretrain   # single stage
    DATA_DIR=/path/to/FACED python run_clisa.py --stage classify --analysis 2c
"""
import argparse
import time

import numpy as np
from scipy.stats import sem

from decoding.config import N_SUBJECTS, WINDOWS_PER_TRIAL
from decoding.concordance import run_concordance
from decoding.labels import load_subjective_labels


STAGES = ["pretrain", "extract", "norm", "lds", "classify", "all"]
ANALYSES = ["2a", "2b", "2c", "2d", "all"]


def _mean_pm_sem(scores):
    return f"{np.mean(scores):.1%} +/- {sem(scores):.1%}"


def _run_pretrain():
    from decoding.clisa.pretrain import run_pretrain
    run_pretrain(fold=None)


def _run_extract():
    from decoding.clisa.extract_features import run_extract
    run_extract()


def _run_norm():
    from decoding.clisa.running_norm import run_running_norm
    run_running_norm()


def _run_lds():
    from decoding.clisa.smooth_lds import run_lds_smoothing
    run_lds_smoothing()


def _run_classify(analysis: str):
    from decoding.clisa.classify import (
        run_clisa_baseline, run_clisa_subjective, run_clisa_single_video,
    )

    results = {}

    if analysis in ("2a", "2b", "all"):
        t = time.time()
        print("CLISA 2a. Cross-subject baseline...")
        baseline = run_clisa_baseline()
        results["2a"] = baseline
        print(f"  {_mean_pm_sem(baseline['scores'])} ({(time.time() - t) / 60:.1f} min)")

    if analysis in ("2b", "all"):
        t = time.time()
        print("CLISA 2b. Concordance/discordance split...")
        subjective_labels = load_subjective_labels()
        from decoding.config import VIDEO_LABELS_9CLASS
        bl = results.get("2a", baseline)
        labels = bl.get("labels")
        if labels is None:
            labels = np.tile(
                np.repeat(VIDEO_LABELS_9CLASS, WINDOWS_PER_TRIAL), (N_SUBJECTS, 1)
            )
        conc = run_concordance(bl["predictions"], labels, subjective_labels)
        results["2b"] = conc
        print(f"  Concordant: {_mean_pm_sem(conc['concordant_acc'])}, "
              f"Discordant: {_mean_pm_sem(conc['discordant_acc'])} "
              f"({(time.time() - t) / 60:.1f} min)")

    if analysis in ("2c", "all"):
        t = time.time()
        print("CLISA 2c. Subjective labels...")
        subjective_labels = load_subjective_labels()
        subj = run_clisa_subjective(subjective_labels)
        results["2c"] = subj
        labels_subj = subj.get(
            "labels", np.repeat(subjective_labels, WINDOWS_PER_TRIAL, axis=1)
        )
        conc_subj = run_concordance(
            subj["predictions"], labels_subj, subjective_labels
        )
        results["2c_conc"] = conc_subj
        print(f"  {_mean_pm_sem(subj['scores'])} "
              f"(concordant {_mean_pm_sem(conc_subj['concordant_acc'])}, "
              f"discordant {_mean_pm_sem(conc_subj['discordant_acc'])}) "
              f"({(time.time() - t) / 60:.1f} min)")

    if analysis in ("2d", "all"):
        t = time.time()
        print("CLISA 2d. Single video per emotion...")
        single = run_clisa_single_video()
        results["2d"] = single
        print(f"  {_mean_pm_sem(single['scores'])} ({(time.time() - t) / 60:.1f} min)")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="CLISA deep learning pipeline for FACED confound analysis"
    )
    parser.add_argument(
        "--stage", choices=STAGES, default="all",
        help="Run a single pipeline stage (default: all)",
    )
    parser.add_argument(
        "--analysis", choices=ANALYSES, default="all",
        help="Which classification analysis to run (only for classify stage)",
    )
    args = parser.parse_args()

    start_time = time.time()

    if args.stage in ("pretrain", "all"):
        t = time.time()
        print("=== Stage 1/5: SimCLR Pretraining ===")
        _run_pretrain()
        print(f"Pretraining done ({(time.time() - t) / 60:.1f} min)")

    if args.stage in ("extract", "all"):
        t = time.time()
        print("=== Stage 2/5: Feature Extraction ===")
        _run_extract()
        print(f"Feature extraction done ({(time.time() - t) / 60:.1f} min)")

    if args.stage in ("norm", "all"):
        t = time.time()
        print("=== Stage 3/5: Running Normalization ===")
        _run_norm()
        print(f"Running normalization done ({(time.time() - t) / 60:.1f} min)")

    if args.stage in ("lds", "all"):
        t = time.time()
        print("=== Stage 4/5: LDS Smoothing ===")
        _run_lds()
        print(f"LDS smoothing done ({(time.time() - t) / 60:.1f} min)")

    if args.stage in ("classify", "all"):
        t = time.time()
        print("=== Stage 5/5: Classification ===")
        results = _run_classify(args.analysis)
        print(f"Classification done ({(time.time() - t) / 60:.1f} min)")
    else:
        results = {}

    elapsed = time.time() - start_time

    if results:
        print()
        print("=" * 78)
        print("CLISA (Chen et al. 2023) — Cross-Subject Results")
        print("=" * 78)
        print()
        print(f"{'Analysis':<42} {'Accuracy (mean +/- SEM)':>20} {'Chance':>8}")
        print("-" * 78)
        if "2a" in results:
            print(f"  {'2a. Cross-subject baseline':<40} "
                  f"{_mean_pm_sem(results['2a']['scores']):>20} {'11.1%':>8}")
        if "2b" in results:
            print(f"  {'2b-i. Concordant trials':<40} "
                  f"{_mean_pm_sem(results['2b']['concordant_acc']):>20} {'11.1%':>8}")
            print(f"  {'2b-ii. Discordant trials':<40} "
                  f"{_mean_pm_sem(results['2b']['discordant_acc']):>20} {'11.1%':>8}")
        if "2c" in results:
            print(f"  {'2c. Subjective labels':<40} "
                  f"{_mean_pm_sem(results['2c']['scores']):>20} {'11.1%':>8}")
        if "2c_conc" in results:
            print(f"  {'2c-i. Subjective concordant':<40} "
                  f"{_mean_pm_sem(results['2c_conc']['concordant_acc']):>20} {'11.1%':>8}")
            print(f"  {'2c-ii. Subjective discordant':<40} "
                  f"{_mean_pm_sem(results['2c_conc']['discordant_acc']):>20} {'11.1%':>8}")
        if "2d" in results:
            print(f"  {'2d. Single video per emotion':<40} "
                  f"{_mean_pm_sem(results['2d']['scores']):>20} {'11.1%':>8}")
        print("=" * 78)

    print(f"\nTotal runtime: {elapsed / 60:.1f} min")


if __name__ == "__main__":
    main()
