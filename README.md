# Stimulus identity rather than emotion drives EEG classification on the FACED dataset

This repository contains code to reproduce the analyses demonstrating that both intra-subject and cross-subject emotion classification accuracy on the FACED dataset (Chen et al., 2023) primarily reflects stimulus identity and temporal autocorrelation rather than generalizable emotion representations.

## Key Findings

### Part 1: Intra-subject decoding

| Analysis | 9-class Accuracy | Chance |
|----------|:---:|:---:|
| 1a. Baseline (replication) | 58.8 ± 1.0% | 11.1% |
| 1b. Single video per emotion (1/3 data) | 71.3 ± 1.3% | 11.1% |

### Part 2: Cross-subject decoding (LinearSVC)

| Analysis | 9-class Accuracy | Chance |
|----------|:---:|:---:|
| 2a. Baseline (replication) | 39.4 ± 1.1% | 11.1% |
| 2b. Concordant trials | 33.9 ± 1.4% | 11.1% |
| 2b. Discordant trials | 36.0 ± 1.8% | 11.1% |
| 2c. Subjective labels (all trials) | 26.8 ± 0.9% | 11.1% |
| 2c. Subjective labels, concordant | 31.9 ± 1.4% | 11.1% |
| 2c. Subjective labels, discordant | 15.3 ± 1.2% | 11.1% |
| 2d. Single video per emotion (1/3 data) | 44.9 ± 1.6% | 11.1% |

### Part 2e: Cross-subject decoding (CLISA)

| Analysis | 9-class Accuracy | Chance |
|----------|:---:|:---:|
| Baseline | 34.2 ± 0.8% | 11.1% |
| Single video per emotion | 50.6 ± 1.8% | 11.1% |

**Interpretation:** The classifier decodes *which video was shown*, not *which emotion was felt*. Evidence:
- Performance *increases* when reducing to one video per emotion, eliminating within-class stimulus variance (1b, 2d)
- Performance is comparable for concordant vs. discordant trials (2b)
- Performance drops when labels reflect subjective reports rather than stimulus categories (2c)
- CLISA (deep learning) reproduces the same confound pattern (2e)

## References

**This paper:**

Gerster, M., et al. "Stimulus identity rather than emotion drives EEG classification on the FACED dataset." *Preprint forthcoming.* <!-- TODO: Update with arXiv link -->

**Original dataset paper:**

Chen, W., et al. "A Large Finer-grained Affective Computing EEG Dataset." *Scientific Data* 10, 740 (2023). DOI: [10.1038/s41597-023-02650-w](https://doi.org/10.1038/s41597-023-02650-w)

## Quick Start

```bash
conda env create -f environment.yml
conda activate faced-stimulus-confound
DATA_DIR=/path/to/FACED_Dataset python run.py
```

Download the FACED dataset from [Synapse (syn50614194)](https://doi.org/10.7303/syn50614194).
See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for full instructions including CLISA, figure generation, and cache behavior.

## Method

The pipeline replicates Chen et al. (2023) using their pre-computed DE features:
- **Features:** Differential Entropy, 5 frequency bands, 1-second windows
- **Normalization:** Running normalization (decay 0.990) + LDS Kalman smoothing
- **Classifiers:** LinearSVC (10-fold CV) and CLISA (contrastive learning)
- **Evaluation:** Per-subject accuracy, mean ± SEM across 123 subjects

## File Structure

```
run.py                          — Entry point: classical ML pipeline
run_clisa.py                    — Entry point: CLISA deep learning pipeline
decoding/
├── config.py                   — Shared parameters (matched to Chen et al. 2023)
├── labels.py                   — Label loading (crowd + subjective)
├── concordance.py              — Concordance/discordance analysis
├── classical/                  — SVM-based decoding
│   ├── classification.py
│   ├── features.py
│   └── normalization.py
└── clisa/                      — CLISA (contrastive learning) decoding
    ├── model.py                — CLISA model definition
    ├── classify.py             — Classification pipeline
    ├── pretrain.py             — Contrastive pre-training
    ├── load_data.py            — Data loading
    └── ...                     — Feature extraction, normalization, utilities
plotting/
├── fig1.py … fig5.py           — Figure scripts
├── panels.py                   — Shared schematic panels
├── data.py                     — Data loading for plots
└── style.py                    — Shared plot styling
```