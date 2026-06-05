# Reproducibility

## Data

Download the FACED dataset from [Synapse (syn50614194)](https://doi.org/10.7303/syn50614194).

Expected directory layout:

```text
FACED_Dataset/
├── EEG_Features/
│   └── DE/
│       └── sub###.pkl
├── Data/
│   └── sub###/
│       └── After_remarks.mat
└── Clisa_data/
    └── sub###.pkl
```

`Clisa_data/` is only needed for `run_clisa.py`.

## Environments

Classical SVM and plotting:

```bash
conda env create -f environment.yml
conda activate faced-stimulus-confound
```

CLISA (requires CUDA):

```bash
conda env create -f environment-clisa.yml
conda activate faced-stimulus-confound-clisa
```

If your cluster uses a different CUDA/PyTorch module stack, install the
matching PyTorch build and keep remaining versions aligned with
`environment-clisa.yml`.

## Commands

Full classical pipeline:

```bash
DATA_DIR=/path/to/FACED_Dataset python run.py
```

Single analysis (after preprocessing):

```bash
DATA_DIR=/path/to/FACED_Dataset python run.py --analysis 2c
```

CLISA:

```bash
DATA_DIR=/path/to/FACED_Dataset python run_clisa.py
```

Figures (from existing results):

```bash
python -m plotting.fig1
python -m plotting.fig2
python -m plotting.fig3
python -m plotting.fig4
python -m plotting.fig5
```

## Cache Behavior

Results are cached under `results/` with adjacent metadata JSON files. Caches
are reused only when metadata (data paths, shapes, parameters) matches. To
rebuild after changing data or settings:

```bash
rm -rf results/de_features.npy results/smooth/
DATA_DIR=/path/to/FACED_Dataset python run.py
```

## Notes

**Validation protocol.** SVM `C` selection and CLISA early stopping use the
held-out fold for model selection, following Chen et al. for direct
replication. Results should not be interpreted as nested-CV estimates.

**Determinism.** Random seed is `7`. Classical results are deterministic given
identical input files and package versions. CLISA requests deterministic cuDNN
behavior, but exact GPU reproducibility depends on hardware and drivers.
