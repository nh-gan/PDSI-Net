# PDSI-Net

Official implementation of **PDSI-Net** for irregular multivariate time-series forecasting.

PDSI-Net preserves discrete observations within local temporal patches, aggregates stable local patterns, and enhances cross-variable dependencies in the frequency domain.

## Environment

The code was tested with Python 3.11 and PyTorch 2.6.

```bash
conda create -n PDSI-Net python=3.11 -y
conda activate PDSI-Net
pip install -r requirements.txt
```

## Data preparation

Datasets are not included in this repository. Place the processed datasets under:

```text
storage/datasets/
├── HumanActivity/
├── MIMIC_III_DeBrouwer2019/
├── Physionet2012/
└── USHCN_DeBrouwer2019/
```

## Training

Run a training script from the project root. The optional argument selects the GPU ID.

```bash
bash scripts/PDSI-Net/P12.sh 0
bash scripts/PDSI-Net/USHCN.sh 0
bash scripts/PDSI-Net/MIMIC_III.sh 0
bash scripts/PDSI-Net/HumanActivity.sh 0
```

Training outputs, checkpoints, generated configurations, datasets, logs, and ablation experiments are intentionally excluded from this repository.
