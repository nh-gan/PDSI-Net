# PDSI-Net: Patch-wise Discrete Spectral Interaction Network for Irregular Multivariate Time Series Forecasting

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.6-EE4C2C?logo=pytorch&logoColor=white)
![Task](https://img.shields.io/badge/Task-IMTS%20Forecasting-6C8EBF)

## 🚀 Introduction

This repository provides the official PyTorch implementation of **PDSI-Net** for Irregular Multivariate Time Series Forecasting (IMTS).

PDSI-Net preserves raw discrete observations within local temporal patches, learns fine-grained local patterns through patch-discrete interaction and mask-aware aggregation, and models global cross-variable dependencies through frequency-domain enhancement.

## 📦 Getting Started

### Recommended Environment

- Python 3.11
- PyTorch 2.6

### 1️⃣ Installation

Create a Conda environment and install the required dependencies:

```bash
conda create -n PDSI-Net python=3.11 -y
conda activate PDSI-Net
pip install -r requirements.txt
```

### 2️⃣ Data Preparation

When the code is run for the first time, the **PhysioNet**, **USHCN**, and **Human Activity** datasets will be downloaded and processed automatically.

The **MIMIC** dataset requires the following manual preprocessing steps:

1. Follow the preprocessing scripts provided in `gru_ode_bayes` to generate the `complete_tensor.csv` file.
2. Place the generated file at the following path. You may need to create the directory manually:

```text
~/.tsdm/rawdata/MIMIC_III_DeBrouwer2019/complete_tensor.csv
```

> Dataset files are not included in this repository.

### 3️⃣ Training

Run the corresponding training script from the project root. The optional argument specifies the GPU ID:

```bash
bash scripts/PDSI-Net/P12.sh 0
bash scripts/PDSI-Net/USHCN.sh 0
bash scripts/PDSI-Net/MIMIC_III.sh 0
bash scripts/PDSI-Net/HumanActivity.sh 0
```

## 📁 Repository Contents

This repository contains only the core PDSI-Net model, data-loading utilities, and training code. Datasets, execution logs, model checkpoints, experimental results, notebooks, baseline models, and ablation-study code are not included.
