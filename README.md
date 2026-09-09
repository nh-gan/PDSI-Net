# PDSI-Net: Patch-wise Discrete Spectral Interaction Network for Irregular Multivariate Time Series Forecasting

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.6-EE4C2C?logo=pytorch&logoColor=white)
![Task](https://img.shields.io/badge/Task-IMTS%20Forecasting-6C8EBF)

## 🚀 Introduction

This repository provides the official PyTorch implementation of **PDSI-Net** for Irregular Multivariate Time Series Forecasting (IMTS).

PDSI-Net preserves raw discrete observations within local temporal patches, learns fine-grained local patterns through patch-discrete interaction and mask-aware aggregation, and models global cross-variable dependencies through frequency-domain enhancement.

## 🧠 Available Models

This repository includes the proposed method and four representative IMTS forecasting baselines:

- **PDSI-Net**
- **tPatchGNN**
- **Hi-Patch**
- **ASTGI**
- **APN**

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

ASTGI additionally requires Faiss. Install the GPU or CPU build according to your environment:

```bash
# GPU build
conda install -c pytorch -c nvidia -c conda-forge faiss-gpu=1.14.2

# CPU build
conda install -c pytorch -c conda-forge faiss-cpu=1.14.2
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

Training scripts follow the structure `scripts/<model>/<dataset>.sh`. Run the corresponding script from the project root; the optional argument specifies the GPU ID:

```bash
bash scripts/PDSI-Net/P12.sh 0
bash scripts/tPatchGNN/P12.sh 0
bash scripts/Hi_Patch/P12.sh 0
bash scripts/ASTGI/P12.sh 0
bash scripts/APN/P12.sh 0
```

Each model directory provides scripts for **PhysioNet (`P12`)**, **USHCN**, **MIMIC-III**, and **Human Activity**.

## 📁 Repository Contents

This repository contains PDSI-Net, the included baseline implementations, data-loading utilities, and training code. Datasets, execution logs, model checkpoints, experimental results, notebooks, and ablation-study code are not included.
