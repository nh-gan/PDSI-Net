# PDSI-Net：面向不规则多变量时间序列预测的分块离散频谱交互网络

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.6-EE4C2C?logo=pytorch&logoColor=white)
![Task](https://img.shields.io/badge/Task-IMTS%20Forecasting-6C8EBF)

## 🚀 简介

本仓库提供 **PDSI-Net** 的官方 PyTorch 实现，用于不规则多变量时间序列预测（Irregular Multivariate Time Series Forecasting, IMTS）。

PDSI-Net 在局部时间块内保留原始离散观测，通过分块—离散交互和掩码感知聚合学习细粒度局部模式，并利用频域增强建模跨变量全局依赖。

## 📦 快速入门

### 推荐环境

- Python 3.11
- PyTorch 2.6

### 1️⃣ 安装

创建 Conda 环境并安装依赖：

```bash
conda create -n PDSI-Net python=3.11 -y
conda activate PDSI-Net
pip install -r requirements.txt
```

### 2️⃣ 数据准备

当您第一次运行代码时，会自动下载并处理 **PhysioNet**、**USHCN** 和 **Human Activity** 数据集。

对于 **MIMIC** 数据集，需要执行以下手动预处理步骤：

1. 按照 `gru_ode_bayes` 中的预处理脚本生成 `complete_tensor.csv` 文件。
2. 将生成的文件放在以下路径下（您可能需要手动创建该目录）：

```text
~/.tsdm/rawdata/MIMIC_III_DeBrouwer2019/complete_tensor.csv
```

> 数据集文件不包含在本仓库中。

### 3️⃣ 训练

在项目根目录下运行对应数据集的训练脚本。脚本后的可选参数用于指定 GPU 编号：

```bash
bash scripts/PDSI-Net/P12.sh 0
bash scripts/PDSI-Net/USHCN.sh 0
bash scripts/PDSI-Net/MIMIC_III.sh 0
bash scripts/PDSI-Net/HumanActivity.sh 0
```

## 📁 仓库说明

本仓库仅提供 PDSI-Net 的主要模型、数据加载与训练代码，不包含数据集、运行日志、模型检查点、实验结果、Notebook、基线模型及消融实验代码。
