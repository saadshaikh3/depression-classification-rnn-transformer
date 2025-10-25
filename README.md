# 🧠 Student Depression Classification with LSTM & Transformer

## 🧭 Overview
This repository contains a **Jupyter/Colab notebook** (`main.ipynb`) that builds, trains, and evaluates two deep learning models for **binary depression classification** on a student dataset:
- **LSTM (RNN)** — sequential model with gated memory,
- **Transformer (encoder)** — attention-based model adapted for tabular inputs.

Both models operate on **structured features** (demographic, academic, lifestyle). Inputs are standardised and reshaped to `(samples, 1, features)` for sequence-capable architectures. **Class imbalance** is addressed via `class_weight`.

---

## 🚀 Quick Start

> Option A: **Google Colab** (recommended) — no local setup required  
> Option B: **Local Jupyter** — Python 3.10+ preferred

#### A) Run on Google Colab
1. Upload **`main.ipynb`** to Colab (or open from GitHub if pushing this repo).
2. Runtime → **Change runtime type** → Set **GPU** (optional).
3. Run all cells.

----

## 📦 Data
Dataset (Kaggle): Student Depression Dataset
URL: https://www.kaggle.com/datasets/hopesb/student-depression-dataset/data

The notebook downloads/loads the CSV (or you can upload it directly to the Colab session).

Basic preprocessing includes: missing value handling, categorical encoding, scaling, and reshaping to (N, 1, F).

Note: Ensure the CSV path in the notebook matches your environment (local path or Colab /content/…).

---

## 🧩 Notebook Contents (what you’ll run)
- Imports & Setup — TensorFlow, scikit-learn, pandas, etc.
- Data Loading & Preprocessing — imputation, encoding, scaling, reshape to (samples, 1, features).
- Train/Validation/Test Split — stratified 80/20 split; validation from train (20%).
- Class Weights — computed with sklearn.utils.class_weight.