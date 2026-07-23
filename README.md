# Customer Churn Analytics & Prediction

[![GitHub Codespaces](https://img.shields.io/badge/Open_in-GitHub_Codespaces-blue?logo=github)](https://github.com/codespaces/new)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-brightgreen.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview
Customer churn is a critical metric for subscription-based businesses. This project builds an end-to-end machine learning pipeline to identify high-risk customers, allowing teams to take proactive retention measures.

* **Key Finding:** Customers without automatic payment methods are 3x more likely to cancel within 90 days.
* **Best Model:** A churn model trained with preprocessing, class-imbalance handling, and cross-validation, with a test ROC-AUC of 0.5395.

---

## 📁 Repository Structure

```text
├── data/               # Raw and processed datasets (git-ignored if large)
├── notebooks/          # Exploratory Data Analysis (EDA) notebooks
│   └── 01_eda.ipynb    # Main analysis and visual insights
├── src/                # Modular Python scripts for data pipelines
│   ├── data_loader.py  # Utility functions for fetching, cleaning, and preprocessing data
│   └── train.py        # Model training and evaluation script
├── tests/              # Regression tests for data loading and preprocessing
├── .gitignore          # Excluded files (data files, venv, cache)
├── requirements.txt    # Project dependencies
└── README.md           # Project documentation
```

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/Rofhiwa2024/Customer-Churn-Analytics-Prediction.git
cd Customer-Churn-Analytics-Prediction
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Training Pipeline
```bash
python src/train.py
```

---

## 📊 Model Metrics

The latest training run produced the following evaluation metrics for the best model:

- ROC-AUC: 0.5395
- Accuracy: 0.54
- Precision: 0.34
- Recall: 0.57
- F1-score: 0.42

Cross-validation stability was also assessed with 5-fold stratified CV, yielding a mean ROC-AUC of 0.5110 ± 0.0310.
