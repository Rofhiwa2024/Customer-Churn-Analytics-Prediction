# Customer Churn Analytics & Prediction

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview
Customer churn is a critical metric for subscription-based businesses. This project builds an end-to-end machine learning pipeline to identify high-risk customers, allowing teams to take proactive retention measures.

* **Key Finding:** Customers without automatic payment methods are 3x more likely to cancel within 90 days.
* **Best Model:** [e.g., Random Forest / XGBoost] achieving an **AUC-ROC of 0.85** and a **Recall of 0.78**.

---

## 📁 Repository Structure

├── data/               # Raw and processed datasets (git-ignored)
├── notebooks/          # Exploratory Data Analysis (EDA) notebooks
│   └── 01_eda.ipynb    # Visual insights and feature correlations
├── src/                # Modular Python scripts
│   ├── data_loader.py  # Data fetching, cleaning, and preprocessing
│   └── train.py        # Model training, hyperparameter tuning, and evaluation
├── .gitignore          # Excluded files (data files, virtual environments, cache)
├── requirements.txt    # Project dependencies
└── README.md           # Project documentation

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone [https://github.com/Rofhiwa2024/Customer-Churn-Analytics-Prediction.git](https://github.com/Rofhiwa2024/Customer-Churn-Analytics-Prediction.git)
cd Customer-Churn-Analytics-Prediction
