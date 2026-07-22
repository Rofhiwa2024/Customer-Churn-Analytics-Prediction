# [Project Title:Customer Churn Analytics & Prediction]

[![GitHub Codespaces](https://img.shields.io/badge/Open_in-GitHub_Codespaces-blue?logo=github)](https://github.com/codespaces/new)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-brightgreen.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview
* **Objective:** [Predict customer churn using machine learning to help subscription services retain users.]
* **Key Findings:** [Identified that users without automatic payment methods are 3x more likely to cancel within 90 days.]

---

## 📁 Repository Structure

```text
├── data/               # Raw and processed datasets (git-ignored if large)
├── notebooks/          # Exploratory Data Analysis (EDA) notebooks
│   └── 01_eda.ipynb    # Main analysis and visual insights
├── src/                # Modular Python scripts for data pipelines
│   ├── data_loader.py  # Utility functions for fetching & processing data
│   └── train.py        # Model training and evaluation script
├── .gitignore          # Excluded files (data files, venv, cache)
├── requirements.txt    # Project dependencies
└── README.md           # Project documentation
