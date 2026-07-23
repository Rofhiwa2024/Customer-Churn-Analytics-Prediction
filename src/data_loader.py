from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def generate_or_load_data(filepath: Optional[str] = None) -> pd.DataFrame:
    """Load a CSV dataset or create a synthetic churn dataset when no file is provided."""
    if filepath:
        df = pd.read_csv(filepath)
    else:
        np.random.seed(42)
        n = 1000
        df = pd.DataFrame(
            {
                "tenure_months": np.random.randint(1, 72, size=n),
                "monthly_charges": np.random.uniform(20.0, 120.0, size=n),
                "contract_type": np.random.choice(["Month-to-month", "One year", "Two year"], size=n),
                "payment_method": np.random.choice(["Electronic check", "Mailed check", "Bank transfer"], size=n),
                "churn": np.random.choice([0, 1], size=n, p=[0.7, 0.3]),
            }
        )
    return df


def prepare_pipeline_data(
    df: pd.DataFrame,
    target_col: str = "churn",
) -> Tuple[np.ndarray, np.ndarray, pd.Series, pd.Series, ColumnTransformer]:
    """Create train/test splits and return preprocessed feature matrices for model training."""
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' was not found in dataframe")

    X = df.drop(columns=[target_col]).copy()
    y = df[target_col].copy()

    numeric_cols = ["tenure_months", "monthly_charges"]
    categorical_cols = ["contract_type", "payment_method"]

    X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())
    X[categorical_cols] = X[categorical_cols].fillna("Unknown")

    if len(X) < 10 or y.nunique() < 2:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=None,
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y,
        )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )

    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    return X_train_proc, X_test_proc, y_train, y_test, preprocessor