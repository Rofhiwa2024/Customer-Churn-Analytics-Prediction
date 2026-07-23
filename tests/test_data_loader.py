import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from data_loader import generate_or_load_data, prepare_pipeline_data


def test_generate_or_load_data_returns_non_empty_dataframe():
    df = generate_or_load_data()

    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert {"tenure_months", "monthly_charges", "contract_type", "payment_method", "churn"}.issubset(df.columns)


def test_prepare_pipeline_data_removes_nulls_and_returns_expected_columns():
    df = pd.DataFrame(
        {
            "tenure_months": [1, 2, np.nan, 4],
            "monthly_charges": [20.0, 30.0, 40.0, np.nan],
            "contract_type": ["Month-to-month", None, "One year", "Two year"],
            "payment_method": ["Electronic check", "Mailed check", "Electronic check", "Bank transfer"],
            "churn": [0, 1, 0, 1],
        }
    )

    X_train, X_test, y_train, y_test, preprocessor = prepare_pipeline_data(df)

    assert X_train.shape[0] == y_train.shape[0]
    assert X_test.shape[0] == y_test.shape[0]
    assert X_train.shape[1] > 0
    assert np.isfinite(X_train).all()
    assert np.isfinite(X_test).all()
    assert preprocessor is not None


def test_prepare_pipeline_data_returns_expected_proportions():
    df = pd.DataFrame(
        {
            "tenure_months": [1, 2, 3, 4, 5, 6, 7, 8],
            "monthly_charges": [20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0],
            "contract_type": ["Month-to-month", "One year", "Month-to-month", "Two year", "Month-to-month", "One year", "Two year", "Two year"],
            "payment_method": ["Electronic check", "Mailed check", "Electronic check", "Bank transfer", "Electronic check", "Mailed check", "Bank transfer", "Bank transfer"],
            "churn": [0, 1, 0, 1, 0, 1, 0, 1],
        }
    )

    X_train, X_test, y_train, y_test, _ = prepare_pipeline_data(df)

    expected_train_size = int(round(len(df) * 0.8))
    expected_test_size = len(df) - expected_train_size

    assert X_train.shape[0] == expected_train_size
    assert X_test.shape[0] == expected_test_size
    assert y_train.shape[0] == expected_train_size
    assert y_test.shape[0] == expected_test_size


def test_prepare_pipeline_data_handles_unknown_categories():
    df = pd.DataFrame(
        {
            "tenure_months": [1, 2, 3, 4],
            "monthly_charges": [20.0, 30.0, 40.0, 50.0],
            "contract_type": ["Month-to-month", "One year", "Month-to-month", "Two year"],
            "payment_method": ["Electronic check", "Mailed check", "Electronic check", "Bank transfer"],
            "churn": [0, 1, 0, 1],
        }
    )

    X_train, X_test, y_train, y_test, _ = prepare_pipeline_data(df)

    assert X_train.shape[0] == y_train.shape[0]
    assert X_test.shape[0] == y_test.shape[0]
    assert X_train.shape[1] > 0
    assert np.isfinite(X_train).all()
    assert np.isfinite(X_test).all()
