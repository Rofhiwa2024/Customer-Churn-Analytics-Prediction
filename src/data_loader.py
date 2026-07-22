import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def generate_or_load_data(filepath=None):
    """Loads a CSV dataset or generates synthetic data if no path is provided."""
    if filepath:
        df = pd.read_csv(filepath)
    else:
        np.random.seed(42)
        n = 1000
        df = pd.DataFrame({
            'tenure_months': np.random.randint(1, 72, size=n),
            'monthly_charges': np.random.uniform(20.0, 120.0, size=n),
            'contract_type': np.random.choice(['Month-to-month', 'One year', 'Two year'], size=n),
            'payment_method': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer'], size=n),
            'churn': np.random.choice([0, 1], size=n, p=[0.7, 0.3])
        })
    return df

def prepare_pipeline_data(df, target_col='churn'):
    """Preprocesses features and returns train/test splits."""
    X = df.drop(columns=[target_col])
    y = df[target_col]

    num_cols = ['tenure_months', 'monthly_charges']
    cat_cols = ['contract_type', 'payment_method']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(drop='first'), cat_cols)
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    return X_train_proc, X_test_proc, y_train, y_test, preprocessor