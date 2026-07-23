from pathlib import Path
from typing import Optional, Tuple

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

from data_loader import generate_or_load_data, prepare_pipeline_data

try:
    from lightgbm import LGBMClassifier
except ImportError:  # pragma: no cover - optional dependency
    LGBMClassifier = None

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover - optional dependency
    XGBClassifier = None


def build_model(model_name: str, y_train) -> object:
    positive_count = int(y_train.sum())
    negative_count = int((y_train == 0).sum())
    scale_pos_weight = max(negative_count / positive_count, 1.0) if positive_count else 1.0

    if model_name == "logistic_regression":
        return LogisticRegression(random_state=42, class_weight="balanced", max_iter=5000)

    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
        )

    if model_name == "xgboost":
        if XGBClassifier is None:
            raise ImportError("xgboost is not installed in this environment")
        return XGBClassifier(
            objective="binary:logistic",
            eval_metric="auc",
            random_state=42,
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
        )

    if model_name == "lightgbm":
        if LGBMClassifier is None:
            raise ImportError("lightgbm is not installed in this environment")
        return LGBMClassifier(
            objective="binary",
            random_state=42,
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
        )

    raise ValueError(f"Unsupported model: {model_name}")


def evaluate_with_cv(model: object, X_train, y_train) -> Tuple[float, float]:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")
    return float(scores.mean()), float(scores.std())


def run_training_pipeline(model_name: str = "auto") -> None:
    print("🔄 Loading dataset...")
    df = generate_or_load_data()

    print("🛠️ Preprocessing features...")
    X_train, X_test, y_train, y_test, preprocessor = prepare_pipeline_data(df)

    candidate_models = []
    if model_name == "auto":
        candidate_models = ["logistic_regression", "random_forest"]
        if XGBClassifier is not None:
            candidate_models.append("xgboost")
        elif LGBMClassifier is not None:
            candidate_models.append("lightgbm")
    else:
        candidate_models = [model_name]

    results = []
    for candidate in candidate_models:
        try:
            model = build_model(candidate, y_train)
            cv_mean, cv_std = evaluate_with_cv(model, X_train, y_train)
            model.fit(X_train, y_train)
            test_proba = model.predict_proba(X_test)[:, 1]
            test_auc = roc_auc_score(y_test, test_proba)
            results.append((candidate, model, cv_mean, cv_std, test_auc))
            print(f"{candidate}: CV ROC-AUC={cv_mean:.4f} ± {cv_std:.4f} | Test ROC-AUC={test_auc:.4f}")
        except Exception as exc:
            print(f"{candidate} could not be trained: {exc}")

    if not results:
        raise RuntimeError("No models were successfully trained")

    best_name, best_model, _, _, best_auc = max(results, key=lambda item: item[4])
    best_model.fit(X_train, y_train)
    best_predictions = best_model.predict(X_test)

    print("\n--- 📊 Best Model Evaluation ---")
    print(f"Best model: {best_name}")
    print(f"Test ROC-AUC: {best_auc:.4f}")
    print(classification_report(y_test, best_predictions))

    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    artifact_path = models_dir / "best_model.pkl"
    artifact = {
        "model_name": best_name,
        "model": best_model,
        "preprocessor": preprocessor,
        "test_auc": best_auc,
    }
    joblib.dump(artifact, artifact_path)
    print(f"Saved best model artifact to {artifact_path}")


if __name__ == "__main__":
    run_training_pipeline()