from data_loader import generate_or_load_data, prepare_pipeline_data
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

def run_training_pipeline():
    print("🔄 Loading dataset...")
    df = generate_or_load_data()

    print("🛠️ Preprocessing features...")
    X_train, X_test, y_train, y_test, _ = prepare_pipeline_data(df)

    # Model 1: Logistic Regression
    lr = LogisticRegression(random_state=42)
    lr.fit(X_train, y_train)
    lr_preds = lr.predict(X_test)
    lr_auc = roc_auc_score(y_test, lr.predict_proba(X_test)[:, 1])

    # Model 2: Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    rf_auc = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])

    print("\n--- 📊 Model Evaluation Results ---")
    print(f"Logistic Regression ROC-AUC: {lr_auc:.4f}")
    print(f"Random Forest ROC-AUC:       {rf_auc:.4f}\n")

    print("Classification Report (Random Forest):")
    print(classification_report(y_test, rf_preds))

if __name__ == "__main__":
    run_training_pipeline()