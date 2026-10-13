"""Train the churn model and save artifacts to model/."""
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score, confusion_matrix,
                             roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).parent
DATA = ROOT / "data" / "churn.csv"
OUT = ROOT / "model"
OUT.mkdir(exist_ok=True)

NUMERIC = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL = ["gender", "Partner", "Dependents", "PhoneService",
               "InternetService", "OnlineSecurity", "TechSupport", "Contract",
               "PaperlessBilling", "PaymentMethod"]
FEATURES = NUMERIC + CATEGORICAL


def main():
    df = pd.read_csv(DATA)
    X, y = df[FEATURES], (df["Churn"] == "Yes").astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)

    pre = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])
    pipe = Pipeline([
        ("pre", pre),
        ("clf", GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.08, max_depth=3, random_state=42)),
    ])
    pipe.fit(X_tr, y_tr)

    proba = pipe.predict_proba(X_te)[:, 1]
    pred = (proba >= 0.5).astype(int)
    fpr, tpr, _ = roc_curve(y_te, proba)
    names = pipe.named_steps["pre"].get_feature_names_out()
    imp = pd.Series(pipe.named_steps["clf"].feature_importances_, index=names)
    imp = imp.sort_values(ascending=False).head(12)

    metrics = {
        "accuracy": accuracy_score(y_te, pred),
        "precision": precision_score(y_te, pred),
        "recall": recall_score(y_te, pred),
        "f1": f1_score(y_te, pred),
        "roc_auc": roc_auc_score(y_te, proba),
        "confusion_matrix": confusion_matrix(y_te, pred).tolist(),
        "roc_curve": {"fpr": fpr[::5].tolist(), "tpr": tpr[::5].tolist()},
        "feature_importance": {k.split("__", 1)[-1]: float(v) for k, v in imp.items()},
        "n_train": int(len(X_tr)), "n_test": int(len(X_te)),
    }
    joblib.dump(pipe, OUT / "model.joblib")
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print({k: round(v, 4) for k, v in metrics.items() if isinstance(v, float)})


if __name__ == "__main__":
    main()
