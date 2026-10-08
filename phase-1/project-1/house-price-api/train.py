"""Train on data/houses.csv and serialize the pipeline to model/model.pkl.

Usage: python train.py [--data data/houses.csv]
"""
import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = ["area_sqft", "bedrooms", "bathrooms", "age_years", "location_score", "garage"]
TARGET = "price"


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = set(FEATURES + [TARGET]) - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing columns: {sorted(missing)}")
    before = len(df)
    df = df.drop_duplicates().dropna(subset=[TARGET])   # can't learn from rows without a price
    print(f"Loaded {before} rows, using {len(df)} after dropping duplicates/no-price rows")
    print("Missing values per feature:\n", df[FEATURES].isna().sum().to_string())
    return df


def main(data_path: str):
    df = load_data(data_path)
    X, y = df[FEATURES], df[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    # Imputer + scaler + model live in ONE pipeline, so model.pkl handles raw input at serving time
    prep = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    pipe = Pipeline([
        ("prep", ColumnTransformer([("num", prep, FEATURES)])),
        ("model", GradientBoostingRegressor(n_estimators=300, max_depth=3, random_state=42)),
    ])

    cv = cross_val_score(pipe, X_tr, y_tr, cv=5, scoring="r2")
    print(f"5-fold CV R2: {cv.mean():.4f} (+/- {cv.std():.4f})")

    pipe.fit(X_tr, y_tr)
    pred = pipe.predict(X_te)
    metrics = {"r2": round(r2_score(y_te, pred), 4),
               "mae": round(mean_absolute_error(y_te, pred), 0),
               "rows": len(df), "data_file": Path(data_path).name}
    print("Test metrics:", metrics)

    Path("model").mkdir(exist_ok=True)
    joblib.dump(pipe, "model/model.pkl")
    json.dump({"features": FEATURES, **metrics}, open("model/metadata.json", "w"), indent=2)
    print("Saved model/model.pkl")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/houses.csv")
    main(ap.parse_args().data)
