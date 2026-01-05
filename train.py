# train.py
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier

from adult_features import AdultFeatureEngineer


SEED = 42
TARGET = "income"
POS_LABEL = ">50K"
NEG_LABEL = "<=50K"

ADULT_COLS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
    "income",
]
INPUT_COLS = [c for c in ADULT_COLS if c != TARGET]

BEST_RF_PARAMS = {
    "clf__n_estimators": 600,
    "clf__min_samples_split": 2,
    "clf__min_samples_leaf": 2,
    "clf__max_features": "sqrt",
    "clf__max_depth": 24,
    "clf__class_weight": "balanced",
}

ARTIFACT_NAME = "model.joblib"


def load_adult(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(
        data_dir / "adult.data",
        header=None,
        names=ADULT_COLS,
        sep=",",
        skipinitialspace=True,
        na_values="?",
    )

    test_df = pd.read_csv(
        data_dir / "adult.test",
        header=None,
        names=ADULT_COLS,
        sep=",",
        skipinitialspace=True,
        na_values="?",
        skiprows=1,
    )
    test_df["income"] = test_df["income"].astype(str).str.replace(".", "", regex=False)

    for df in (train_df, test_df):
        obj_cols = df.select_dtypes(include="object").columns
        df[obj_cols] = df[obj_cols].apply(lambda s: s.astype(str).str.strip())

    return train_df, test_df


def make_y(df: pd.DataFrame) -> np.ndarray:
    return (df[TARGET] == POS_LABEL).astype(int).values


def calc_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, thr: float = 0.5
) -> Dict[str, float]:
    y_pred = (y_prob >= thr).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "f1": float(f1_score(y_true, y_pred)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    data_dir = project_dir / "dataset"
    artifact_path = project_dir / ARTIFACT_NAME

    train_df, test_df = load_adult(data_dir)

    X_train = train_df[INPUT_COLS]
    y_train = make_y(train_df)

    X_test = test_df[INPUT_COLS]
    y_test = make_y(test_df)

    # Determine numeric/categorical cols after FE
    fe = AdultFeatureEngineer()
    X_sample = fe.transform(X_train.head(5000))
    num_cols = X_sample.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X_sample.select_dtypes(include=["object"]).columns.tolist()

    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imp", SimpleImputer(strategy="median"))]), num_cols),
            (
                "cat",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="most_frequent")),
                        ("ohe", ohe),
                    ]
                ),
                cat_cols,
            ),
        ],
        remainder="drop",
    )

    rf_kwargs = {k.replace("clf__", ""): v for k, v in BEST_RF_PARAMS.items()}

    model = Pipeline(
        steps=[
            ("fe", AdultFeatureEngineer()),
            ("prep", preprocessor),
            ("clf", RandomForestClassifier(random_state=SEED, n_jobs=-1, **rf_kwargs)),
        ]
    )

    model.fit(X_train, y_train)

    test_prob = model.predict_proba(X_test)[:, 1]
    test_metrics = calc_metrics(y_test, test_prob, thr=0.5)

    bundle: Dict[str, Any] = {
        "model": model,
        "model_name": "RandomForest",
        "threshold": 0.5,
        "pos_label": POS_LABEL,
        "neg_label": NEG_LABEL,
        "input_columns": INPUT_COLS,
        "best_params": rf_kwargs,
        "test_metrics": test_metrics,
        "trained_at_unix": int(time.time()),
    }

    joblib.dump(bundle, artifact_path)
    print(f"Saved artifact: {artifact_path}")
    print(json.dumps(test_metrics, indent=2))


if __name__ == "__main__":
    main()
