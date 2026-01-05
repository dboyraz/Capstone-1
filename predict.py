# predict.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Union

import joblib
import numpy as np
import pandas as pd

import adult_features  # noqa: F401  (needed so joblib can unpickle)


def load_bundle(model_path: Path) -> dict:
    return joblib.load(model_path)


def ensure_dataframe(
    payload: Union[Dict[str, Any], List[Dict[str, Any]]],
    input_columns: List[str],
) -> pd.DataFrame:
    rows = [payload] if isinstance(payload, dict) else payload
    df = pd.DataFrame(rows)

    for c in input_columns:
        if c not in df.columns:
            df[c] = np.nan

    return df[input_columns]


def predict(bundle: dict, payload: Union[Dict[str, Any], List[Dict[str, Any]]]) -> dict:
    single = isinstance(payload, dict)

    model = bundle["model"]
    thr = float(bundle.get("threshold", 0.5))
    pos_label = bundle.get("pos_label", ">50K")
    neg_label = bundle.get("neg_label", "<=50K")
    input_columns = bundle["input_columns"]

    X = ensure_dataframe(payload, input_columns)
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= thr).astype(int)
    labels = [pos_label if p == 1 else neg_label for p in pred]

    result = {
        "model_name": bundle.get("model_name", "model"),
        "threshold": thr,
        "pred": pred.tolist(),
        "label": labels,
        "proba_gt_50k": [float(p) for p in proba],
    }

    if single:
        result["pred"] = result["pred"][0]
        result["label"] = result["label"][0]
        result["proba_gt_50k"] = result["proba_gt_50k"][0]

    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="model.joblib", help="Path to model artifact")
    ap.add_argument(
        "--input", default=None, help="Path to JSON file (dict or list[dict])"
    )
    args = ap.parse_args()

    bundle = load_bundle(Path(args.model))

    if args.input:
        payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    else:
        payload = {
            "age": 39,
            "workclass": "State-gov",
            "fnlwgt": 77516,
            "education": "Bachelors",
            "education-num": 13,
            "marital-status": "Never-married",
            "occupation": "Adm-clerical",
            "relationship": "Not-in-family",
            "race": "White",
            "sex": "Male",
            "capital-gain": 2174,
            "capital-loss": 0,
            "hours-per-week": 40,
            "native-country": "United-States",
        }

    out = predict(bundle, payload)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
