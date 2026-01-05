# adult_features.py
from __future__ import annotations
from typing import Optional
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class AdultFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, drop_cols: Optional[list[str]] = None):
        self.drop_cols = drop_cols or ["fnlwgt", "education"]

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        for c in self.drop_cols:
            if c in df.columns:
                df.drop(columns=[c], inplace=True)

        for c in [
            "age",
            "education-num",
            "capital-gain",
            "capital-loss",
            "hours-per-week",
        ]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        for c in ["workclass", "occupation", "native-country"]:
            if c in df.columns:
                df[f"{c}_missing"] = df[c].isna().astype(int)
                df[c] = df[c].fillna("Unknown")

        gain = df["capital-gain"].fillna(0) if "capital-gain" in df.columns else 0
        loss = df["capital-loss"].fillna(0) if "capital-loss" in df.columns else 0

        df["capital_net"] = gain - loss
        df["has_capital"] = ((gain > 0) | (loss > 0)).astype(int)
        df["capital_gain_log1p"] = np.log1p(gain)
        df["capital_loss_log1p"] = np.log1p(loss)

        def hours_bucket(h):
            if pd.isna(h):
                return "Unknown"
            if h < 35:
                return "part_time"
            if h <= 45:
                return "full_time"
            return "overtime"

        if "hours-per-week" in df.columns:
            df["hours_bucket"] = df["hours-per-week"].apply(hours_bucket)

        obj_cols = df.select_dtypes(include="object").columns
        df[obj_cols] = df[obj_cols].apply(lambda s: s.astype(str).str.strip())
        return df
