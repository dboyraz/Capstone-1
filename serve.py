# serve.py
from __future__ import annotations

import os
from typing import Any, Dict, List, Union

import joblib
from fastapi import FastAPI
from pydantic import BaseModel

import adult_features  # noqa: F401  (needed so joblib can unpickle)
from predict import predict


MODEL_PATH = os.getenv("MODEL_PATH", "model.joblib")

app = FastAPI(title="Adult Income Prediction Service")
bundle = None


class PredictRequest(BaseModel):
    data: Union[Dict[str, Any], List[Dict[str, Any]]]


@app.on_event("startup")
def _load_model():
    global bundle
    bundle = joblib.load(MODEL_PATH)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": bundle is not None,
        "model_path": MODEL_PATH,
        "model_name": (bundle or {}).get("model_name"),
    }


@app.post("/predict")
def predict_endpoint(req: PredictRequest):
    return predict(bundle, req.data)
