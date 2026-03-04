from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

_MODEL_CACHE: dict[str, Any] = {}


def _candidate_paths() -> list[Path]:
    env_path = os.getenv("EEG_MODEL_PATH")
    candidates = [
        Path("data/models/eeg_familiarity_model.joblib"),
        Path("data/eeg_familiarity_model.joblib"),
        Path("eeg_familiarity_model.joblib"),
    ]
    if env_path:
        candidates.insert(0, Path(env_path))
    return candidates


def _load_model_bundle() -> dict[str, Any]:
    try:
        import joblib
    except ImportError as exc:
        raise RuntimeError("joblib is required to load pretrained EEG model") from exc

    if "bundle" in _MODEL_CACHE:
        return _MODEL_CACHE["bundle"]

    model_path = next((p for p in _candidate_paths() if p.exists()), None)
    if model_path is None:
        searched = ", ".join(str(p) for p in _candidate_paths())
        raise RuntimeError(f"Pretrained EEG model not found. Searched: {searched}")

    raw = joblib.load(model_path)

    bundle: dict[str, Any]
    if isinstance(raw, dict):
        bundle = raw
    else:
        bundle = {"model": raw}

    bundle.setdefault("model_path", str(model_path))
    _MODEL_CACHE["bundle"] = bundle
    return bundle


def _coerce_features(features: Any) -> np.ndarray:
    arr = np.asarray(features, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


def ml_classifier(features: np.ndarray) -> bool:
    bundle = _load_model_bundle()

    model = bundle.get("model") or bundle.get("classifier")
    scaler = bundle.get("scaler")
    threshold = float(bundle.get("threshold", os.getenv("EEG_MODEL_THRESHOLD", "0.5")))

    if model is None:
        raise RuntimeError("Loaded model bundle does not contain a model/classifier")

    X = _coerce_features(features)
    if scaler is not None:
        X = scaler.transform(X)

    if hasattr(model, "predict_proba"):
        proba = np.asarray(model.predict_proba(X), dtype=float)
        if proba.ndim == 2 and proba.shape[1] > 1:
            unfamiliar_prob = float(proba[0, 1])
        else:
            unfamiliar_prob = float(proba.ravel()[0])
        return unfamiliar_prob >= threshold

    pred = np.asarray(model.predict(X)).ravel()
    if pred.size == 0:
        raise RuntimeError("Model returned empty prediction")
    return bool(int(pred[0]))
