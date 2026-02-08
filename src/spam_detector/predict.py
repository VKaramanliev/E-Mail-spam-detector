from __future__ import annotations
from sklearn.pipeline import Pipeline
from spam_detector.models import predict_label


def classify_text(model: Pipeline, text: str) -> str:
    label, spam_proba = predict_label(model, text)
    return f"label={label} spam_proba={spam_proba:.4f}"