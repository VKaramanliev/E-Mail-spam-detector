from __future__ import annotations

from pathlib import Path

import pytest
from sklearn.pipeline import Pipeline

from spam_detector.models import TrainConfig, build_pipeline, load_model, predict_label, save_model
from spam_detector.train import train_model


@pytest.mark.parametrize("kind", ["logreg", "nb", "sgd"])
def test_build_pipeline_has_expected_steps(kind: str) -> None:
    cfg = TrainConfig(max_features=1000, random_state=42)
    model = build_pipeline(cfg, kind=kind)
    assert isinstance(model, Pipeline)

    assert "tfidf" in model.named_steps
    assert "clf" in model.named_steps

    tfidf = model.named_steps["tfidf"]
    clf = model.named_steps["clf"]

    assert hasattr(tfidf, "fit_transform")
    assert hasattr(clf, "fit")


def test_train_and_predict_label_and_proba_range() -> None:
    texts = [
        "Hi, are we meeting tomorrow?",
        "Limited offer!!! Click now to win prize",
        "Please review the attached invoice",
        "You won a free gift card, click here",
        "Let's grab lunch next week",
        "URGENT: claim your prize now!!!",
    ]
    labels = ["ham", "spam", "ham", "spam", "ham", "spam"]

    cfg = TrainConfig(max_features=5000, random_state=42)
    result = train_model(texts, labels, cfg)
    label, proba = predict_label(result.model, "win a prize now click here")

    assert label in {"spam", "ham"}
    assert 0.0 <= proba <= 1.0


def test_save_and_load_model_roundtrip(tmp_path: Path) -> None:
    texts = ["hello friend", "WIN money now", "see you tomorrow", "free prize click"]
    labels = ["ham", "spam", "ham", "spam"]

    cfg = TrainConfig(max_features=2000, random_state=42)
    result = train_model(texts, labels, cfg)

    out = tmp_path / "m.joblib"
    save_model(result.model, str(out))

    loaded = load_model(str(out))
    assert isinstance(loaded, Pipeline)

    label, proba = predict_label(loaded, "win money")
    assert label in {"spam", "ham"}
    assert 0.0 <= proba <= 1.0