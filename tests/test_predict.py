from __future__ import annotations

from spam_detector.models import TrainConfig
from spam_detector.train import train_model
from spam_detector.predict import classify_text

def test_classify_text_format() -> None:
    texts = ["hello friend", "WIN money now", "see you tomorrow", "free prize click"]
    labels = ["ham", "spam", "ham", "spam"]
    cfg = TrainConfig(max_features=2000)
    result = train_model(texts, labels, cfg)

    out = classify_text(result.model, "win money")
    assert out.startswith("label=")
    assert "spam_proba=" in out

def test_classify_text_contains_label_value() -> None:
    texts = ["hello friend", "WIN money now", "see you tomorrow", "free prize click"]
    labels = ["ham", "spam", "ham", "spam"]

    cfg = TrainConfig(max_features=2000)
    result = train_model(texts, labels, cfg)

    out = classify_text(result.model, "hello")
    assert ("label=ham" in out) or ("label=spam" in out)