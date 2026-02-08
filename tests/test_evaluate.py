from __future__ import annotations

from pathlib import Path

from spam_detector.evaluate import evaluate_model, format_evaluation, save_text_report
from spam_detector.models import TrainConfig
from spam_detector.train import train_model


def _trained_model():
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
    return train_model(texts, labels, cfg).model


def test_evaluate_model_returns_metric_shapes_and_valid_confusion_sum() -> None:
    model = _trained_model()

    texts = ["hello friend", "free prize click now", "see you tomorrow", "win money now"]
    labels = ["ham", "spam", "ham", "spam"]

    res = evaluate_model(model, texts, labels)

    assert 0 <= res.accuracy <= 1
    assert 0 <= res.f1_macro <= 1
    assert isinstance(res.report, str)

    assert len(res.confusion) == 2
    assert len(res.confusion[0]) == 2
    assert len(res.confusion[1]) == 2

    cm_sum = sum(sum(row) for row in res.confusion)
    assert cm_sum == len(labels)

    assert "ham" in res.report
    assert "spam" in res.report


def test_format_evaluation_contains_key_parts() -> None:
    model = _trained_model()
    res = evaluate_model(model, ["hello", "win money"], ["ham", "spam"])

    txt = format_evaluation(res)
    assert "accuracy=" in txt
    assert "f1_macro=" in txt
    assert "Confusion matrix" in txt
    assert "Classification report" in txt


def test_save_text_report_writes_file(tmp_path: Path) -> None:
    model = _trained_model()
    res = evaluate_model(model, ["hello", "win money"], ["ham", "spam"])

    out = tmp_path / "reports" / "eval.txt"
    save_text_report(res, out)

    assert out.exists()
    content = out.read_text(encoding="utf-8")
    assert "accuracy=" in content
