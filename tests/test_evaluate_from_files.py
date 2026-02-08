from __future__ import annotations

from pathlib import Path

from spam_detector.evaluate import evaluate_from_files
from spam_detector.models import TrainConfig, build_pipeline, save_model


def test_evaluate_from_files_roundtrip(tmp_path: Path):
    csv_path = tmp_path / "data.csv"
    csv_path.write_text(
        "text,label\n"
        "free money,spam\n"
        "hello,ham\n"
        "win prize,spam\n"
        "meeting,ham\n",
        encoding="utf-8",
    )

    cfg = TrainConfig()
    model = build_pipeline(cfg, kind="logreg")
    model.fit(["free money", "hello", "win prize", "meeting"], ["spam", "ham", "spam", "ham"])

    model_path = tmp_path / "model.joblib"
    save_model(model, str(model_path))

    res = evaluate_from_files(model_path, csv_path)
    assert 0.0 <= res.accuracy <= 1.0
    assert 0.0 <= res.f1_macro <= 1.0
    assert isinstance(res.confusion, list)
    assert len(res.confusion) == 2 and len(res.confusion[0]) == 2
    assert "ham" in res.report and "spam" in res.report