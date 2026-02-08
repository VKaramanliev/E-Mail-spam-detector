from __future__ import annotations

import math

from spam_detector.models import TrainConfig
from spam_detector.train import train_all_models, train_model


def _toy_data() -> tuple[list[str], list[str]]:
    texts = [
        "Hi, are we meeting tomorrow?",
        "Limited offer!!! Click now to win prize",
        "Please review the attached invoice",
        "You won a free gift card, click here",
        "Let's grab lunch next week",
        "URGENT: claim your prize now!!!",
        "see you tomorrow",
        "win money now click",
    ]
    labels = ["ham", "spam", "ham", "spam", "ham", "spam", "ham", "spam"]
    return texts, labels


def test_train_model_nb_and_logreg_work() -> None:
    texts, labels = _toy_data()
    cfg = TrainConfig(max_features=5000, random_state=42)

    nb_res = train_model(texts, labels, cfg, kind="nb")
    assert nb_res.kind == "nb"
    assert 0 <= nb_res.test.accuracy <= 1

    lr_res = train_model(texts, labels, cfg, kind="logreg")
    assert lr_res.kind == "logreg"
    assert 0 <= lr_res.test.f1_macro <= 1


def test_train_model_sgd_produces_loss_curve() -> None:
    texts, labels = _toy_data()
    cfg = TrainConfig(max_features=5000, random_state=42)

    epochs = 6
    res = train_model(texts, labels, cfg, kind="sgd", sgd_epochs=epochs)

    assert res.kind == "sgd"
    assert len(res.val_loss_curve) == epochs
    assert all(math.isfinite(v) and v >= 0 for v in res.val_loss_curve)


def test_train_all_models_returns_all_kinds() -> None:
    texts, labels = _toy_data()
    cfg = TrainConfig(max_features=5000, random_state=42)

    results = train_all_models(texts, labels, cfg, sgd_epochs=5)
    kinds = {r.kind for r in results}
    assert kinds == {"nb", "logreg", "sgd"}
