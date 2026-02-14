from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from spam_detector.evaluate import EvalResult, evaluate_model
from spam_detector.models import ModelKind, TrainConfig, build_pipeline


@dataclass(frozen=True)
class SplitData:
    x_train: list[str]
    y_train: list[str]
    x_val: list[str]
    y_val: list[str]
    x_test: list[str]
    y_test: list[str]


def split_train_val_test(
    texts: list[str],
    labels: list[str],
    *,
    random_state: int,
    test_size: float = 0.15,
    val_size: float = 0.15,
) -> SplitData:
    x_tmp, x_test, y_tmp, y_test = train_test_split(
        texts,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )

    val_size_rel = val_size / (1.0 - test_size)
    x_train, x_val, y_train, y_val = train_test_split(
        x_tmp,
        y_tmp,
        test_size=val_size_rel,
        random_state=random_state,
        stratify=y_tmp,
    )

    return SplitData(
        x_train=list(x_train),
        y_train=list(y_train),
        x_val=list(x_val),
        y_val=list(y_val),
        x_test=list(x_test),
        y_test=list(y_test),
    )


@dataclass(frozen=True)
class TrainResult:
    kind: ModelKind
    model: Pipeline
    val: EvalResult
    test: EvalResult
    val_loss_curve: list[float]


def _sgd_train_with_epochs(
    cfg: TrainConfig,
    split: SplitData,
    *,
    epochs: int = 12,
) -> tuple[Pipeline, list[float]]:    
    pipe = build_pipeline(cfg, kind="sgd")
    tfidf = pipe.named_steps["tfidf"]
    clf = pipe.named_steps["clf"]

    xtr = tfidf.fit_transform(split.x_train)
    xval = tfidf.transform(split.x_val)

    classes = np.array(["ham", "spam"], dtype=object)

    losses: list[float] = []

    for _ in range(int(epochs)):
        clf.partial_fit(xtr, split.y_train, classes=classes)

        if hasattr(clf, "predict_proba"):
            p = clf.predict_proba(xval)
            losses.append(float(log_loss(split.y_val, p, labels=list(classes))))
        else:
            preds = clf.predict(xval)
            acc = float(np.mean(np.array(preds) == np.array(split.y_val)))
            losses.append(float(1.0 - acc))

    return pipe, losses


def train_model(
    texts: list[str],
    labels: list[str],
    cfg: TrainConfig,
    *,
    kind: ModelKind = "logreg",
    sgd_epochs: int = 12,
) -> TrainResult:
    split = split_train_val_test(
        texts,
        labels,
        random_state=cfg.random_state,
        test_size=0.15,
        val_size=0.15,
    )

    val_loss_curve: list[float] = []

    if kind == "sgd":
        model, val_loss_curve = _sgd_train_with_epochs(cfg, split, epochs=sgd_epochs)
    else:
        model = build_pipeline(cfg, kind=kind)
        model.fit(split.x_train, split.y_train)

    val_res = evaluate_model(model, split.x_val, split.y_val)
    test_res = evaluate_model(model, split.x_test, split.y_test)

    return TrainResult(
        kind=kind,
        model=model,
        val=val_res,
        test=test_res,
        val_loss_curve=val_loss_curve,
    )


def train_all_models(
    texts: list[str],
    labels: list[str],
    cfg: TrainConfig,
    *,
    sgd_epochs: int = 12,
) -> list[TrainResult]:
    results: list[TrainResult] = []
    for kind in ("nb", "logreg", "sgd"):
        results.append(train_model(texts, labels, cfg, kind=kind, sgd_epochs=sgd_epochs))
    return results
