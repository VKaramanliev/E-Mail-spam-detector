from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

ModelKind = Literal["logreg", "nb", "sgd"]


@dataclass(frozen=True)
class TrainConfig:
    max_features: int = 50_000
    ngram_range: tuple[int, int] = (1, 2)
    random_state: int = 42

    # LogReg
    c: float = 2.0
    max_iter: int = 2000

    # Naive Bayes
    nb_alpha: float = 1.0

    # SGD
    sgd_alpha: float = 1e-5
    sgd_lr: str = "optimal"
    sgd_eta0: float = 0.001


def build_pipeline(cfg: TrainConfig, kind: ModelKind = "logreg") -> Pipeline:
    vectorizer = TfidfVectorizer(
        lowercase=True,
        max_features=cfg.max_features,
        ngram_range=cfg.ngram_range,
        min_df=2,
    )

    if kind == "logreg":
        clf = LogisticRegression(
            C=cfg.c,
            max_iter=cfg.max_iter,
            random_state=cfg.random_state,
            n_jobs=None,
        )
    elif kind == "nb":
        clf = MultinomialNB(alpha=cfg.nb_alpha)
    elif kind == "sgd":
        clf = SGDClassifier(
            loss="log_loss",
            alpha=cfg.sgd_alpha,
            learning_rate=cfg.sgd_lr,
            eta0=cfg.sgd_eta0,
            random_state=cfg.random_state,
        )
    else:
        raise ValueError(f"Unknown model kind: {kind}")

    return Pipeline([("tfidf", vectorizer), ("clf", clf)])


def save_model(model: Pipeline, path: str) -> None:
    joblib.dump(model, path)


def load_model(path: str) -> Pipeline:
    obj: Any = joblib.load(path)
    if not isinstance(obj, Pipeline):
        raise TypeError("The object is not a sklearn Pipeline")
    return obj


def predict_label(model: Pipeline, text: str) -> tuple[str, float]:
    pred = str(model.predict([text])[0])
    proba = 0.0

    clf = model.named_steps.get("clf", None)

    if hasattr(model, "predict_proba") and clf is not None and hasattr(clf, "classes_"):
        probs = model.predict_proba([text])[0]
        classes = list(clf.classes_)
        if "spam" in classes:
            idx = classes.index("spam")
            proba = float(probs[idx])

    return pred, proba