from __future__ import annotations
from dataclasses import dataclass

from pathlib import Path

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from spam_detector.data import load_dataset
from spam_detector.models import load_model

@dataclass(frozen=True)
class EvalResult:
    accuracy: float
    f1_macro: float
    report: str
    confusion: list[list[int]]

def evaluate_model(model: Pipeline, texts: list[str], labels: list[str]) -> EvalResult:
    preds = model.predict(texts)

    acc = float(accuracy_score(labels, preds))
    f1m = float(f1_score(labels, preds, average="macro"))
    report = classification_report(labels, preds, digits=4)

    cm = confusion_matrix(labels, preds, labels=["ham", "spam"])
    confusion = cm.astype(int).tolist()

    return EvalResult(accuracy=acc, f1_macro=f1m, report=report, confusion=confusion)

def evaluate_from_files(model_path: str | Path, data_csv: str | Path) -> EvalResult:
    model = load_model(str(model_path))
    ds = load_dataset(data_csv)
    X_train, X_test, y_train, y_test = train_test_split(
        ds.texts, ds.labels, test_size=0.2, random_state=42, stratify=ds.labels
    )
    return evaluate_model(model, X_test, y_test)


def format_evaluation(result: EvalResult) -> str:
    lines: list[str] = []
    lines.append(f"accuracy={result.accuracy:.4f}")
    lines.append(f"f1_macro={result.f1_macro:.4f}")
    lines.append("")
    lines.append("Confusion matrix (labels: ham, spam):")
    lines.append(str(result.confusion))
    lines.append("")
    lines.append("Classification report:")
    lines.append(result.report)
    return "\n".join(lines)

def save_text_report(result: EvalResult, out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(format_evaluation(result), encoding="utf-8")