from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPORTS_DIR = Path("artifacts") / "reports"
DEFAULT_LABELS: tuple[str, str] = ("ham", "spam")


@dataclass(frozen=True)
class PlotPaths:
    confusion_matrix_png: Path | None = None
    precision_bar_png: Path | None = None
    loss_curve_png: Path | None = None
    comparison_png: Path | None = None


def ensure_reports_dir(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)


def to_matrix(cm: Sequence[Sequence[float]]) -> np.ndarray:
    arr = np.asarray(cm, dtype=float)
    if arr.shape != (2, 2):
        raise ValueError(f"Confusion matrix must be 2x2, got shape {arr.shape}")
    return arr


def save_confusion_matrix_png(
    confusion_matrix_2x2: Sequence[Sequence[float]],
    *,
    out_dir: str | Path = REPORTS_DIR,
    filename: str = "confusion_matrix.png",
    labels: tuple[str, str] = DEFAULT_LABELS,
    title: str = "Confusion matrix",
    dpi: int = 150,
) -> Path:
    out_dir = Path(out_dir)
    ensure_reports_dir(out_dir)
    out_path = out_dir / filename

    cm = to_matrix(confusion_matrix_2x2)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm)

    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks([0, 1], labels=list(labels))
    ax.set_yticks([0, 1], labels=list(labels))

    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{int(cm[i, j])}", ha="center", va="center")

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def save_precision_bar_png(
    precision_by_label: Mapping[str, float],
    *,
    out_dir: str | Path = REPORTS_DIR,
    filename: str = "precision_bar.png",
    labels_order: Sequence[str] = DEFAULT_LABELS,
    title: str = "Precision by class",
    dpi: int = 150,
) -> Path:
    out_dir = Path(out_dir)
    ensure_reports_dir(out_dir)
    out_path = out_dir / filename

    labels = list(labels_order)
    values = [float(precision_by_label.get(lbl, 0)) for lbl in labels]

    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(labels))

    ax.bar(x, values)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("precision")
    ax.set_title(title)

    for i, v in enumerate(values):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", va="bottom")

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def save_loss_curve_png(
    losses: Sequence[float],
    *,
    out_dir: str | Path = REPORTS_DIR,
    filename: str = "loss_curve.png",
    title: str = "Validation loss per epoch",
    dpi: int = 150,
) -> Path:
    out_dir = Path(out_dir)
    ensure_reports_dir(out_dir)
    out_path = out_dir / filename

    fig, ax = plt.subplots(figsize=(6, 4))
    xs = np.arange(1, len(losses) + 1)
    ax.plot(xs, list(losses))
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def save_model_comparison_png(
    rows: Sequence[tuple[str, float, float]],
    *,
    out_dir: str | Path = REPORTS_DIR,
    filename: str = "model_comparison.png",
    title: str = "Test metrics comparison (accuracy & f1_macro)",
    dpi: int = 150,
) -> Path:
    """
    rows: [(model_name, test_accuracy, test_f1_macro), ...]
    """
    out_dir = Path(out_dir)
    ensure_reports_dir(out_dir)
    out_path = out_dir / filename

    names = [r[0] for r in rows]
    accs = [float(r[1]) for r in rows]
    f1s = [float(r[2]) for r in rows]

    x = np.arange(len(names))
    w = 0.35

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - w / 2, accs, width=w, label="accuracy")
    ax.bar(x + w / 2, f1s, width=w, label="f1_macro")
    ax.set_xticks(x, names, rotation=0)
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def save_reports(
    *,
    confusion_matrix_2x2: Sequence[Sequence[float]],
    precision_by_label: Mapping[str, float],
    out_dir: str | Path = REPORTS_DIR,
    prefix: str = "eval",
    labels: tuple[str, str] = DEFAULT_LABELS,
    loss_curve: Sequence[float] | None = None,
    comparison_rows: Sequence[tuple[str, float, float]] | None = None,
) -> PlotPaths:
    out_dir = Path(out_dir)
    ensure_reports_dir(out_dir)

    cm_path = save_confusion_matrix_png(
        confusion_matrix_2x2,
        out_dir=out_dir,
        filename=f"{prefix}_confusion_matrix.png",
        labels=labels,
        title="Confusion matrix",
    )

    prec_path = save_precision_bar_png(
        precision_by_label,
        out_dir=out_dir,
        filename=f"{prefix}_precision_bar.png",
        labels_order=list(labels),
        title="Precision by class",
    )

    loss_path: Path | None = None
    if loss_curve is not None and len(list(loss_curve)) > 0:
        loss_path = save_loss_curve_png(
            list(loss_curve),
            out_dir=out_dir,
            filename=f"{prefix}_loss_curve.png",
            title="Validation loss per epoch",
        )

    comp_path: Path | None = None
    if comparison_rows is not None and len(list(comparison_rows)) > 0:
        comp_path = save_model_comparison_png(
            list(comparison_rows),
            out_dir=out_dir,
            filename=f"{prefix}_model_comparison.png",
        )

    return PlotPaths(
        confusion_matrix_png=cm_path,
        precision_bar_png=prec_path,
        loss_curve_png=loss_path,
        comparison_png=comp_path,
    )