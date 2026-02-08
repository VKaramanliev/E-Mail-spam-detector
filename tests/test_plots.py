from __future__ import annotations

from pathlib import Path

import pytest

from spam_detector.plots import (
    save_confusion_matrix_png,
    save_loss_curve_png,
    save_model_comparison_png,
    save_precision_bar_png,
    to_matrix,
)


def test_to_matrix_valid() -> None:
    arr = to_matrix([[1, 2], [3, 4]])
    assert arr.shape == (2, 2)


def test_to_matrix_invalid_shape() -> None:
    with pytest.raises(ValueError):
        to_matrix([[1, 2, 3], [4, 5, 6]])


def test_save_confusion_matrix_png_create_file(tmp_path: Path) -> None:
    out = save_confusion_matrix_png([[5, 1], [2, 7]], out_dir=tmp_path, filename="cm.png")
    assert out.exists()
    assert out.suffix == ".png"
    assert out.stat().st_size > 0


def test_save_precision_bar_png_creates_file(tmp_path: Path) -> None:
    out = save_precision_bar_png(
        {"ham": 0.9, "spam": 0.8},
        out_dir=tmp_path,
        filename="prec.png",
        labels_order=("ham", "spam"),
    )

    assert out.exists()
    assert out.suffix == ".png"
    assert out.stat().st_size > 0


def test_save_loss_curve_png_creates_file(tmp_path: Path) -> None:
    out = save_loss_curve_png([0.9, 0.7, 0.6, 0.55], out_dir=tmp_path, filename="loss.png")
    assert out.exists()
    assert out.suffix == ".png"
    assert out.stat().st_size > 0


def test_save_model_comparison_png_creates_file(tmp_path: Path) -> None:
    rows = [
        ("nb", 0.80, 0.78),
        ("logreg", 0.85, 0.84),
        ("sgd", 0.83, 0.82),
    ]
    out = save_model_comparison_png(rows, out_dir=tmp_path, filename="cmp.png")
    assert out.exists()
    assert out.suffix == ".png"
    assert out.stat().st_size > 0