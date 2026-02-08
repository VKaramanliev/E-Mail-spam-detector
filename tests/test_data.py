from __future__ import annotations
from pathlib import Path

import pandas as pd
import pytest

from spam_detector.data import load_dataset

def test_load_dataset_ok(tmp_path: Path) -> None:
    p = tmp_path / "d.csv"
    pd.DataFrame(
        {"text": ["hello", "WIN money now"], "label": ["ham", "spam"]}
    ).to_csv(p, index=False)

    ds = load_dataset(p)
    assert len(ds.texts) == 2
    assert ds.labels == ["ham", "spam"]

def test_load_dataset_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_dataset(tmp_path / "nope.csv")

def test_load_dataset_missing_columns(tmp_path: Path) -> None:
    p = tmp_path / "d.csv"
    pd.DataFrame({"body": ["x"], "y": ["ham"]}).to_csv(p, index=False)

    with pytest.raises(ValueError):
        load_dataset(p)

def test_load_dataset_strips_and_lowercase_labels(tmp_path: Path) -> None:
    p = tmp_path / "d.csv"
    pd.DataFrame(
        {"text": [" hello ", None, " WIN "], "label": [" HAM ", "SPAM", " spam "]}
    ).to_csv(p, index=False)

    ds = load_dataset(p)
    assert ds.texts == ["hello", "", "WIN"]
    assert ds.labels == ["ham", "spam", "spam"]

def test_load_dataset_empty_dataset_raises(tmp_path: Path) -> None:
    p = tmp_path / "d.csv"
    pd.DataFrame({"text": [], "label": []}).to_csv(p, index=False)

    with pytest.raises(ValueError):
        load_dataset(p)