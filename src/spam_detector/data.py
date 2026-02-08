from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import pandas as pd

@dataclass(frozen=True)
class Dataset:
    texts: list[str]
    labels: list[str]

REQUIRED_COLUMNS: tuple[str, str] = ("text", "label")
ALLOWED_LABELS: set[str] = {"ham", "spam"}

def load_dataset(csv_path: Path) -> Dataset:
    if not csv_path.exists():
        raise FileNotFoundError("Dataset not found")
    
    df = pd.read_csv(csv_path)
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in {csv_path}")
        
    texts = (
        df["text"]
        .fillna("")
        .astype(str)
        .map(lambda s: s.strip())
        .tolist()
    )

    labels = (
        df["label"]
        .fillna("")
        .astype(str)
        .map(lambda s: s.strip().lower())
        .tolist()
    )

    if not texts or not labels:
        raise ValueError("Empty dataset")
        
    if len(texts) != len(labels):
        raise ValueError("Mismatched length")
    
    bad = sorted({lbl for lbl in labels if lbl not in ALLOWED_LABELS})
    if bad:
        raise ValueError(f"Invalid labels found: {bad}. Allowed: {sorted(ALLOWED_LABELS)}")
        
    return Dataset(texts=texts, labels=labels)