from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Paths:
    project_root: Path
    data_raw: Path
    artifacts_dir: Path
    models_dir: Path
    reports_dir: Path

def default_paths() -> Paths:
    root = Path(__file__).resolve().parents[2]
    artifacts = root / "artifacts"
    return Paths(
        project_root=root,
        data_raw=root / "data" / "raw",
        artifacts_dir=artifacts,
        models_dir=artifacts / "models",
        reports_dir=artifacts / "reports"
    )
