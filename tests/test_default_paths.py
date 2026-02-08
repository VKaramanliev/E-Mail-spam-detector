from __future__ import annotations

from spam_detector.config import default_paths


def test_default_paths_structure():
    p = default_paths()

    assert p.artifacts_dir.name == "artifacts"
    assert p.models_dir.as_posix().endswith("artifacts/models")
    assert p.reports_dir.as_posix().endswith("artifacts/reports")
    assert p.data_raw.as_posix().endswith("data/raw")