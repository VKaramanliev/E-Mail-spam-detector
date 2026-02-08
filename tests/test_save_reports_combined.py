from __future__ import annotations

from pathlib import Path

from spam_detector.plots import save_reports


def test_save_reports_creates_expected_files(tmp_path: Path):
    out_dir = tmp_path / "reports"

    paths = save_reports(
        confusion_matrix_2x2=[[3, 1], [2, 4]],
        precision_by_label={"ham": 0.8, "spam": 0.7},
        out_dir=out_dir,
        prefix="x",
        loss_curve=[0.9, 0.7, 0.5],
        comparison_rows=[("nb", 0.9, 0.88), ("logreg", 0.95, 0.94)],
    )

    assert paths.confusion_matrix_png is not None and paths.confusion_matrix_png.exists()
    assert paths.precision_bar_png is not None and paths.precision_bar_png.exists()
    assert paths.loss_curve_png is not None and paths.loss_curve_png.exists()
    assert paths.comparison_png is not None and paths.comparison_png.exists()

    for p in [paths.confusion_matrix_png, paths.precision_bar_png, paths.loss_curve_png, paths.comparison_png]:
        assert p.stat().st_size > 0