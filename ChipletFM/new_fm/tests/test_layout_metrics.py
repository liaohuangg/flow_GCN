from __future__ import annotations

from new_fm.metrics.layout import summarize_metric_rows


def test_layout_metric_summary_averages_core_fields() -> None:
    rows = [
        {
            "sample_id": "a",
            "gen_hpwl": 2.0,
            "original_hpwl": 1.0,
            "hpwl_ratio": 2.0,
            "legality_score": 0.5,
            "overlap_area": 1.0,
            "boundary_violation_area": 3.0,
            "overlap_ratio": 0.1,
            "boundary_violation_ratio": 0.3,
        },
        {
            "sample_id": "b",
            "gen_hpwl": 4.0,
            "original_hpwl": 2.0,
            "hpwl_ratio": 2.0,
            "legality_score": 1.0,
            "overlap_area": 0.0,
            "boundary_violation_area": 1.0,
            "overlap_ratio": 0.0,
            "boundary_violation_ratio": 0.1,
        },
    ]
    summary = summarize_metric_rows(rows)
    assert summary["num_samples"] == 2
    assert summary["gen_hpwl"] == 3.0
    assert summary["legality_score"] == 0.75
    assert summary["boundary_violation_ratio"] == 0.2

