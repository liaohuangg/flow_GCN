from __future__ import annotations

from typing import Dict, List

import torch

from new_fm.data.normalization import denormalize_pos
from new_fm.data.schema import LayoutBatch
from new_fm.metrics.hpwl import graph_hpwl
from new_fm.metrics.legality import (
    boundary_violation_area,
    legality_score,
    overlap_area,
)


def evaluate_generated_batch(batch: LayoutBatch, gen_pos: torch.Tensor) -> List[Dict[str, float | str]]:
    rows: List[Dict[str, float | str]] = []
    for i, sample_id in enumerate(batch.sample_id):
        sl = slice(int(batch.ptr[i]), int(batch.ptr[i + 1]))
        edge_index = _sample_edge_index(batch, i)
        origin = batch.canvas_origin[i]
        canvas_size = batch.canvas_size[i]

        gen_abs = denormalize_pos(gen_pos[sl], origin, canvas_size)
        orig_abs = denormalize_pos(batch.target_pos[sl], origin, canvas_size)
        size_abs = batch.size[sl] * canvas_size.view(1, 2)

        gen_hpwl = graph_hpwl(gen_abs, edge_index)
        original_hpwl = graph_hpwl(orig_abs, edge_index)
        overlap = overlap_area(gen_abs, size_abs)
        boundary = boundary_violation_area(gen_abs, size_abs, origin, canvas_size)
        score = legality_score(gen_abs, size_abs, origin, canvas_size)
        total_area = (size_abs[:, 0] * size_abs[:, 1]).sum().clamp_min(1e-6)

        rows.append(
            {
                "sample_id": sample_id,
                "gen_hpwl": float(gen_hpwl.cpu()),
                "original_hpwl": float(original_hpwl.cpu()),
                "hpwl_ratio": float((gen_hpwl / original_hpwl.clamp_min(1e-6)).cpu()),
                "legality_score": float(score.cpu()),
                "overlap_area": float(overlap.cpu()),
                "boundary_violation_area": float(boundary.cpu()),
                "overlap_ratio": float((overlap / total_area).cpu()),
                "boundary_violation_ratio": float((boundary / total_area).cpu()),
            }
        )
    return rows


def evaluate_original_batch(batch: LayoutBatch) -> List[Dict[str, float | str]]:
    return evaluate_generated_batch(batch, batch.target_pos)


def summarize_metric_rows(rows: List[Dict[str, float | str]]) -> Dict[str, object]:
    if not rows:
        return {"num_samples": 0, "samples": []}
    metric_keys = [
        "gen_hpwl",
        "original_hpwl",
        "hpwl_ratio",
        "legality_score",
        "overlap_area",
        "boundary_violation_area",
        "overlap_ratio",
        "boundary_violation_ratio",
    ]
    summary: Dict[str, object] = {"num_samples": len(rows), "samples": rows}
    for key in metric_keys:
        summary[key] = sum(float(row[key]) for row in rows) / len(rows)
    return summary


def _sample_edge_index(batch: LayoutBatch, sample_idx: int) -> torch.Tensor:
    start = int(batch.ptr[sample_idx])
    end = int(batch.ptr[sample_idx + 1])
    mask = (batch.edge_index[0] >= start) & (batch.edge_index[0] < end)
    return batch.edge_index[:, mask] - start

