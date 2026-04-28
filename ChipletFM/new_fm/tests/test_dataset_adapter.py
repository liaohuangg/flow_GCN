from __future__ import annotations

from pathlib import Path

from new_fm.data.adapter import ChipletLayoutDataset


def test_dataset_adapter_outputs_required_fields() -> None:
    path = Path("datasets/graph/v3/dataset.pkl")
    ds = ChipletLayoutDataset(path, split="train", max_samples=1)
    sample = ds[0]
    assert sample.target_pos.ndim == 2 and sample.target_pos.shape[1] == 2
    assert sample.size.shape == sample.target_pos.shape
    assert sample.node_feat.shape[0] == sample.target_pos.shape[0]
    assert sample.edge_index.shape[0] == 2
    assert sample.edge_attr.shape[0] == sample.edge_index.shape[1]
    assert sample.movable_mask.shape[0] == sample.target_pos.shape[0]
    assert sample.canvas.origin.shape == (2,)
    assert sample.canvas.size.shape == (2,)

