from __future__ import annotations

from new_fm.models.flow_model import FlowMatchingModel
from new_fm.models.legacy_flow import LegacyFlowMatchingWrapper


def build_model(node_dim: int, edge_dim: int, config: dict, device: str):
    cfg = dict(config)
    name = cfg.pop("name", "native")
    if name == "legacy_fm":
        cfg.setdefault("device", device)
        return LegacyFlowMatchingWrapper(node_dim=node_dim, edge_dim=edge_dim, **cfg)
    if name in {"native", "fm_gnn_v2"}:
        return FlowMatchingModel(node_dim=node_dim, edge_dim=edge_dim, **cfg)
    raise ValueError(f"unknown model name: {name}")

