from __future__ import annotations

import json
import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import torch
from torch.utils.data import Dataset

from .normalization import normalize_pos, normalize_size
from .schema import Canvas, LayoutSample


@dataclass(frozen=True)
class DatasetEntry:
    kind: str
    sample_id: str
    split: str
    graph_path: str | None = None
    placement_path: str | None = None
    split_name: str | None = None
    split_index: int | None = None


class ChipletLayoutDataset(Dataset):
    def __init__(
        self,
        dataset_path: str | Path,
        split: str = "train",
        max_samples: int | None = None,
        split_file: str | Path | None = None,
        seed: int = 123,
    ) -> None:
        self.dataset_path = Path(dataset_path)
        self.split = split
        self.seed = seed
        self.entries = self._build_entries(split_file)
        if max_samples is not None:
            self.entries = self.entries[:max_samples]
        self._aggregate_cache: Dict[str, Any] = {}

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> LayoutSample:
        entry = self.entries[idx]
        if entry.kind == "aggregate":
            obj = self._load_aggregate()
            item = obj[entry.split_name][entry.split_index]  # type: ignore[index]
            return convert_graph_item(
                graph=item["graph"],
                placement=item["placement"],
                sample_id=entry.sample_id,
            )
        if entry.kind == "pair":
            with open(entry.graph_path, "rb") as f:
                graph = pickle.load(f)
            with open(entry.placement_path, "rb") as f:
                placement = pickle.load(f)
            return convert_graph_item(graph=graph, placement=placement, sample_id=entry.sample_id)
        raise ValueError(f"unknown dataset entry kind: {entry.kind}")

    def _load_aggregate(self) -> Dict[str, Any]:
        key = str(self.dataset_path)
        if key not in self._aggregate_cache:
            with open(self.dataset_path, "rb") as f:
                self._aggregate_cache[key] = pickle.load(f)
        return self._aggregate_cache[key]

    def _build_entries(self, split_file: str | Path | None) -> List[DatasetEntry]:
        if split_file is not None and Path(split_file).exists():
            return [
                DatasetEntry(**e)
                for e in _load_split_file(split_file)
                if e["split"] == self.split
            ]
        if self.dataset_path.is_file():
            return _entries_from_aggregate(self.dataset_path, self.split)
        if self.dataset_path.is_dir():
            return _entries_from_pair_dir(self.dataset_path, self.split, self.seed)
        raise FileNotFoundError(self.dataset_path)


def convert_graph_item(graph: Any, placement: Any, sample_id: str) -> LayoutSample:
    x = _as_float_tensor(graph["x"])
    target_pos_raw = _as_float_tensor(placement)
    edge_index = torch.as_tensor(graph["edge_index"], dtype=torch.long)
    edge_attr_raw = _as_float_tensor(graph["edge_attr"])

    size_raw = x[:, :2].abs()
    origin, canvas_size = _canvas_from_graph(graph, target_pos_raw, size_raw)
    target_center_raw = target_pos_raw[:, :2] + size_raw / 2.0
    target_pos = normalize_pos(target_center_raw, origin, canvas_size)
    size = normalize_size(size_raw, canvas_size)
    edge_attr = _normalize_edge_attr(edge_attr_raw, canvas_size)

    node_power = _optional_vector(graph, "node_power", x.shape[0])
    node_power = node_power / node_power.abs().max().clamp_min(1.0)
    is_ports = _optional_bool(graph, "is_ports", x.shape[0])
    is_macros = _optional_bool(graph, "is_macros", x.shape[0])
    movable_mask = is_macros.clone()
    if not bool(movable_mask.any()):
        movable_mask = torch.ones(x.shape[0], dtype=torch.bool)

    node_feat = torch.cat(
        [
            size,
            node_power.view(-1, 1),
            is_ports.float().view(-1, 1),
            is_macros.float().view(-1, 1),
        ],
        dim=1,
    )

    return LayoutSample(
        sample_id=sample_id,
        target_pos=target_pos.float(),
        size=size.float(),
        node_feat=node_feat.float(),
        edge_index=edge_index.long(),
        edge_attr=edge_attr.float(),
        movable_mask=movable_mask.bool(),
        canvas=Canvas(origin=origin.float(), size=canvas_size.float()),
    )


def write_split_file(dataset_path: str | Path, output_path: str | Path, seed: int = 123) -> None:
    dataset_path = Path(dataset_path)
    if dataset_path.is_file():
        entries = []
        for split in ("train", "val", "test"):
            entries.extend(_entries_from_aggregate(dataset_path, split))
    else:
        entries = []
        for split in ("train", "val", "test"):
            entries.extend(_entries_from_pair_dir(dataset_path, split, seed))
    serializable = [
        {
            "split": _split_from_entry(e),
            "kind": e.kind,
            "sample_id": e.sample_id,
            "graph_path": e.graph_path,
            "placement_path": e.placement_path,
            "split_name": e.split_name,
            "split_index": e.split_index,
        }
        for e in entries
    ]
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(serializable, indent=2), encoding="utf-8")


def _entries_from_aggregate(path: Path, split: str) -> List[DatasetEntry]:
    with open(path, "rb") as f:
        obj = pickle.load(f)
    train_len = len(obj.get("train", []))
    val_len = len(obj.get("val", []))
    if split == "train":
        return [
            DatasetEntry("aggregate", f"train_{i}", "train", split_name="train", split_index=i)
            for i in range(train_len)
        ]
    val_cut = val_len // 2
    if split == "val":
        indices = range(0, val_cut)
    elif split == "test":
        indices = range(val_cut, val_len)
    else:
        raise ValueError(f"unknown split: {split}")
    return [
        DatasetEntry("aggregate", f"{split}_{i}", split, split_name="val", split_index=i)
        for i in indices
    ]


def _entries_from_pair_dir(path: Path, split: str, seed: int) -> List[DatasetEntry]:
    graph_files = sorted(path.glob("graph*.pickle"), key=lambda p: _numeric_suffix(p.stem))
    pairs: List[DatasetEntry] = []
    for graph_path in graph_files:
        idx = _numeric_suffix(graph_path.stem)
        placement_path = path / f"output{idx}.pickle"
        if placement_path.exists():
            pairs.append(
                DatasetEntry(
                    "pair",
                    f"pair_{idx}",
                    "unknown",
                    graph_path=str(graph_path),
                    placement_path=str(placement_path),
                )
            )
    rng = random.Random(seed)
    pairs = pairs[:]
    rng.shuffle(pairs)
    n = len(pairs)
    train_end = int(n * 0.9)
    val_end = int(n * 0.95)
    split_map = {"train": pairs[:train_end], "val": pairs[train_end:val_end], "test": pairs[val_end:]}
    return [
        DatasetEntry(
            kind=e.kind,
            sample_id=e.sample_id,
            split=split,
            graph_path=e.graph_path,
            placement_path=e.placement_path,
            split_name=e.split_name,
            split_index=e.split_index,
        )
        for e in split_map[split]
    ]


def _load_split_file(path: str | Path) -> List[Dict[str, Any]]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _split_from_entry(entry: DatasetEntry) -> str:
    return entry.split


def _canvas_from_graph(
    graph: Any, lower_left_pos: torch.Tensor, size: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    if "chip_size" in graph:
        chip = _as_float_tensor(graph["chip_size"]).flatten()
        if chip.numel() >= 4:
            origin = chip[:2]
            canvas_size = chip[2:4].abs().clamp_min(1.0)
            return origin, canvas_size
        if chip.numel() >= 2:
            return torch.zeros(2), chip[:2].abs().clamp_min(1.0)
    max_xy = (lower_left_pos[:, :2] + size[:, :2]).max(dim=0).values
    min_xy = lower_left_pos[:, :2].min(dim=0).values
    return min_xy, (max_xy - min_xy).clamp_min(1.0)


def _optional_vector(graph: Any, key: str, n: int) -> torch.Tensor:
    if key in graph:
        return _as_float_tensor(graph[key]).flatten()[:n]
    return torch.zeros(n, dtype=torch.float32)


def _optional_bool(graph: Any, key: str, n: int) -> torch.Tensor:
    if key in graph:
        return torch.as_tensor(graph[key], dtype=torch.bool).flatten()[:n]
    return torch.zeros(n, dtype=torch.bool)


def _as_float_tensor(value: Any) -> torch.Tensor:
    return torch.as_tensor(value, dtype=torch.float32)


def _normalize_edge_attr(edge_attr: torch.Tensor, canvas_size: torch.Tensor) -> torch.Tensor:
    if edge_attr.numel() == 0:
        return edge_attr.float()
    out = edge_attr.float().clone()
    diag = torch.linalg.norm(canvas_size.float()).clamp_min(1.0)
    if out.shape[1] >= 1:
        out[:, 0] = out[:, 0] / out[:, 0].abs().max().clamp_min(1.0)
    if out.shape[1] >= 2:
        out[:, 1] = out[:, 1] / diag
    if out.shape[1] >= 3:
        out[:, 2] = out[:, 2] / canvas_size.max().clamp_min(1.0)
    if out.shape[1] >= 4:
        out[:, 3] = out[:, 3] / canvas_size.max().clamp_min(1.0)
    return out


def _numeric_suffix(text: str) -> int:
    digits = "".join(ch for ch in text if ch.isdigit())
    return int(digits) if digits else -1
