#!/usr/bin/env python3
"""
Convert flow_GCN chiplet placement data into chipdiffusion's graph dataset format.

Output layout:
    <output_dir>/
      config.yaml
      graph0.pickle
      output0.pickle
      graph1.pickle
      output1.pickle
      ...

The generated graph pickles contain torch_geometric.data.Data objects with the
minimum fields required by chipdiffusion's graph training pipeline:
    - x:         (V, 2) node sizes [width, height]
    - edge_index:(2, E) directed edges with forward+reverse entries
    - edge_attr: (E, 4) pin offsets; zero-filled to model center-to-center nets
    - is_ports:  (V,) all False
    - is_macros: (V,) all True
    - chip_size: (4,) [xmin, ymin, xmax, ymax] inferred from the GT placement

Additional metadata is stored for future extensions:
    - node_power
    - edge_weight
    - edge_type
    - edge_emib_length
    - edge_emib_max_width
    - edge_emib_bump_width
    - system_id
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch_geometric.data import Data


def _extract_system_id(path: Path) -> int | None:
    stem = path.stem
    if not stem.startswith("system_"):
        return None
    try:
        return int(stem.split("_", 1)[1])
    except ValueError:
        return None


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _paired_files(input_dir: Path, placement_dir: Path) -> List[Tuple[int, Path, Path]]:
    input_files = {
        sys_id: path
        for path in input_dir.glob("system_*.json")
        if (sys_id := _extract_system_id(path)) is not None
    }
    placement_files = {
        sys_id: path
        for path in placement_dir.glob("system_*.json")
        if (sys_id := _extract_system_id(path)) is not None
    }
    common_ids = sorted(input_files.keys() & placement_files.keys())
    return [(sys_id, input_files[sys_id], placement_files[sys_id]) for sys_id in common_ids]


def _chip_bbox(placement_chiplets: List[dict], margin_ratio: float = 0.1) -> torch.Tensor:
    xs_min = [float(ch["x-position"]) for ch in placement_chiplets]
    ys_min = [float(ch["y-position"]) for ch in placement_chiplets]
    xs_max = [float(ch["x-position"]) + float(ch["width"]) for ch in placement_chiplets]
    ys_max = [float(ch["y-position"]) + float(ch["height"]) for ch in placement_chiplets]
    xmin = min(xs_min) if xs_min else 0.0
    ymin = min(ys_min) if ys_min else 0.0
    xmax = max(xs_max) if xs_max else 1.0
    ymax = max(ys_max) if ys_max else 1.0

    # Guard against degenerate boxes; chipdiffusion assumes positive canvas size.
    if math.isclose(xmin, xmax):
        xmax = xmin + 1.0
    if math.isclose(ymin, ymax):
        ymax = ymin + 1.0

    width = xmax - xmin
    height = ymax - ymin
    x_margin = width * margin_ratio
    y_margin = height * margin_ratio
    xmin -= x_margin
    ymin -= y_margin
    xmax += x_margin
    ymax += y_margin
    return torch.tensor([xmin, ymin, xmax, ymax], dtype=torch.float32)


def _build_sample(
    input_data: dict,
    placement_data: dict,
    system_id: int,
    margin_ratio: float,
) -> Tuple[Data, torch.Tensor]:
    chiplets = input_data["chiplets"]
    placement_chiplets = placement_data["chiplets"]

    input_names = [str(ch["name"]) for ch in chiplets]
    placement_by_name = {str(ch["name"]): ch for ch in placement_chiplets}
    missing = [name for name in input_names if name not in placement_by_name]
    if missing:
        raise ValueError(f"Placement missing chiplets {missing} for system_{system_id}")

    node_sizes = torch.tensor(
        [[float(ch["width"]), float(ch["height"])] for ch in chiplets],
        dtype=torch.float32,
    )
    node_power = torch.tensor(
        [float(ch.get("power", 0.0)) for ch in chiplets],
        dtype=torch.float32,
    )

    placement = torch.tensor(
        [
            [
                float(placement_by_name[name]["x-position"]),
                float(placement_by_name[name]["y-position"]),
            ]
            for name in input_names
        ],
        dtype=torch.float32,
    )

    name_to_idx = {name: idx for idx, name in enumerate(input_names)}
    edge_pairs: List[List[int]] = []
    edge_attr: List[List[float]] = []
    edge_weight: List[float] = []
    edge_type: List[float] = []
    edge_emib_length: List[float] = []
    edge_emib_max_width: List[float] = []
    edge_emib_bump_width: List[float] = []

    for conn in input_data.get("connections", []):
        src = name_to_idx[str(conn["node1"])]
        dst = name_to_idx[str(conn["node2"])]
        wire_count = float(conn.get("wireCount", 0.0))
        emib_type = str(conn.get("EMIBType", "interfaceC"))
        emib_type_value = 0.0 if emib_type == "interfaceC" else 1.0
        emib_length = float(conn.get("EMIB_length", 0.0))
        emib_max_width = float(conn.get("EMIB_max_width", 0.0))
        emib_bump_width = float(conn.get("EMIB_bump_width", 0.0))

        for u, v in ((src, dst), (dst, src)):
            edge_pairs.append([u, v])
            # chipdiffusion's HPWL/legality code expects 4-D pin offsets.
            # We do not have pin-level geometry, so use center-to-center nets.
            edge_attr.append([0.0, 0.0, 0.0, 0.0])
            edge_weight.append(wire_count)
            edge_type.append(emib_type_value)
            edge_emib_length.append(emib_length)
            edge_emib_max_width.append(emib_max_width)
            edge_emib_bump_width.append(emib_bump_width)

    if edge_pairs:
        edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()
        edge_attr_tensor = torch.tensor(edge_attr, dtype=torch.float32)
        edge_weight_tensor = torch.tensor(edge_weight, dtype=torch.float32)
        edge_type_tensor = torch.tensor(edge_type, dtype=torch.float32)
        edge_emib_length_tensor = torch.tensor(edge_emib_length, dtype=torch.float32)
        edge_emib_max_width_tensor = torch.tensor(edge_emib_max_width, dtype=torch.float32)
        edge_emib_bump_width_tensor = torch.tensor(edge_emib_bump_width, dtype=torch.float32)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr_tensor = torch.zeros((0, 4), dtype=torch.float32)
        edge_weight_tensor = torch.zeros((0,), dtype=torch.float32)
        edge_type_tensor = torch.zeros((0,), dtype=torch.float32)
        edge_emib_length_tensor = torch.zeros((0,), dtype=torch.float32)
        edge_emib_max_width_tensor = torch.zeros((0,), dtype=torch.float32)
        edge_emib_bump_width_tensor = torch.zeros((0,), dtype=torch.float32)

    data = Data(
        x=node_sizes,
        edge_index=edge_index,
        edge_attr=edge_attr_tensor,
        is_ports=torch.zeros(len(chiplets), dtype=torch.bool),
        is_macros=torch.ones(len(chiplets), dtype=torch.bool),
        chip_size=_chip_bbox(placement_chiplets, margin_ratio=margin_ratio),
        node_power=node_power,
        edge_weight=edge_weight_tensor,
        edge_type=edge_type_tensor,
        edge_emib_length=edge_emib_length_tensor,
        edge_emib_max_width=edge_emib_max_width_tensor,
        edge_emib_bump_width=edge_emib_bump_width_tensor,
        system_id=torch.tensor([system_id], dtype=torch.long),
    )
    return data, placement


def _write_config(output_dir: Path, train_samples: int, val_samples: int) -> None:
    config_text = (
        f"train_samples: {train_samples}\n"
        f"val_samples: {val_samples}\n"
        "scale: 1\n"
    )
    (output_dir / "config.yaml").write_text(config_text, encoding="utf-8")


def convert_dataset(
    input_dir: Path,
    placement_dir: Path,
    output_dir: Path,
    val_ratio: float,
    margin_ratio: float,
) -> None:
    pairs = _paired_files(input_dir, placement_dir)
    if not pairs:
        raise RuntimeError("No paired input/placement samples found.")

    output_dir.mkdir(parents=True, exist_ok=True)
    val_samples = max(1, round(len(pairs) * val_ratio)) if len(pairs) > 1 else 1
    val_samples = min(val_samples, len(pairs))
    train_samples = len(pairs) - val_samples
    _write_config(output_dir, train_samples, val_samples)

    index_map = []
    for new_idx, (system_id, input_path, placement_path) in enumerate(pairs):
        input_data = _load_json(input_path)
        placement_data = _load_json(placement_path)
        cond, placement = _build_sample(input_data, placement_data, system_id, margin_ratio)

        with (output_dir / f"graph{new_idx}.pickle").open("wb") as f:
            pickle.dump(cond, f)
        with (output_dir / f"output{new_idx}.pickle").open("wb") as f:
            pickle.dump(placement.numpy(), f)

        index_map.append(
            {
                "dataset_index": new_idx,
                "system_id": system_id,
                "input_file": str(input_path),
                "placement_file": str(placement_path),
            }
        )

    with (output_dir / "index_map.json").open("w", encoding="utf-8") as f:
        json.dump(index_map, f, indent=2)

    print(f"Converted {len(pairs)} paired samples")
    print(f"Train samples: {train_samples}")
    print(f"Val samples: {val_samples}")
    print(f"Output dir: {output_dir}")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    default_input_dir = repo_root / "flow_GCN" / "gcn_thermal" / "dataset" / "input_test"
    default_placement_dir = repo_root / "flow_GCN" / "gcn_thermal" / "dataset" / "output" / "placement"
    default_output_dir = repo_root / "chipdiffusion" / "datasets" / "graph" / "chiplet_flow_gcn"

    parser = argparse.ArgumentParser(
        description="Convert flow_GCN chiplet data into chipdiffusion graph dataset format."
    )
    parser.add_argument("--input-dir", type=Path, default=default_input_dir)
    parser.add_argument("--placement-dir", type=Path, default=default_placement_dir)
    parser.add_argument("--output-dir", type=Path, default=default_output_dir)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument(
        "--margin-ratio",
        type=float,
        default=0.1,
        help="Expand the inferred chip bounding box by this fraction on each side.",
    )
    args = parser.parse_args()

    if not (0.0 < args.val_ratio <= 1.0):
        raise SystemExit("--val-ratio must be in (0, 1].")

    convert_dataset(
        input_dir=args.input_dir,
        placement_dir=args.placement_dir,
        output_dir=args.output_dir,
        val_ratio=args.val_ratio,
        margin_ratio=args.margin_ratio,
    )


if __name__ == "__main__":
    main()
