#!/usr/bin/env python3
"""
Convert flow_GCN chiplet placement data into a single consolidated dataset.pkl
that chipdiffusion can load directly from datasets/graph/<task>.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
from pathlib import Path

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


def _paired_files(input_dir: Path, placement_dir: Path) -> list[tuple[int, Path, Path]]:
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


def _chip_bbox(placement_chiplets: list[dict], margin_ratio: float = 0.1) -> torch.Tensor:
    xs_min = [float(ch["x-position"]) for ch in placement_chiplets]
    ys_min = [float(ch["y-position"]) for ch in placement_chiplets]
    xs_max = [float(ch["x-position"]) + float(ch["width"]) for ch in placement_chiplets]
    ys_max = [float(ch["y-position"]) + float(ch["height"]) for ch in placement_chiplets]
    xmin = min(xs_min) if xs_min else 0.0
    ymin = min(ys_min) if ys_min else 0.0
    xmax = max(xs_max) if xs_max else 1.0
    ymax = max(ys_max) if ys_max else 1.0

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


def _edge_features(
    wire_count: float,
    emib_type_value: float,
    emib_length: float,
    emib_bump_width: float,
) -> list[float]:
    # The first four edge_attr dimensions are reserved for source/dest pin offsets.
    # Chiplet inputs do not provide pin locations, so append normalized connection
    # features after the pin-offset block for the GNN edge encoder.
    return [
        0.0,
        0.0,
        0.0,
        0.0,
        wire_count / 1024.0,
        emib_length / 25.6,
        emib_type_value,
        emib_bump_width / 0.5,
    ]


def _build_example(
    input_data: dict,
    placement_data: dict,
    system_id: int,
    margin_ratio: float,
) -> dict:
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
    forward_edges = []
    reverse_edges = []

    for conn in input_data.get("connections", []):
        src = name_to_idx[str(conn["node1"])]
        dst = name_to_idx[str(conn["node2"])]
        wire_count = float(conn.get("wireCount", 0.0))
        emib_type = str(conn.get("EMIBType", "interfaceC"))
        emib_type_value = 0.0 if emib_type == "interfaceC" else 1.0
        emib_length = float(conn.get("EMIB_length", 0.0))
        emib_max_width = float(conn.get("EMIB_max_width", 0.0))
        emib_bump_width = float(conn.get("EMIB_bump_width", 0.0))

        attr = {
            "edge_attr": _edge_features(
                wire_count,
                emib_type_value,
                emib_length,
                emib_bump_width,
            ),
            "edge_weight": wire_count,
            "edge_type": emib_type_value,
            "edge_emib_length": emib_length,
            "edge_emib_max_width": emib_max_width,
            "edge_emib_bump_width": emib_bump_width,
        }
        forward_edges.append({"edge_pair": [src, dst], **attr})
        reverse_edges.append({"edge_pair": [dst, src], **attr})

    edges = forward_edges + reverse_edges
    edge_pairs = [edge["edge_pair"] for edge in edges]
    edge_attr = [edge["edge_attr"] for edge in edges]
    edge_weight = [edge["edge_weight"] for edge in edges]
    edge_type = [edge["edge_type"] for edge in edges]
    edge_emib_length = [edge["edge_emib_length"] for edge in edges]
    edge_emib_max_width = [edge["edge_emib_max_width"] for edge in edges]
    edge_emib_bump_width = [edge["edge_emib_bump_width"] for edge in edges]

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

    graph = Data(
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
    return {
        "file_idx": system_id,
        "graph": graph,
        "placement": placement,
    }


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

    examples = [
        _build_example(_load_json(inp), _load_json(plc), system_id, margin_ratio)
        for system_id, inp, plc in pairs
    ]
    dataset = {
        "train": examples[:train_samples],
        "val": examples[train_samples:],
        "meta": {
            "total_samples": len(examples),
            "train_samples": train_samples,
            "val_samples": val_samples,
            "input_dir": str(input_dir),
            "placement_dir": str(placement_dir),
        },
    }

    with (output_dir / "dataset.pkl").open("wb") as f:
        pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)

    _write_config(output_dir, train_samples, val_samples)
    with (output_dir / "index_map.json").open("w", encoding="utf-8") as f:
        json.dump(
            [
                {
                    "dataset_index": i,
                    "system_id": system_id,
                    "split": "train" if i < train_samples else "val",
                    "input_file": str(inp),
                    "placement_file": str(plc),
                }
                for i, (system_id, inp, plc) in enumerate(pairs)
            ],
            f,
            indent=2,
        )

    print(f"Converted {len(examples)} paired samples")
    print(f"Train samples: {train_samples}")
    print(f"Val samples: {val_samples}")
    print(f"Output dir: {output_dir}")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Convert flow_GCN dataset into a single chipdiffusion dataset.pkl.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=repo_root.parent / "Dataset" / "dataset" / "input_test",
    )
    parser.add_argument(
        "--placement-dir",
        type=Path,
        default=repo_root.parent / "Dataset" / "dataset" / "output" / "placement",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "datasets" / "graph" / "v2",
    )
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
        args.input_dir,
        args.placement_dir,
        args.output_dir,
        args.val_ratio,
        args.margin_ratio,
    )


if __name__ == "__main__":
    main()
