from __future__ import annotations

import argparse
import json
import pickle
import random
from pathlib import Path
from typing import Any, Dict, List

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        default=r"D:\WORK\flow_GCN\Dataset\dataset\input_test",
    )
    parser.add_argument(
        "--placement-dir",
        default=r"D:\WORK\flow_GCN\Dataset\dataset\output\placement",
    )
    parser.add_argument("--output-dir", default="new_fm/dataset")
    parser.add_argument("--max-samples", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--train-ratio", type=float, default=0.9)
    args = parser.parse_args()

    build_dataset(
        input_dir=Path(args.input_dir),
        placement_dir=Path(args.placement_dir),
        output_dir=Path(args.output_dir),
        max_samples=args.max_samples,
        seed=args.seed,
        train_ratio=args.train_ratio,
    )


def build_dataset(
    input_dir: Path,
    placement_dir: Path,
    output_dir: Path,
    max_samples: int,
    seed: int,
    train_ratio: float,
) -> None:
    ids = sorted(
        set(_system_id(p) for p in input_dir.glob("system_*.json"))
        & set(_system_id(p) for p in placement_dir.glob("system_*.json"))
    )
    ids = ids[:max_samples]
    rng = random.Random(seed)
    shuffled = ids[:]
    rng.shuffle(shuffled)

    split_at = int(len(shuffled) * train_ratio)
    train_ids = set(shuffled[:split_at])

    train: List[Dict[str, Any]] = []
    val: List[Dict[str, Any]] = []
    index_map: Dict[str, Dict[str, Any]] = {}

    for sid in ids:
        input_path = input_dir / f"system_{sid}.json"
        placement_path = placement_dir / f"system_{sid}.json"
        item = convert_system(input_path, placement_path, sid)
        target = train if sid in train_ids else val
        target.append(item)
        index_map[str(sid)] = {
            "input": str(input_path),
            "placement": str(placement_path),
            "split": "train" if sid in train_ids else "val",
        }

    dataset = {
        "train": train,
        "val": val,
        "meta": {
            "total_samples": len(ids),
            "train_samples": len(train),
            "val_samples": len(val),
            "input_dir": str(input_dir),
            "placement_dir": str(placement_dir),
            "coordinate_format": "lower_left",
            "id_rule": "system_<id>.json matched across input and placement directories",
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "dataset.pkl", "wb") as f:
        pickle.dump(dataset, f)
    (output_dir / "index_map.json").write_text(
        json.dumps(index_map, indent=2), encoding="utf-8"
    )
    (output_dir / "config.yaml").write_text(
        "\n".join(
            [
                f"total_samples: {len(ids)}",
                f"train_samples: {len(train)}",
                f"val_samples: {len(val)}",
                "coordinate_format: lower_left",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {output_dir / 'dataset.pkl'}")
    print(f"total={len(ids)} train={len(train)} val={len(val)}")


def convert_system(input_path: Path, placement_path: Path, system_id: int) -> Dict[str, Any]:
    input_obj = json.loads(input_path.read_text(encoding="utf-8"))
    placement_obj = json.loads(placement_path.read_text(encoding="utf-8"))

    placement_chiplets = placement_obj["chiplets"]
    names = [chip["name"] for chip in placement_chiplets]
    name_to_idx = {name: i for i, name in enumerate(names)}

    size = torch.tensor(
        [[float(chip["width"]), float(chip["height"])] for chip in placement_chiplets],
        dtype=torch.float32,
    )
    placement = torch.tensor(
        [
            [float(chip["x-position"]), float(chip["y-position"])]
            for chip in placement_chiplets
        ],
        dtype=torch.float32,
    )
    node_power = torch.tensor(
        [float(chip.get("power", 0.0)) for chip in placement_chiplets],
        dtype=torch.float32,
    )
    edge_index_parts = []
    edge_attr_parts = []
    edge_weight_parts = []
    edge_type_parts = []

    for conn in input_obj.get("connections", []):
        if conn["node1"] not in name_to_idx or conn["node2"] not in name_to_idx:
            continue
        u = name_to_idx[conn["node1"]]
        v = name_to_idx[conn["node2"]]
        attr = [
            float(conn.get("wireCount", 0.0)),
            float(conn.get("EMIB_length", 0.0)),
            float(conn.get("EMIB_max_width", 0.0)),
            float(conn.get("EMIB_bump_width", 0.0)),
        ]
        edge_index_parts.extend([[u, v], [v, u]])
        edge_attr_parts.extend([attr, attr])
        edge_weight_parts.extend([attr[0], attr[0]])
        edge_type = _edge_type(conn.get("EMIBType", ""))
        edge_type_parts.extend([edge_type, edge_type])

    if edge_index_parts:
        edge_index = torch.tensor(edge_index_parts, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr_parts, dtype=torch.float32)
        edge_weight = torch.tensor(edge_weight_parts, dtype=torch.float32)
        edge_type = torch.tensor(edge_type_parts, dtype=torch.float32)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 4), dtype=torch.float32)
        edge_weight = torch.empty((0,), dtype=torch.float32)
        edge_type = torch.empty((0,), dtype=torch.float32)

    lower_left = placement
    upper_right = placement + size
    canvas_size = upper_right.max(dim=0).values - lower_left.min(dim=0).values
    canvas_origin = lower_left.min(dim=0).values

    graph = {
        "x": size,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "chip_size": torch.cat([canvas_origin, canvas_size.clamp_min(1.0)]).float(),
        "node_power": node_power,
        "edge_weight": edge_weight,
        "edge_type": edge_type,
        "edge_emib_length": edge_attr[:, 1] if edge_attr.numel() else torch.empty((0,)),
        "edge_emib_max_width": edge_attr[:, 2] if edge_attr.numel() else torch.empty((0,)),
        "edge_emib_bump_width": edge_attr[:, 3] if edge_attr.numel() else torch.empty((0,)),
        "is_ports": torch.zeros((len(names),), dtype=torch.bool),
        "is_macros": torch.ones((len(names),), dtype=torch.bool),
        "system_id": torch.tensor([system_id], dtype=torch.long),
    }
    return {"file_idx": system_id, "graph": graph, "placement": placement}


def _system_id(path: Path) -> int:
    return int(path.stem.split("_")[-1])


def _edge_type(value: str) -> float:
    if not value:
        return 0.0
    suffix = value[-1].upper()
    if "A" <= suffix <= "Z":
        return float(ord(suffix) - ord("A") + 1)
    return 0.0


if __name__ == "__main__":
    main()

