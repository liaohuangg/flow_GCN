from __future__ import annotations

import argparse
from statistics import mean

import torch

from new_fm.data.adapter import ChipletLayoutDataset, write_split_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--write-split", default=None)
    args = parser.parse_args()

    if args.write_split:
        write_split_file(args.dataset, args.write_split)

    ds = ChipletLayoutDataset(args.dataset, split=args.split, max_samples=args.max_samples)
    node_counts = []
    edge_counts = []
    pos_values = []
    size_values = []

    for sample in ds:
        node_counts.append(sample.target_pos.shape[0])
        edge_counts.append(sample.edge_index.shape[1])
        pos_values.append(sample.target_pos)
        size_values.append(sample.size)

    print(f"samples: {len(ds)}")
    print(_describe_counts("nodes", node_counts))
    print(_describe_counts("edges", edge_counts))
    if pos_values:
        pos = torch.cat(pos_values, dim=0)
        size = torch.cat(size_values, dim=0)
        print(f"coord_range_normalized: min={pos.min(dim=0).values.tolist()} max={pos.max(dim=0).values.tolist()}")
        print(f"size_range_normalized: min={size.min(dim=0).values.tolist()} max={size.max(dim=0).values.tolist()}")


def _describe_counts(name: str, values: list[int]) -> str:
    if not values:
        return f"{name}: empty"
    return (
        f"{name}: min={min(values)} max={max(values)} "
        f"mean={mean(values):.2f}"
    )


if __name__ == "__main__":
    main()

