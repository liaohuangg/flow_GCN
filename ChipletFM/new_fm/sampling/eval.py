from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from new_fm.data.adapter import ChipletLayoutDataset
from new_fm.data.schema import collate_layout_samples
from new_fm.metrics.layout import (
    evaluate_generated_batch,
    evaluate_original_batch,
    summarize_metric_rows,
)
from new_fm.models.factory import build_model
from new_fm.sampling.guidance_args import add_guidance_args, apply_guidance_overrides
from new_fm.sampling.legalizer_args import add_legalizer_args, legalizer_kwargs
from new_fm.utils.config import load_config
from new_fm.visualization.plot_layout import plot_saved_sample_from_tensors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--output", default=None)
    parser.add_argument("--eval-num", type=int, default=None)
    parser.add_argument("--out-num", type=int, default=None)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--device", default="auto")
    add_guidance_args(parser)
    add_legalizer_args(parser)
    args = parser.parse_args()
    run_eval(args)


def run_eval(args: argparse.Namespace) -> None:
    ckpt = torch.load(args.ckpt, map_location="cpu") if args.ckpt else None
    if ckpt is not None:
        config = apply_guidance_overrides(ckpt["config"], args)
    elif args.config:
        config = load_config(args.config)
    else:
        raise ValueError("--config is required when --ckpt is not provided")

    data_cfg = config["dataset"]
    dataset_path = args.dataset or data_cfg["path"]
    sample_cfg = config.get("sampling", {})
    eval_num = (
        args.eval_num
        or args.num_samples
        or sample_cfg.get("eval_num")
        or sample_cfg.get("num_samples")
        or data_cfg.get(f"{args.split}_max_samples")
    )
    out_num = int(args.out_num if args.out_num is not None else sample_cfg.get("out_num", 0))

    ds = ChipletLayoutDataset(
        dataset_path,
        split=args.split,
        max_samples=eval_num,
        seed=int(data_cfg.get("seed", 123)),
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate_layout_samples)
    device = _resolve_device(args.device)

    model = None
    steps = int(args.steps or sample_cfg.get("steps", 50))
    if ckpt is not None:
        model = build_model(
            node_dim=int(ckpt["node_dim"]),
            edge_dim=int(ckpt["edge_dim"]),
            config=config["model"],
            device=str(device),
        ).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()

    rows = []
    output_path = Path(args.output) if args.output else Path("new_fm/log/eval/metrics.json")
    vis_dir = output_path.parent / "visualizations"
    for idx, batch in enumerate(loader):
        batch = batch.to(device)
        if model is None:
            rows.extend(evaluate_original_batch(batch))
            gen = batch.target_pos
        else:
            with torch.no_grad():
                gen = model.sample(batch, steps=steps)
            if args.legalize_mode != "none" and args.legalize_steps > 0:
                gen = model.legalize(batch, gen, mode=args.legalize_mode, **legalizer_kwargs(args))
            rows.extend(evaluate_generated_batch(batch, gen))
        if idx < out_num:
            plot_saved_sample_from_tensors(
                target_pos=batch.target_pos.cpu(),
                generated_pos=gen.cpu(),
                size=batch.size.cpu(),
                output=vis_dir / f"{batch.sample_id[0]}.png",
            )

    summary = summarize_metric_rows(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


if __name__ == "__main__":
    main()
