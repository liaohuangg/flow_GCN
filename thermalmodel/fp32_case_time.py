"""fp32_case_time.py

Re-time the FP32 checkpoint including:
  1) per-case end-to-end time (includes DataLoader + H2D copy) and
  2) per-case model forward time (excludes DataLoader/H2D).

Then update ONLY the [fp32] block in:
  /root/placement/flow_GCN/thermalmodel/test_result/per_case_time.log

Usage (chipdiffusion env):
  python fp32_case_time.py --n_cases 1000 --eval_bs 8 --device cuda
"""

import argparse
import os
import re
import sys
import time
from typing import Dict, List, Tuple

import torch

# Allow running from within thermalmodel/
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def _list_first_n_cases(thermal_map_root: str, n: int) -> List[Tuple[int, int]]:
    power_dir = os.path.join(thermal_map_root, "powercsv")
    pat = re.compile(r"system_power_(\d+)_(\d+)\.csv$")
    cases: List[Tuple[int, int]] = []
    for fn in os.listdir(power_dir):
        m = pat.match(fn)
        if not m:
            continue
        cases.append((int(m.group(1)), int(m.group(2))))
    cases.sort()
    return cases[:n]


def _mk_dataset(*, thermal_map_root: str, cases):
    from thermalmodel.dataLoader import ThermalDataset

    return ThermalDataset(
        thermal_map_rel=os.path.relpath(thermal_map_root, start=_PROJECT_ROOT),
        hotspot_cfg_rel="Dataset/dataset/output/thermal/hotspot_config",
        power_grid_size=128,
        temp_grid_size=64,
        cases=cases,
    )


@torch.no_grad()
def time_fp32_ckpt(
    ckpt_path: str,
    *,
    n_cases: int,
    eval_bs: int,
    seed: int,
    device: str,
    thermal_map_root: str,
    num_workers: int,
    pin_memory: bool,
) -> Dict[str, float]:
    from thermalmodel.guidance_model import ThermalGuidanceNet

    torch.manual_seed(seed)

    dev = torch.device("cuda" if device == "cuda" and torch.cuda.is_available() else "cpu")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = ThermalGuidanceNet(base=int(ckpt.get("base", 32))).to(dev)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    cases = _list_first_n_cases(thermal_map_root, n_cases)
    ds_t0 = time.perf_counter()
    dataset = _mk_dataset(thermal_map_root=thermal_map_root, cases=cases)
    ds_build_s = time.perf_counter() - ds_t0

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=eval_bs,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    if dev.type == "cuda":
        torch.cuda.synchronize()

    t_wall0 = time.perf_counter()
    t_dl_s = 0.0
    t_fwd_s = 0.0
    n_seen = 0

    for batch in loader:
        dl0 = time.perf_counter()

        power = batch["power"].to(dev, non_blocking=False)
        layout = batch["layout"].to(dev, non_blocking=False)
        totalp = batch.get("total_power")
        totalp = totalp.to(dev, non_blocking=False) if totalp is not None else None

        if dev.type == "cuda":
            torch.cuda.synchronize()

        dl1 = time.perf_counter()
        t_dl_s += (dl1 - dl0)

        if dev.type == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            _ = model(power, layout, totalp)
            end.record()
            torch.cuda.synchronize()
            t_fwd_s += start.elapsed_time(end) / 1e3
        else:
            f0 = time.perf_counter()
            _ = model(power, layout, totalp)
            t_fwd_s += time.perf_counter() - f0

        n_seen += int(power.shape[0])

    if dev.type == "cuda":
        torch.cuda.synchronize()

    t_total_s = time.perf_counter() - t_wall0

    per_case_total_ms = t_total_s * 1e3 / max(n_seen, 1)
    per_case_dl_ms = t_dl_s * 1e3 / max(n_seen, 1)
    per_case_total_minus_dl_ms = (t_total_s - t_dl_s) * 1e3 / max(n_seen, 1)

    return {
        "ckpt": ckpt_path,
        "device": str(dev),
        "n_cases": int(n_seen),
        "eval_bs": int(eval_bs),
        "num_workers": int(num_workers),
        "pin_memory": bool(pin_memory),
        "dataset_build_s": float(ds_build_s),
        "total_wall_s": float(t_total_s),
        "forward_only_s": float(t_fwd_s),
        "dataloader_like_s": float(t_dl_s),
        "per_case_total_ms": float(per_case_total_ms),
        "per_case_forward_ms": float(t_fwd_s * 1e3 / max(n_seen, 1)),
        "per_case_dataloader_like_ms": float(per_case_dl_ms),
        "per_case_total_minus_dataloader_ms": float(per_case_total_minus_dl_ms),
        "total_minus_forward_s": float(t_total_s - t_fwd_s),
        "total_minus_forward_minus_dl_s": float(t_total_s - t_fwd_s - t_dl_s),
        "model_param_dtype": str(next(model.parameters()).dtype),
    }


def _replace_fp32_block(*, log_path: str, new_block: str) -> None:
    with open(log_path, "r", encoding="utf-8") as f:
        s = f.read()

    # Replace the [fp32] block under [model_timing] (first occurrence of [fp32]).
    start_tok = "[fp32]"
    i0 = s.find(start_tok)
    if i0 < 0:
        raise RuntimeError(f"Did not find {start_tok} in {log_path}")

    next_tok = "\n[hotspot_timing]"
    i1 = s.find(next_tok, i0)
    if i1 < 0:
        raise RuntimeError(f"Did not find [hotspot_timing] after {start_tok} in {log_path}")

    out = s[:i0] + new_block + s[i1:]

    with open(log_path, "w", encoding="utf-8") as f:
        f.write(out)


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--fp32_ckpt",
        type=str,
        default="/root/placement/flow_GCN/thermalmodel/checkpoints/guidance_b96_lr2e-4_ep200_20260430_resume/guidance_net_b96_lr2e-4_totalp_avg_resume_ep0195_seed0_bs8_lr2e-04_base96_gw0p1.pth",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval_bs", type=int, default=8)
    ap.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--n_cases", type=int, default=1000)
    ap.add_argument(
        "--thermal_map_root",
        type=str,
        default="/root/placement/flow_GCN/Dataset/dataset/output/thermal/thermal_map",
    )
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--pin_memory", action="store_true")
    ap.add_argument(
        "--out_time_log",
        type=str,
        default="/root/placement/flow_GCN/thermalmodel/test_result/per_case_time.log",
    )
    return ap


def main() -> None:
    args = build_argparser().parse_args()

    t = time_fp32_ckpt(
        args.fp32_ckpt,
        n_cases=args.n_cases,
        eval_bs=args.eval_bs,
        seed=args.seed,
        device=args.device,
        thermal_map_root=args.thermal_map_root,
        num_workers=args.num_workers,
        pin_memory=bool(args.pin_memory),
    )

    new_block = (
        "[fp32]\n"
        f"ckpt={t['ckpt']}\n"
        f"device={t['device']}\n"
        f"n_cases={t['n_cases']} eval_bs={t['eval_bs']} num_workers={t['num_workers']} pin_memory={t['pin_memory']}\n"
        f"model_param_dtype={t['model_param_dtype']}\n"
        f"dataset_build_s={t['dataset_build_s']:.6f}\n"
        f"total_wall_s={t['total_wall_s']:.6f}\n"
        f"forward_only_s={t['forward_only_s']:.6f}\n"
        f"dataloader_like_s={t['dataloader_like_s']:.6f}\n"
        f"per_case_total_ms={t['per_case_total_ms']:.6f}\n"
        f"per_case_forward_ms={t['per_case_forward_ms']:.6f}\n"
        f"per_case_dataloader_like_ms={t['per_case_dataloader_like_ms']:.6f}\n"
        f"per_case_total_minus_dataloader_ms={t['per_case_total_minus_dataloader_ms']:.6f}\n"
        f"total_minus_forward_s={t['total_minus_forward_s']:.6f}\n"
        f"total_minus_forward_minus_dl_s={t['total_minus_forward_minus_dl_s']:.6f}\n"
        "\n"
    )

    _replace_fp32_block(log_path=args.out_time_log, new_block=new_block)
    print(f"UPDATED [fp32] block in {args.out_time_log}")


if __name__ == "__main__":
    main()
