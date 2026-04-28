import argparse
import os
import sys

import torch

# Allow running `python guidance_pipeline.py ...` from within thermalmodel/
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from thermalmodel.guidance_model import ThermalGuidanceNet


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_fp32_model_from_ckpt(ckpt_path: str, device: torch.device) -> tuple[ThermalGuidanceNet, dict]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    base = int(ckpt.get("base", 32))
    model = ThermalGuidanceNet(base=base).to(device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    return model, ckpt


def export_fp16_from_fp32(ckpt_path: str, out_path: str) -> None:
    device = _device()
    model, ckpt = _load_fp32_model_from_ckpt(ckpt_path, device=device)

    # fp16 weights (typical for GPU inference). Keep buffers as-is.
    model.half()

    out = {
        "src_ckpt": ckpt_path,
        "epoch": int(ckpt.get("epoch", -1)),
        "model": model.state_dict(),
        "stats": ckpt.get("stats"),
        "grid_size": ckpt.get("grid_size", 128),
        "base": ckpt.get("base", 32),
        "fp16": True,
        "quant": "fp16",
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(out, out_path)
    print(f"[export] fp16 saved: {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train_fp32")
    t.add_argument("--epochs", type=int, default=200)
    t.add_argument("--batch_size", type=int, default=8)
    t.add_argument("--lr", type=float, default=5e-4)
    t.add_argument("--base", type=int, default=64)
    t.add_argument("--grad_w", type=float, default=0.1)
    t.add_argument("--ckpt_every", type=int, default=5)
    t.add_argument("--seed", type=int, default=0)

    q = sub.add_parser("train_qat")
    q.add_argument("--epochs", type=int, default=200)
    q.add_argument("--batch_size", type=int, default=8)
    q.add_argument("--lr", type=float, default=5e-4)
    q.add_argument("--base", type=int, default=64)
    q.add_argument("--grad_w", type=float, default=0.1)
    q.add_argument("--ckpt_every", type=int, default=5)
    q.add_argument("--seed", type=int, default=0)

    # Keep subcommand name for backward compatibility; now exports fp16 only.
    p = sub.add_parser("export_ptq")
    p.add_argument("--src_ckpt", type=str, required=True)
    p.add_argument("--out_dir", type=str, default="")

    args = ap.parse_args()

    ckpt_dir = os.path.join(os.path.dirname(__file__), "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    if args.cmd == "train_fp32":
        # Delegate to guidance_model.py
        out_dir = os.path.join(ckpt_dir, "fp32")
        os.makedirs(out_dir, exist_ok=True)
        cmd = (
            f"python {os.path.join(os.path.dirname(__file__), 'guidance_model.py')} train "
            f"--epochs {args.epochs} --batch_size {args.batch_size} --lr {args.lr} "
            f"--base {args.base} --grad_w {args.grad_w} --ckpt_every {args.ckpt_every} "
            f"--seed {args.seed} --ckpt_tag fp32 --out_dir {out_dir}"
        )
        print(cmd)
        os.system(cmd)
        return

    if args.cmd == "train_qat":
        out_dir = os.path.join(ckpt_dir, "qat")
        os.makedirs(out_dir, exist_ok=True)
        cmd = (
            f"python {os.path.join(os.path.dirname(__file__), 'guidance_model.py')} train "
            f"--epochs {args.epochs} --batch_size {args.batch_size} --lr {args.lr} "
            f"--base {args.base} --grad_w {args.grad_w} --ckpt_every {args.ckpt_every} "
            f"--seed {args.seed} --qat --ckpt_tag qat --out_dir {out_dir}"
        )
        print(cmd)
        os.system(cmd)
        return

    if args.cmd == "export_ptq":
        out_dir = args.out_dir or os.path.join(ckpt_dir, "ptq")
        os.makedirs(out_dir, exist_ok=True)

        stem = os.path.splitext(os.path.basename(args.src_ckpt))[0]
        fp16_path = os.path.join(out_dir, f"{stem}_fp16.pth")
        export_fp16_from_fp32(args.src_ckpt, fp16_path)
        return


if __name__ == "__main__":
    main()
