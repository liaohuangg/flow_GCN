from __future__ import annotations

import argparse


def add_legalizer_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--legalize-mode", default="none", choices=["none", "scheduled", "opt"])
    parser.add_argument("--legalize-steps", type=int, default=0)
    parser.add_argument("--legalize-step-size", type=float, default=0.2)
    parser.add_argument("--legalize-softmax-min", type=float, default=5.0)
    parser.add_argument("--legalize-softmax-max", type=float, default=50.0)
    parser.add_argument("--legalize-legality-weight", type=float, default=1.0)
    parser.add_argument("--legalize-hpwl-weight", type=float, default=0.0)
    parser.add_argument("--legalize-softmax-critical-factor", type=float, default=1.0)
    parser.add_argument("--legalize-guidance-critical-factor", type=float, default=1.0)
    parser.add_argument("--legalize-zero-hpwl-factor", type=float, default=1.0)
    parser.add_argument("--legalize-legality-increase-factor", type=float, default=1.0)
    parser.add_argument("--legalize-macros-only", action="store_true")


def legalizer_kwargs(args: argparse.Namespace) -> dict:
    return {
        "step_size": args.legalize_step_size,
        "grad_descent_steps": args.legalize_steps,
        "softmax_min": args.legalize_softmax_min,
        "softmax_max": args.legalize_softmax_max,
        "save_videos": False,
        "legality_weight": args.legalize_legality_weight,
        "hpwl_weight": args.legalize_hpwl_weight,
        "softmax_critical_factor": args.legalize_softmax_critical_factor,
        "guidance_critical_factor": args.legalize_guidance_critical_factor,
        "zero_hpwl_factor": args.legalize_zero_hpwl_factor,
        "legality_increase_factor": args.legalize_legality_increase_factor,
        "macros_only": args.legalize_macros_only,
    }
