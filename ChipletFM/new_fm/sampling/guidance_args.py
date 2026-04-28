from __future__ import annotations

import argparse
from typing import Any, Dict


def add_guidance_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--guidance-mode", default=None, choices=["none", "sgd", "opt"])
    parser.add_argument("--legality-guidance-weight", type=float, default=None)
    parser.add_argument("--hpwl-guidance-weight", type=float, default=None)
    parser.add_argument("--grad-descent-steps", type=int, default=None)
    parser.add_argument("--grad-descent-rate", type=float, default=None)
    parser.add_argument("--alpha-init", type=float, default=None)
    parser.add_argument("--alpha-lr", type=float, default=None)
    parser.add_argument("--alpha-critical-factor", type=float, default=None)
    parser.add_argument("--legality-potential-target", type=float, default=None)
    parser.add_argument("--legality-softmax-factor-min", type=float, default=None)
    parser.add_argument("--legality-softmax-factor-max", type=float, default=None)
    parser.add_argument("--legality-softmax-critical-factor", type=float, default=None)
    parser.add_argument("--use-adam-guidance", action="store_true")


def apply_guidance_overrides(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    cfg = dict(config)
    model_cfg = dict(cfg.get("model", {}))
    if model_cfg.get("name") != "legacy_fm":
        cfg["model"] = model_cfg
        return cfg

    mapping = {
        "guidance_mode": args.guidance_mode,
        "legality_guidance_weight": args.legality_guidance_weight,
        "hpwl_guidance_weight": args.hpwl_guidance_weight,
        "grad_descent_steps": args.grad_descent_steps,
        "grad_descent_rate": args.grad_descent_rate,
        "alpha_init": args.alpha_init,
        "alpha_lr": args.alpha_lr,
        "alpha_critical_factor": args.alpha_critical_factor,
        "legality_potential_target": args.legality_potential_target,
        "legality_softmax_factor_min": args.legality_softmax_factor_min,
        "legality_softmax_factor_max": args.legality_softmax_factor_max,
        "legality_softmax_critical_factor": args.legality_softmax_critical_factor,
    }
    for key, value in mapping.items():
        if value is not None:
            model_cfg[key] = value
    if args.use_adam_guidance:
        model_cfg["use_adam"] = True
    cfg["model"] = model_cfg
    return cfg

