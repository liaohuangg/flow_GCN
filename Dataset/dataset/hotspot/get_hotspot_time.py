import argparse
import os
import re
import shlex
import time
from pathlib import Path
from typing import List, Tuple


def _list_first_n_cases(thermal_map_root: Path, n: int) -> List[Tuple[int, int]]:
    power_dir = thermal_map_root / "powercsv"
    if not power_dir.is_dir():
        raise FileNotFoundError(f"powercsv dir not found: {power_dir}")

    pat = re.compile(r"system_power_(\d+)_(\d+)\.csv$")
    cases: List[Tuple[int, int]] = []
    for fn in os.listdir(power_dir):
        m = pat.match(fn)
        if not m:
            continue
        cases.append((int(m.group(1)), int(m.group(2))))
    cases.sort()
    return cases[:n]


def _run_one_hotspot_case(
    *,
    hotspot_bin: Path,
    subdir: Path,
    model_type: str,
    detailed_3d: str,
) -> float:
    # Required inputs in subdir
    config_file = subdir / "hotspot.config"
    flp_file = subdir / "system.flp"
    layer_file = subdir / "Chiplet.lcf"

    # HotSpot output targets (we won't parse them; just avoid plotting/csv)
    steady_file = subdir / "Chiplet.steady"
    grid_steady_file = subdir / "Chiplet.grid.steady"

    script_dir = Path(__file__).resolve().parent
    material_file = script_dir / "example.materials"

    for p in (hotspot_bin, config_file, flp_file, layer_file, material_file):
        if not p.is_file():
            raise FileNotFoundError(f"Missing required file: {p}")

    # Use relative paths and run inside subdir (same as run_hotspot.py)
    hotspot_rel = os.path.relpath(hotspot_bin, start=subdir)
    config_rel = os.path.relpath(config_file, start=subdir)
    flp_rel = os.path.relpath(flp_file, start=subdir)
    layer_rel = os.path.relpath(layer_file, start=subdir)
    steady_rel = os.path.relpath(steady_file, start=subdir)
    grid_steady_rel = os.path.relpath(grid_steady_file, start=subdir)
    materials_rel = os.path.relpath(material_file, start=subdir)

    cmd_parts = [
        hotspot_rel,
        "-c",
        config_rel,
        "-f",
        flp_rel,
        "-p",
        # ptrace is expected to be system.ptrace in subdir
        "system.ptrace",
        "-steady_file",
        steady_rel,
        "-model_type",
        model_type,
        "-detailed_3D",
        detailed_3d,
        "-grid_layer_file",
        layer_rel,
        "-grid_steady_file",
        grid_steady_rel,
        "-materials_file",
        materials_rel,
    ]
    cmd_str = " ".join(shlex.quote(x) for x in cmd_parts)

    t0 = time.perf_counter()
    old_cwd = os.getcwd()
    try:
        os.chdir(subdir)
        rc = os.system(cmd_str)
    finally:
        os.chdir(old_cwd)
    t1 = time.perf_counter()

    if rc != 0:
        raise RuntimeError(f"HotSpot failed in {subdir} (rc={rc})")

    return t1 - t0


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Measure HotSpot runtime for first N thermal_map cases.")
    ap.add_argument(
        "--thermal_map_root",
        type=str,
        default="/root/placement/flow_GCN/Dataset/dataset/output/thermal/thermal_map",
    )
    ap.add_argument(
        "--hotspot_config_root",
        type=str,
        default="/root/placement/flow_GCN/Dataset/dataset/output/thermal/hotspot_config",
    )
    ap.add_argument(
        "--hotspot_bin",
        type=str,
        default="/root/placement/flow_GCN/Dataset/dataset/hotspot/HotSpot/hotspot",
    )
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--model_type", type=str, default="grid")
    ap.add_argument("--detailed_3d", type=str, default="on")
    return ap


def main() -> None:
    args = build_argparser().parse_args()

    thermal_map_root = Path(args.thermal_map_root)
    hotspot_config_root = Path(args.hotspot_config_root)
    hotspot_bin = Path(args.hotspot_bin)

    cases = _list_first_n_cases(thermal_map_root, args.n)
    if not cases:
        raise RuntimeError("No cases found")

    t_total = 0.0
    per_case_s: List[float] = []

    # For each (i,j), set system.ptrace -> system_{j}.ptrace, then run once.
    for idx, (i, j) in enumerate(cases):
        subdir = hotspot_config_root / f"system_{i}_config"
        if not subdir.is_dir():
            raise FileNotFoundError(f"Missing config dir: {subdir}")

        src_ptrace = subdir / f"system_{j}.ptrace"
        if not src_ptrace.is_file():
            # fallback to system.ptrace if exists
            fallback = subdir / "system.ptrace"
            if not fallback.is_file():
                raise FileNotFoundError(f"Missing ptrace: {src_ptrace} (and no system.ptrace fallback)")
        else:
            dst = subdir / "system.ptrace"
            if dst.exists():
                try:
                    dst.unlink()
                except Exception:
                    pass
            try:
                dst.symlink_to(src_ptrace.name)
            except OSError:
                # If symlink not allowed, copy file
                import shutil

                shutil.copyfile(src_ptrace, dst)

        dt = _run_one_hotspot_case(
            hotspot_bin=hotspot_bin,
            subdir=subdir,
            model_type=args.model_type,
            detailed_3d=args.detailed_3d,
        )
        per_case_s.append(dt)
        t_total += dt

        if (idx + 1) % 50 == 0:
            print(f"[hotspot_time] {idx+1}/{len(cases)} cases, avg_s={t_total/(idx+1):.6f}")

    avg_s = t_total / len(per_case_s)
    p50 = sorted(per_case_s)[len(per_case_s) // 2]

    # Print a machine-readable single line for the caller to parse.
    # Units are seconds.
    print(
        "hotspot_time"
        f" n_cases={len(per_case_s)}"
        f" total_s={t_total:.6f}"
        f" avg_s={avg_s:.6f}"
        f" p50_s={p50:.6f}"
        f" thermal_map_root={thermal_map_root}"
        f" hotspot_config_root={hotspot_config_root}"
        f" model_type={args.model_type}"
        f" detailed_3d={args.detailed_3d}"
    )


if __name__ == "__main__":
    main()
