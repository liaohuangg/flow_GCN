import os
import re
from pathlib import Path


def _extract_layer_values(grid_steady_text: str, target_layer: int) -> list[tuple[int, float]]:
    """Parse HotSpot *.grid.steady content and return (idx, value) pairs for a layer."""
    header = f"Layer {target_layer}:"
    in_layer = False
    out: list[tuple[int, float]] = []

    for raw in grid_steady_text.splitlines():
        line = raw.strip()
        if not line:
            continue

        if line.startswith("Layer ") and line.endswith(":"):
            in_layer = (line == header)
            continue

        if not in_layer:
            continue

        # Expected: "<grid_id>\t<temp>" (or spaces)
        parts = re.split(r"\s+", line)
        if len(parts) < 2:
            continue
        try:
            idx = int(parts[0])
            val = float(parts[1])
        except ValueError:
            continue
        out.append((idx, val))

    return out


def main() -> None:
    # All paths are relative to this script's directory
    script_dir = Path(__file__).resolve().parent

    configs_root = (script_dir / "../output/thermal/hotspot_config").resolve()
    dst_dir = (script_dir / "../output/thermal/thermal_map/tempcsv").resolve()
    avg_dst_dir = (script_dir / "../output/thermal/thermal_map/avgtempcsv").resolve()
    dst_dir.mkdir(parents=True, exist_ok=True)
    avg_dst_dir.mkdir(parents=True, exist_ok=True)

    ran = 0
    for subdir in sorted(configs_root.glob("system_*_config")):
        if not subdir.is_dir():
            continue
        m = re.fullmatch(r"system_(\d+)_config", subdir.name)
        if not m:
            continue
        i = int(m.group(1))

        # 同一布局下支持多个温度输出：Chiplet_{j}.grid.steady（j 与 ptrace 序号一致）
        grid_candidates = sorted(subdir.glob("Chiplet_*.grid.steady"))
        if not grid_candidates:
            grid_candidates = [subdir / "Chiplet.grid.steady"]

        for grid_steady in grid_candidates:
            if not grid_steady.is_file():
                # Some runs may not have produced this file; skip.
                continue

            mgrid = re.fullmatch(r"Chiplet_(\d+)\.grid\.steady", grid_steady.name)
            j = int(mgrid.group(1)) if mgrid else 0

            values = _extract_layer_values(grid_steady.read_text(encoding="utf-8", errors="replace"), target_layer=2)
            if not values:
                # Some configs may not have Layer 2 depending on model/lcf; skip them.
                continue

            out_path = dst_dir / f"system_temp_{i}_{j}.csv"
            s = 0.0
            with out_path.open("w", encoding="utf-8", newline="\n") as f:
                for idx, val in values:
                    f.write(f"{idx},{val}\n")
                    s += float(val)

            avg = s / float(len(values))
            avg_out = avg_dst_dir / f"system_avgtemp_{i}_{j}.csv"
            avg_out.write_text(f"{avg}\n", encoding="utf-8")

            ran += 1

    print(f"done: {ran}")


if __name__ == "__main__":
    main()
