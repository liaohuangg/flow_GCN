# 芯粒的布局在/root/placement/flow_GCN/Dataset/dataset/output/thermal/hotspot_config/system_i_config/system.flp中，芯粒的功率在/root/placement/flow_GCN/Dataset/dataset/output/thermal/hotspot_config/system_1_config/system.ptrace中
# 1. 所有的芯粒布局，都是以左下角坐标+GRID_MM = 0.01mm为单位的, 我将这种网格就称为mm_grid
# 2. 现在要求芯粒布局的网格要与热仿真的网格64*64或者32*32（应该是可以在程序中通过#define修改配置的）可以对应上，我将这种粗的网格称为hotspot_grid
# 3. 现在已知每个芯粒的坐标以及功率，要求生成一个64*64或者32*32的带有单位功率的矩阵，矩阵中的每个元素表示对应位置的功率值，如果该位置没有芯粒，则功率值为0，如果该热仿真网格中有多个芯粒的mm_grid单位功率，你考虑下应该怎么计算该hotspot_grid的单位功率
# 4. 如果该布局的mm_grid不能够整除hotspot_grid，那么你需要考虑如何处理边界情况，比如说如果一个芯粒的坐标在hotspot_grid的边界上，那么它应该被分配到哪个hotspot_grid中，或者说如果一个芯粒的坐标在两个hotspot_grid的交界处，那么它应该被分配到哪个hotspot_grid中
# 5. 最后，你需要将生成的64*64或者32*32的带有单位功率的矩阵(格式是第一列是网格编号，第二列是单位hotspot_grid的功耗，例如在64*64中，网格编号1-4096，按照行优先顺序组成这样的hotspot_grid
# 1, 2, 3, 4, ..., 64,        ← 第1行（hotspot_grid中的第一行）
# 65, 66, 67, ..., 128,       ← 第2行
# 129, 130, ..., 192,         ← 第3行
# 保存为csv文件放在/root/placement/flow_GCN/Dataset/dataset/output/thermal/thermal_map/powercsv目录下（使用相对路径哦），文件名可以根据芯粒布局的json文件名来命名，比如说如果芯粒布局的目录为system_1_config，那么生成的csv文件名可以是system_1_power.csv
# 实现好之后，先对5个芯粒布局进行这样的system_1_power.csv生成（这就是hotspot_grid 标号 单位hotspot_grid的功耗）

from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


@dataclass(frozen=True)
class RectMm:
    name: str
    x: float
    y: float
    w: float
    h: float

    @property
    def x2(self) -> float:
        return self.x + self.w

    @property
    def y2(self) -> float:
        return self.y + self.h


def _parse_floats(parts: List[str], idxs: Iterable[int]) -> Tuple[float, ...]:
    out: List[float] = []
    for i in idxs:
        out.append(float(parts[i]))
    return tuple(out)


def read_flp_rects(system_flp: Path) -> List[RectMm]:
    """读取 HotSpot .flp：每行格式为 <name> <width> <height> <x> <y>，单位 m。"""
    lines = system_flp.read_text(encoding="utf-8", errors="replace").splitlines()
    rects: List[RectMm] = []

    for line in lines:
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        parts = re.split(r"\s+", s)
        if len(parts) < 5:
            continue

        name = parts[0]
        w_m, h_m, x_m, y_m = _parse_floats(parts, (1, 2, 3, 4))
        rects.append(
            RectMm(
                name=name,
                x=x_m * 1000.0,
                y=y_m * 1000.0,
                w=w_m * 1000.0,
                h=h_m * 1000.0,
            )
        )

    if not rects:
        raise ValueError(f"未从 {system_flp} 解析到任何块")
    return rects


def read_ptrace_powers(system_ptrace: Path) -> Dict[str, float]:
    """读取 .ptrace：首行为模块名，次行为功耗（W）。"""
    lines = system_ptrace.read_text(encoding="utf-8", errors="replace").splitlines()
    if len(lines) < 2:
        raise ValueError(f"{system_ptrace} 行数不足，无法解析 ptrace")

    names = re.split(r"\s+", lines[0].strip())
    powers = re.split(r"\s+", lines[1].strip())
    if len(names) != len(powers):
        raise ValueError(f"{system_ptrace} name/power 列数不一致: {len(names)} vs {len(powers)}")

    out: Dict[str, float] = {}
    for n, p in zip(names, powers):
        out[n] = float(p)
    return out


def _intersect_area_mm2(a: RectMm, b: RectMm) -> float:
    x1 = max(a.x, b.x)
    y1 = max(a.y, b.y)
    x2 = min(a.x2, b.x2)
    y2 = min(a.y2, b.y2)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    return (x2 - x1) * (y2 - y1)


def accumulate_power_to_grid(rects_mm: List[RectMm], power_w: Dict[str, float], grid_n: int) -> List[float]:
    """将模块功耗按相交面积比例累加到 grid_n x grid_n 网格。

    - 热网格覆盖区域取所有模块的外接框 bbox。
    - 模块跨越多个 cell 时，按 (相交面积/模块面积) 分摊功耗。
    - 边界：用 floor/ceil-1 的方式确定覆盖 cell 范围，避免边界重复归属。
    """

    bbox_x0 = min(r.x for r in rects_mm)
    bbox_y0 = min(r.y for r in rects_mm)
    bbox_x1 = max(r.x2 for r in rects_mm)
    bbox_y1 = max(r.y2 for r in rects_mm)

    bbox_w = bbox_x1 - bbox_x0
    bbox_h = bbox_y1 - bbox_y0
    if bbox_w <= 0 or bbox_h <= 0:
        raise ValueError("bbox 尺寸非法")

    cell_w = bbox_w / grid_n
    cell_h = bbox_h / grid_n

    acc = [0.0 for _ in range(grid_n * grid_n)]

    for r in rects_mm:
        if r.name not in power_w:
            continue

        # 过滤 TIM 等（按 gen_flp_trace 的命名习惯：T + 数字）
        if r.name.startswith("T") and r.name[1:].isdigit():
            continue

        p = float(power_w[r.name])
        if p == 0.0:
            continue

        area = r.w * r.h
        if area <= 0:
            continue

        ix0 = max(0, min(grid_n - 1, int(math.floor((r.x - bbox_x0) / cell_w))))
        iy0 = max(0, min(grid_n - 1, int(math.floor((r.y - bbox_y0) / cell_h))))
        ix1 = max(0, min(grid_n - 1, int(math.ceil((r.x2 - bbox_x0) / cell_w) - 1)))
        iy1 = max(0, min(grid_n - 1, int(math.ceil((r.y2 - bbox_y0) / cell_h) - 1)))

        for iy in range(iy0, iy1 + 1):
            y0 = bbox_y0 + iy * cell_h
            y1 = y0 + cell_h
            for ix in range(ix0, ix1 + 1):
                x0 = bbox_x0 + ix * cell_w
                x1 = x0 + cell_w
                cell = RectMm(name="cell", x=x0, y=y0, w=(x1 - x0), h=(y1 - y0))
                inter = _intersect_area_mm2(r, cell)
                if inter <= 0:
                    continue
                acc[iy * grid_n + ix] += p * (inter / area)

    return acc


def write_powercsv(csv_path: Path, acc: List[float]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for idx, p in enumerate(acc, start=1):
            w.writerow([idx, f"{p:.6f}"])


def write_totalpowercsv(csv_path: Path, acc: List[float]) -> None:
    """Write total power as a single numeric value."""
    total = float(sum(acc))
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.write_text(f"{total:.6f}\n", encoding="utf-8")


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="从 system.flp + system.ptrace 生成 hotspot_grid 功耗 csv")
    parser.add_argument(
        "--configs_root",
        type=str,
        default=str((script_dir / "../output/thermal/hotspot_config").resolve()),
        help="包含 system_i_config 子目录的根目录",
    )
    parser.add_argument("--grid", type=int, default=64, help="网格尺寸（例如 32/64/128/256）")
    parser.add_argument("--startid", type=int, required=True, help="仅处理 i>=startid")
    parser.add_argument("--endid", type=int, required=True, help="仅处理 i<=endid")
    args = parser.parse_args()

    if args.startid > args.endid:
        raise ValueError("--startid 不能大于 --endid")

    configs_root = Path(args.configs_root).resolve()
    out_dir = (script_dir / "../output/thermal/thermal_map/powercsv").resolve()
    out_total_dir = (script_dir / "../output/thermal/thermal_map/totalpowercsv").resolve()

    made = 0
    for subdir in sorted(configs_root.glob("system_*_config")):
        m = re.fullmatch(r"system_(\d+)_config", subdir.name)
        if not m:
            continue
        i = int(m.group(1))

        if i < args.startid or i > args.endid:
            continue

        flp = subdir / "system.flp"
        if not flp.is_file():
            continue

        rects = read_flp_rects(flp)

        # 同一布局目录下，遍历 system_*.ptrace（排除 system.ptrace）
        ptraces = sorted(subdir.glob("system_*.ptrace"))
        ptraces = [p for p in ptraces if p.name != "system.ptrace"]
        if not ptraces:
            # 兼容旧逻辑：只有 system.ptrace
            ptrace = subdir / "system.ptrace"
            if not ptrace.is_file():
                continue
            ptraces = [ptrace]

        for ptrace in ptraces:
            mptr = re.fullmatch(r"system_(\d+)\.ptrace", ptrace.name)
            j = int(mptr.group(1)) if mptr else 0

            pw = read_ptrace_powers(ptrace)
            acc = accumulate_power_to_grid(rects, pw, grid_n=args.grid)

            csv_path = out_dir / f"system_power_{i}_{j}.csv"
            write_powercsv(csv_path, acc)
            print(f"✅ write {csv_path}")

            total_csv_path = out_total_dir / f"system_totalpower_{i}_{j}.csv"
            write_totalpowercsv(total_csv_path, acc)
            print(f"✅ write {total_csv_path}")

        made += 1

    print(f"done: {made}")


if __name__ == "__main__":
    main()
#   python gen_powercsv.py --grid 64 --startid 1 --endid 5  
