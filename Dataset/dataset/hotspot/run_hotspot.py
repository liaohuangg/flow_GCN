import argparse
import os
import re
import shutil
import time
import shlex
from pathlib import Path


def _configure_matplotlib_backend() -> None:
    # Avoid Qt backend crashes in headless env.
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
    except Exception:
        pass


_configure_matplotlib_backend()




def _read_grid_steady_layer(grid_steady_file: Path, layer_num: int) -> tuple[list[float] | None, int | None, int | None]:
    """Read HotSpot .grid.steady values for a layer; infer rows/cols as sqrt(n)."""
    if not grid_steady_file.is_file():
        return None, None, None

    temps: list[float] = []
    in_target = False
    header = f"Layer {layer_num}:"

    for raw in grid_steady_file.read_text(encoding="utf-8", errors="replace").splitlines():
        s = raw.strip()
        if not s:
            continue
        if s.startswith("Layer ") and s.endswith(":"):
            if s == header:
                in_target = True
                continue
            if in_target:
                break
            continue
        if not in_target:
            continue

        parts = re.split(r"\s+", s)
        if len(parts) >= 2:
            try:
                temps.append(float(parts[1]))
            except ValueError:
                pass

    if not temps:
        return None, None, None

    n = len(temps)
    side = int(round(n ** 0.5))
    if side * side != n:
        side = int(n ** 0.5)
    return temps, side, side


def _read_flp_layout_mm(flp_file: Path) -> tuple[list[tuple[str, float, float, float, float]], list[tuple[str, float, float, float, float]]]:
    """Return (chiplets, tims) from FLP, converted to mm.

    FLP line: name width height x y (in meters).
    Chiplet: name is single uppercase letter or startswith 'chiplet'.
    TIM: name startswith 'TIM' or 'T'+digits.
    """
    chiplets: list[tuple[str, float, float, float, float]] = []
    tims: list[tuple[str, float, float, float, float]] = []
    if not flp_file.is_file():
        return chiplets, tims

    for raw in flp_file.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = re.split(r"\s+", line)
        if len(parts) < 5:
            continue
        name = parts[0]
        try:
            w_m, h_m, x_m, y_m = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        except ValueError:
            continue
        w, h, x, y = w_m * 1000.0, h_m * 1000.0, x_m * 1000.0, y_m * 1000.0

        if name.startswith("chiplet") or (len(name) == 1 and name.isupper()):
            chiplets.append((name, w, h, x, y))
        elif name.startswith("TIM") or (name.startswith("T") and len(name) > 1 and name[1:].isdigit()):
            tims.append((name, w, h, x, y))

    return chiplets, tims


def _read_ptrace_power_dict(ptrace_file: Path) -> dict[str, float]:
    power_dict: dict[str, float] = {}
    if not ptrace_file.is_file():
        return power_dict
    lines = ptrace_file.read_text(encoding="utf-8", errors="replace").splitlines()
    if len(lines) < 2:
        return power_dict
    names = re.split(r"\s+", lines[0].strip())
    powers = re.split(r"\s+", lines[1].strip())
    if len(names) != len(powers):
        return power_dict
    for n, p in zip(names, powers):
        try:
            power_dict[n] = float(p)
        except ValueError:
            pass
    return power_dict


def plot_grid_layer2_thermal_map(
    flp_file: Path,
    grid_steady_file: Path,
    output_image: Path,
    layer_num: int = 2,
) -> None:
    """Plot Layer 2 grid thermal map (HotSpot palette) and overlay chiplets + power labels."""
    try:
        import matplotlib.patches as patches
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
        from matplotlib.patheffects import withStroke
        import numpy as np
    except Exception as e:
        print(f"[plot] skip: matplotlib unavailable ({e})")
        return

    power_dict = _read_ptrace_power_dict(flp_file.with_suffix(".ptrace"))

    temps, rows, cols = _read_grid_steady_layer(grid_steady_file, layer_num)
    if not temps or rows is None or cols is None:
        print(f"[plot] skip: cannot read Layer {layer_num} from {grid_steady_file}")
        return

    temps_k = np.array(temps, dtype=float).reshape(rows, cols)
    temps_k = np.flipud(temps_k)
    temps_c = temps_k - 273.15

    chiplets, tims = _read_flp_layout_mm(flp_file)
    all_blocks = chiplets + tims
    total_width = total_length = 0.0
    for _, w, h, x, y in all_blocks:
        total_width = max(total_width, x + w)
        total_length = max(total_length, y + h)
    if total_width <= 0 or total_length <= 0:
        total_width = total_length = max(0.01, 0.05)

    palette_rgb = [
        (255, 0, 0),
        (255, 51, 0),
        (255, 102, 0),
        (255, 153, 0),
        (255, 204, 0),
        (255, 255, 0),
        (204, 255, 0),
        (153, 255, 0),
        (102, 255, 0),
        (51, 255, 0),
        (0, 255, 0),
        (0, 255, 51),
        (0, 255, 102),
        (0, 255, 153),
        (0, 255, 204),
        (0, 255, 255),
        (0, 204, 255),
        (0, 153, 255),
        (0, 102, 255),
        (0, 51, 255),
        (0, 0, 255),
    ]
    palette_rgb = list(reversed(palette_rgb))
    palette_norm = [(r / 255.0, g / 255.0, b / 255.0) for r, g, b in palette_rgb]
    cmap = ListedColormap(palette_norm, name="hotspot_grid_palette")

    fig, ax = plt.subplots(1, figsize=(10, 8))
    im = ax.imshow(
        temps_c,
        cmap=cmap,
        extent=(0, total_width, 0, total_length),
        origin="lower",
        aspect="auto",
    )

    max_c = float(np.max(temps_c))
    avg_c = float(np.mean(temps_c))
    im.set_clim(float(np.min(temps_c)), max_c)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Temperature (°C)", fontsize=18)
    cbar.ax.tick_params(labelsize=14)

    ax.set_title(f"Layer {layer_num} Grid Thermal Map (Max = {max_c:.2f} °C, AVG = {avg_c:.2f} °C)", fontsize=18)
    ax.set_xlabel("X (mm)", fontsize=18)
    ax.set_ylabel("Y (mm)", fontsize=18)
    ax.tick_params(axis="both", labelsize=14)

    for name, w, h, x, y in chiplets:
        rect = patches.Rectangle((x, y), w, h, linewidth=1.5, edgecolor="black", facecolor="none")
        ax.add_patch(rect)
        cx, cy = x + w / 2.0, y + h / 2.0
        power = power_dict.get(name, 0.0)
        power_text = f"{power:.2f}W" if power > 0 else ""
        label = f"{name}\n{power_text}" if power_text else name
        ax.text(
            cx,
            cy,
            label,
            ha="center",
            va="center",
            fontsize=14,
            fontweight="bold",
            color="white",
            linespacing=1.2,
            path_effects=[withStroke(linewidth=2, foreground="black")],
        )

    for name, w, h, x, y in tims:
        rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor="red", facecolor="none", alpha=0.7)
        ax.add_patch(rect)
        cx, cy = x + w / 2.0, y + h / 2.0
        ax.text(
            cx,
            cy,
            name,
            ha="center",
            va="center",
            fontsize=12,
            color="white",
            path_effects=[withStroke(linewidth=1, foreground="black")],
        )

    output_image.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_image, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"[plot] saved: {output_image}")

def _read_bbox_width_from_system_sub_flp(system_sub_flp: Path) -> float:
    """Read Unit0 width from system_sub.flp (meters)."""
    for raw in system_sub_flp.read_text(encoding="utf-8", errors="replace").splitlines():
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        parts = re.split(r"\s+", s)
        if len(parts) < 3:
            continue
        if parts[0] != "Unit0":
            continue
        try:
            return float(parts[1])
        except ValueError:
            break
    raise ValueError(f"无法从 {system_sub_flp} 解析 Unit0 宽度")


def _replace_flag_value(config_text: str, flag: str, new_value: float) -> str:
    """Replace config flag numeric value with new_value (first occurrence only)."""
    pattern = re.compile(
        rf"(^[ \t]*{re.escape(flag)}[ \t]+)([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
        re.MULTILINE,
    )

    def repl(m: re.Match) -> str:
        prefix = m.group(1)
        return f"{prefix}{new_value:.6f}"

    out, n = pattern.subn(repl, config_text, count=1)
    if n != 1:
        raise ValueError(f"在 hotspot.config 中未找到或找到多个字段：{flag}（匹配次数={n}）")
    return out



def prepare_hotspot_configs(
    configs_root: Path,
    template_config: Path,
    start_id: int | None = None,
    end_id: int | None = None,
    dry_run: bool = False,
) -> int:
    """
    1) 将 template_config 复制到每个 system_i_config 子目录下（文件名 hotspot.config）
    2) 读取子目录内 system_sub.flp，获取 bbox_width
    3) 修改复制后的 hotspot.config：-s_sink 与 -s_spreader 的数值都设为 bbox_width
    """
    if (start_id is None) != (end_id is None):
        raise ValueError("范围过滤需要同时提供 start_id 和 end_id")
    if start_id is not None and end_id is not None and start_id > end_id:
        raise ValueError("start_id 不能大于 end_id")

    configs_root = configs_root.resolve()
    template_config = template_config.resolve()
    if not template_config.is_file():
        raise FileNotFoundError(f"模板 config 不存在：{template_config}")
    if not configs_root.is_dir():
        raise FileNotFoundError(f"输出根目录不存在：{configs_root}")

    updated = 0
    for subdir in sorted(configs_root.glob("system_*_config")):
        if not subdir.is_dir():
            continue

        m = re.fullmatch(r"system_(\d+)_config", subdir.name)
        if not m:
            continue
        i = int(m.group(1))
        if start_id is not None and end_id is not None:
            if i < start_id or i > end_id:
                continue

        system_sub_flp = subdir / "system_sub.flp"
        if not system_sub_flp.is_file():
            raise FileNotFoundError(f"缺少 {system_sub_flp}")

        bbox_width = _read_bbox_width_from_system_sub_flp(system_sub_flp)
        bbox_width = float(bbox_width) + 0.002

        dst_config = subdir / "hotspot.config"
        if not dry_run:
            shutil.copyfile(template_config, dst_config)

            cfg = dst_config.read_text(encoding="utf-8", errors="replace")
            cfg = _replace_flag_value(cfg, "-s_sink", bbox_width)
            cfg = _replace_flag_value(cfg, "-s_spreader", bbox_width)
            dst_config.write_text(cfg, encoding="utf-8")

        updated += 1

    return updated


def _rename_if_exists(dir_path: Path, src: str, dst: str) -> None:
    src_p = dir_path / src
    dst_p = dir_path / dst
    if not src_p.exists():
        raise FileNotFoundError(f"未找到输出文件：{src_p}")
    if dst_p.exists():
        dst_p.unlink()
    src_p.rename(dst_p)


def _move_to_dir(src_path: Path, dst_dir: Path, dst_name: str) -> Path:
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst_path = dst_dir / dst_name
    if not src_path.exists():
        raise FileNotFoundError(f"未找到输出文件：{src_path}")
    if dst_path.exists():
        dst_path.unlink()
    shutil.move(str(src_path), str(dst_path))
    return dst_path


def run_hotspot(
    configs_root: Path,
    start_id: int | None = None,
    end_id: int | None = None,
    csv_output_root: Path | None = None,
    model_type: str = "grid",
    detailed_3d: str = "on",
    plot: bool = False,
) -> int:
    """
    参考 ThermalGCN/dataset/run.py 的 HotSpot 调用方式，在每个 system_i_config 下运行：

    - hotspot_bin: dataset/hotspot/hotspot
    - config_file: system_i_config/hotspot.config
    - layer_file: dataset/hotspot/Chiplet.lcf
    - flp/ptrace: system_i_config/system.flp 与 system_i_config/system.ptrace

    运行后，将子目录内生成的：
    Edge.csv / Power.csv / Temperature.csv / totalPower.csv
    重命名为：
    Edge_i.csv / Power_i.csv / Temperature_i.csv / totalPower_i.csv
    """
    if (start_id is None) != (end_id is None):
        raise ValueError("范围过滤需要同时提供 start_id 和 end_id")
    if start_id is not None and end_id is not None and start_id > end_id:
        raise ValueError("start_id 不能大于 end_id")

    script_dir = Path(__file__).resolve().parent
    hotspot_bin = script_dir / "HotSpot" / "hotspot"

    if csv_output_root is None:
        csv_output_root = (script_dir / "../output/thermal/thermal_map").resolve()
    else:
        csv_output_root = csv_output_root.resolve()

    configs_root = configs_root.resolve()
    if not configs_root.is_dir():
        raise FileNotFoundError(f"configs_root 不存在：{configs_root}")
    if not hotspot_bin.is_file():
        raise FileNotFoundError(f"hotspot_bin 不存在：{hotspot_bin}")

    ran = 0
    for subdir in sorted(configs_root.glob("system_*_config")):
        if not subdir.is_dir():
            continue
        m = re.fullmatch(r"system_(\d+)_config", subdir.name)
        if not m:
            continue
        i = int(m.group(1))
        if start_id is not None and end_id is not None:
            if i < start_id or i > end_id:
                continue

        config_file = subdir / "hotspot.config"
        flp_file = subdir / "system.flp"
        ptrace_file = subdir / "system.ptrace"
        layer_file = subdir / "Chiplet.lcf"
        steady_file = subdir / "Chiplet.steady"
        grid_steady_file = subdir / "Chiplet.grid.steady"
        material_file = script_dir / "example.materials"

        for p in (config_file, flp_file, layer_file):
            if not p.is_file():
                raise FileNotFoundError(f"缺少运行所需文件：{p}")

        # 同一布局下，遍历 system_1.ptrace..system_10.ptrace 做热仿真
        ptrace_candidates = sorted(subdir.glob("system_*.ptrace"))
        if not ptrace_candidates:
            # 兼容旧逻辑：只有 system.ptrace
            if not ptrace_file.is_file():
                raise FileNotFoundError(f"缺少运行所需文件：{ptrace_file}")
            ptrace_candidates = [ptrace_file]

        for ptrace_path in ptrace_candidates:
            mptr = re.fullmatch(r"system_(\d+)\.ptrace", ptrace_path.name)
            j = int(mptr.group(1)) if mptr else 0

            # 按要求：命令行里使用相对路径（相对于 subdir）
            hotspot_rel = os.path.relpath(hotspot_bin, start=subdir)
            config_rel = os.path.relpath(config_file, start=subdir)
            flp_rel = os.path.relpath(flp_file, start=subdir)
            ptrace_rel = os.path.relpath(ptrace_path, start=subdir)
            layer_rel = os.path.relpath(layer_file, start=subdir)
            steady_rel = os.path.relpath(steady_file, start=subdir)
            grid_steady_rel = os.path.relpath(grid_steady_file, start=subdir)
            materials = os.path.relpath(material_file, start=subdir)
            # 参考 ThermalGCN/dataset/run.py：用 os.system(cmd) 调用，并输出命令字符串
            cmd_parts = [
                hotspot_rel,
                "-c",
                config_rel,
                "-f",
                flp_rel,
                "-p",
                ptrace_rel,
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
                materials
            ]
            cmd_str = " ".join(shlex.quote(x) for x in cmd_parts)
            print(cmd_str)
            t0 = time.time()
            # 在子目录中执行，保证相对路径生效且输出文件落在该目录
            old_cwd = os.getcwd()
            try:
                os.chdir(subdir)
                rc = os.system(cmd_str)
            finally:
                os.chdir(old_cwd)
            if rc != 0:
                raise RuntimeError(f"HotSpot 执行失败：system_{i} ptrace={ptrace_path.name}，返回码={rc}")
            t1 = time.time()
            print(f"[run_hotspot] system_{i} ptrace_{j}: {t1 - t0:.3f}s")

            # 重命名本次仿真生成的 grid steady 文件，带上 j
            grid_steady_for_plot = subdir / "Chiplet.grid.steady"
            if j > 0:
                _rename_if_exists(subdir, "Chiplet.grid.steady", f"Chiplet_{j}.grid.steady")
                _rename_if_exists(subdir, "Chiplet.steady", f"Chiplet_{j}.steady")
                grid_steady_for_plot = subdir / f"Chiplet_{j}.grid.steady"

            if plot:
                # 让 plot 使用对应 ptrace（避免一直读 system.ptrace 造成 i_j 对不上）
                ptrace_symlink = subdir / "system.ptrace"
                if ptrace_symlink.exists():
                    ptrace_symlink.unlink()
                try:
                    ptrace_symlink.symlink_to(ptrace_path.name)
                except OSError:
                    shutil.copyfile(ptrace_path, ptrace_symlink)

                # 每个 ptrace 生成一张 thermal 图（文件名带 j）
                fig_out = (script_dir / "../output/thermal/thermal_map/fig" / f"system_{i}_thermal_{j}.png").resolve()
                plot_grid_layer2_thermal_map(
                    flp_file=flp_file,
                    grid_steady_file=grid_steady_for_plot,
                    output_image=fig_out,
                    layer_num=2,
                )

        # _move_to_dir(subdir / "Edge.csv", csv_output_root, f"Edge_{i}.csv")
        # _move_to_dir(subdir / "Power.csv", csv_output_root, f"Power_{i}.csv")
        # _move_to_dir(subdir / "Temperature.csv", csv_output_root, f"Temperature_{i}.csv")
        # _move_to_dir(subdir / "totalPower.csv", csv_output_root, f"totalPower_{i}.csv")

        ran += 1

    return ran


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="为每个 system_i_config 复制 hotspot.config 并按 system_sub.flp 修改 -s_sink/-s_spreader",
    )
    # 为每个子目录下，创建data子目录（用于存放.csv文件)
    parser.add_argument(
        "--configs_root",
        type=str,
        default=str((script_dir / "../output/thermal/hotspot_config").resolve()),
        help="包含 system_i_config 子目录的根目录",
    )
    parser.add_argument(
        "--template_config",
        type=str,
        default=str((script_dir / "hotspot.config").resolve()),
        help="模板 hotspot.config 路径（建议使用相对脚本路径）",
    )
    parser.add_argument("--start_id", type=int, default=None, help="起始编号（包含）")
    parser.add_argument("--end_id", type=int, default=None, help="结束编号（包含）")
    parser.add_argument("--dry_run", action="store_true", help="只检查不写文件")
    parser.add_argument("--plot", action="store_true", help="生成 thermal 图（默认不生成）")
    args = parser.parse_args()

    if not args.dry_run:
        # 先确保每个 system_i_config 下都有 hotspot.config
        updated = prepare_hotspot_configs(
            configs_root=Path(args.configs_root),
            template_config=Path(args.template_config),
            start_id=args.start_id,
            end_id=args.end_id,
            dry_run=args.dry_run,
        )
        print(f"✅ 完成：处理 {updated} 个 system_i_config 目录")

        ran = run_hotspot(
            configs_root=Path(args.configs_root),
            start_id=args.start_id,
            end_id=args.end_id,
            plot=args.plot,
        )
        print(f"✅ HotSpot 完成：运行 {ran} 个 system_i_config 目录")


if __name__ == "__main__":
    main()

'''
hotspot可执行路径在 /root/placement/flow_GCN/Dataset/dataset/hotspot/HotSpot/hotspot
python3 /root/workspace/flow_GCN/Dataset/dataset/hotspot/run_hotspot.py \
  --configs_root /root/workspace/flow_GCN/Dataset/dataset/output/thermal/hotspot_config \
  --start_id 1 --end_id 1000

python3 /root/workspace/flow_GCN/Dataset/dataset/hotspot/run_hotspot.py \
  --configs_root /root/workspace/flow_GCN/Dataset/dataset/output/thermal/hotspot_config \
  --start_id 1 --end_id 2

python3 /root/workspace/flow_GCN/Dataset/dataset/hotspot/run_hotspot.py \
  --configs_root /root/workspace/flow_GCN/Dataset/dataset/output/thermal/hotspot_config \
  --start_id 1 --end_id 1000 \
  --dry_run
'''