import argparse
import os
import re
import shutil
import time
import shlex
from pathlib import Path


def _read_bbox_width_from_system_sub_flp(system_sub_flp: Path) -> float:
    """
    读取 system_sub.flp 中 Unit0 行的 <width> <height>，返回 bbox_width。
    约定：width/height 应相等（外接框正方形）；若不相等则取二者最大值。
    """
    text = system_sub_flp.read_text(encoding="utf-8", errors="replace")
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = re.split(r"\s+", line)
        if len(parts) < 3:
            continue
        if parts[0] != "Unit0":
            continue
        w = float(parts[1])
        h = float(parts[2])
        return w if abs(w - h) < 1e-12 else max(w, h)
    raise ValueError(f"未在 {system_sub_flp} 中找到 Unit0 行")


def _replace_flag_value(config_text: str, flag: str, new_value: float) -> str:
    """
    将 config 中形如：<whitespace><flag><whitespace><number> 的 number 替换为 new_value。
    保留原有缩进与分隔符风格。
    """
    # 例：\t\t-s_sink\t\t\t0.06
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
    hotspot_bin = script_dir / "hotspot"

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
        material_file = subdir / "example.materials"

        for p in (config_file, flp_file, ptrace_file, layer_file):
            if not p.is_file():
                raise FileNotFoundError(f"缺少运行所需文件：{p}")

        # 按要求：命令行里使用相对路径（相对于 subdir）
        hotspot_rel = os.path.relpath(hotspot_bin, start=subdir)
        config_rel = os.path.relpath(config_file, start=subdir)
        flp_rel = os.path.relpath(flp_file, start=subdir)
        ptrace_rel = os.path.relpath(ptrace_file, start=subdir)
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
            raise RuntimeError(f"HotSpot 执行失败：system_{i}，返回码={rc}")
        t1 = time.time()
        print(f"[run_hotspot] system_{i}: {t1 - t0:.3f}s")

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
    args = parser.parse_args()

    # updated = prepare_hotspot_configs(
    #     configs_root=Path(args.configs_root),
    #     template_config=Path(args.template_config),
    #     start_id=args.start_id,
    #     end_id=args.end_id,
    #     dry_run=args.dry_run,
    # )
    # print(f"✅ 完成：处理 {updated} 个 system_i_config 目录")

    if not args.dry_run:
        ran = run_hotspot(
            configs_root=Path(args.configs_root),
            start_id=args.start_id,
            end_id=args.end_id,
        )
        print(f"✅ HotSpot 完成：运行 {ran} 个 system_i_config 目录")


if __name__ == "__main__":
    main()

'''
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