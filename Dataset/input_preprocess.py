#!/usr/bin/env python3
"""
Convert .cfg files to JSON format for chiplet placement.
"""

import os
import re
import json
from pathlib import Path
import random
from typing import List, Tuple, Dict, Optional

def parse_cfg_file(cfg_path):
    """
    Parse a .cfg file and extract chiplet information.
    
    Args:
        cfg_path: Path to the .cfg file
        
    Returns:
        dict with keys: widths, heights, powers, connections_matrix
    """
    with open(cfg_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Extract widths
    widths_match = re.search(r'widths\s*=\s*(.+?)(?:\n|$)', content, re.MULTILINE)
    if not widths_match:
        raise ValueError(f"Could not find 'widths' in {cfg_path}")
    widths_str = widths_match.group(1).strip()
    widths = [float(x.strip()) for x in widths_str.split(',') if x.strip()]
    
    # Extract heights
    heights_match = re.search(r'heights\s*=\s*(.+?)(?:\n|$)', content, re.MULTILINE)
    if not heights_match:
        raise ValueError(f"Could not find 'heights' in {cfg_path}")
    heights_str = heights_match.group(1).strip()
    heights = [float(x.strip()) for x in heights_str.split(',') if x.strip()]
    
    # Extract powers
    powers_match = re.search(r'powers\s*=\s*(.+?)(?:\n|$)', content, re.MULTILINE)
    if not powers_match:
        raise ValueError(f"Could not find 'powers' in {cfg_path}")
    powers_str = powers_match.group(1).strip()
    powers = [float(x.strip()) for x in powers_str.split(',') if x.strip()]
    
    # Extract connections matrix (may span multiple lines)
    # Find the connections section
    connections_match = re.search(r'connections\s*=\s*(.+?)(?=\n\n|\n\[|\n[a-z]+\s*=|$)', content, re.DOTALL)
    if not connections_match:
        raise ValueError(f"Could not find 'connections' in {cfg_path}")
    connections_str = connections_match.group(1).strip()
    
    # Parse the matrix: split by semicolon to get rows, then by comma to get values
    rows = []
    for row_str in connections_str.split(';'):
        row_str = row_str.strip()
        if not row_str:
            continue
        # Remove tabs and extra spaces, split by comma
        row = [int(x.strip()) for x in row_str.split(',') if x.strip()]
        if row:  # Only add non-empty rows
            rows.append(row)
    
    # Validate dimensions
    if len(widths) != len(heights) or len(widths) != len(powers):
        raise ValueError(f"Dimension mismatch: widths={len(widths)}, heights={len(heights)}, powers={len(powers)}")
    
    if len(rows) != len(widths):
        raise ValueError(f"Connections matrix rows ({len(rows)}) != chiplet count ({len(widths)})")
    
    for i, row in enumerate(rows):
        if len(row) != len(widths):
            raise ValueError(f"Connections matrix row {i} has {len(row)} columns, expected {len(widths)}")
    
    return {
        'widths': widths,
        'heights': heights,
        'powers': powers,
        'connections_matrix': rows
    }


def matrix_to_connections(connections_matrix, default_emib_type: str = "interfaceC"):
    """
    Convert adjacency matrix to list of connections.
    
    输出格式：每条边为无向边，包含 node1, node2, wireCount, EMIBType, EMIB_length, EMIB_max_width, EMIB_bump_width。
    wireCount 直接取自 cfg 的 connections 矩阵 connections_matrix[i][j]（chiplet i 与 j 之间的线数）。
    EMIBType 默认标注为 interfaceC，可用 update_connection.py 按范围重新标注。
    EMIB_length = wireCount / LinearIODensity，EMIB_max_width = max_Reach_length - 2*(wireCount/AreaIODensity)/EMIB_length，
    EMIB_bump_width = (wireCount / AreaIODensity) / EMIB_length。
    默认接口 interfaceC：LinearIODensity=40, max_Reach_length=100, AreaIODensity=80。
    
    Args:
        connections_matrix: 2D list representing adjacency matrix（对称矩阵，值为线数 wireCount）
        default_emib_type: 默认 EMIBType 标注（默认 "interfaceC"）
        
    Returns:
        List of dicts: [{"node1": "A", "node2": "B", "wireCount": 200, "EMIBType": "interfaceC", "EMIB_length": ..., "EMIB_max_width": ..., "EMIB_bump_width": ...}, ...]
    """
    linear_io = 40
    max_reach = 100.0
    area_io = 80.0

    connections = []
    num_chiplets = len(connections_matrix)
    
    def index_to_name(idx):
        return chr(ord('A') + idx)
    
    for i in range(num_chiplets):
        for j in range(i + 1, num_chiplets):
            wire_count = int(connections_matrix[i][j])
            if wire_count > 0:
                emib_length = wire_count / linear_io if linear_io > 0 else 2.5
                emib_max_width = max_reach - 2 * (wire_count / area_io) / emib_length if emib_length > 0 else max_reach
                emib_bump_width = (wire_count / area_io) / emib_length if emib_length > 0 and area_io > 0 else 0.0
                connections.append({
                    "node1": index_to_name(i),
                    "node2": index_to_name(j),
                    "wireCount": wire_count,
                    "EMIBType": default_emib_type,
                    "EMIB_length": round(emib_length, 4),
                    "EMIB_max_width": round(emib_max_width, 4),
                    "EMIB_bump_width": round(emib_bump_width, 4),
                })
    
    return connections


def cfg_to_json(cfg_path, output_dir, default_emib_type: str = "interfaceC"):
    """
    Convert a .cfg file to JSON format.
    
    Args:
        cfg_path: Path to input .cfg file
        output_dir: Directory to save the JSON file
        default_emib_type: 默认 EMIBType 标注（默认 "interfaceC"）
    """
    print(f"Processing: {cfg_path}")
    
    # Parse the .cfg file
    data = parse_cfg_file(cfg_path)
    
    # Create chiplets list
    chiplets = []
    for i in range(len(data['widths'])):
        chiplet_name = chr(ord('A') + i)  # 0->A, 1->B, 2->C, ...
        chiplets.append({
            'name': chiplet_name,
            'width': data['widths'][i],
            'height': data['heights'][i],
            'power': int(data['powers'][i])  # Power is typically an integer
        })
    
    # Convert connections matrix to list of {node1, node2, wireCount, EMIBType}
    connections = matrix_to_connections(data['connections_matrix'], default_emib_type=default_emib_type)
    
    # Create JSON structure
    json_data = {
        'chiplets': chiplets,
        'connections': connections
    }
    
    # Generate output filename
    cfg_name = Path(cfg_path).stem  # Get filename without extension
    output_path = os.path.join(output_dir, f"{cfg_name}.json")
    
    # Write JSON file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2)
    
    print(f"  -> Saved to: {output_path}")
    print(f"  -> Chiplets: {len(chiplets)}, Connections: {len(connections)}")
    return output_path


CFG_BW_CHOICES = [128, 256, 512, 1024]


def _random_chiplet_dims(
    chiplet_count: int,
    width_range: Tuple[int, int] = (3, 30),
    height_range: Tuple[int, int] = (3, 30),
    ar_min: float = 0.8,
    ar_max: float = 1.25,
    max_tries: int = 10000,
) -> Tuple[List[float], List[float]]:
    """
    Generate random chiplet widths/heights with an aspect ratio constraint.

    Aspect ratio is defined as (w/h) and must be within [ar_min, ar_max].
    """
    widths: List[float] = []
    heights: List[float] = []
    tries = 0
    while len(widths) < chiplet_count:
        tries += 1
        if tries > max_tries:
            raise RuntimeError("Failed to generate chiplet dimensions within aspect ratio constraints.")
        w = random.randint(width_range[0], width_range[1])
        h = random.randint(height_range[0], height_range[1])
        ar = w / h if h != 0 else float("inf")
        if ar_min <= ar <= ar_max:
            widths.append(float(w))
            heights.append(float(h))
    return widths, heights


def _generate_connected_graph_edges(
    n: int,
    extra_edge_prob: float = 0.25,
) -> List[Tuple[int, int, int]]:
    """
    Generate a connected undirected weighted graph on n nodes.

    Returns a list of edges (i, j, bw) with i < j.
    """
    if n < 1:
        return []
    if n == 1:
        return []

    # 1) Build a random spanning tree to guarantee connectivity.
    nodes = list(range(n))
    random.shuffle(nodes)
    edges: List[Tuple[int, int, int]] = []
    connected = {nodes[0]}
    remaining = set(nodes[1:])
    while remaining:
        a = random.choice(tuple(connected))
        b = random.choice(tuple(remaining))
        bw = random.choice(CFG_BW_CHOICES)
        i, j = (a, b) if a < b else (b, a)
        edges.append((i, j, bw))
        connected.add(b)
        remaining.remove(b)

    # 2) Add some extra random edges.
    edge_set = {(i, j) for i, j, _ in edges}
    for i in range(n):
        for j in range(i + 1, n):
            if (i, j) in edge_set:
                continue
            if random.random() < extra_edge_prob:
                edges.append((i, j, random.choice(CFG_BW_CHOICES)))
                edge_set.add((i, j))

    return edges


def _edges_to_connection_matrix(n: int, edges: List[Tuple[int, int, int]]) -> List[List[int]]:
    mat = [[0 for _ in range(n)] for _ in range(n)]
    for i, j, bw in edges:
        mat[i][j] = int(bw)
        mat[j][i] = int(bw)
    return mat


def _connection_matrix_to_uve(connections_matrix: List[List[int]]) -> Tuple[List[int], List[int], List[int]]:
    """
    Convert symmetric adjacency matrix to u/v/e lists (directed entries for every non-zero matrix cell).
    This follows the format seen in cpu-dram.cfg.
    """
    n = len(connections_matrix)
    u: List[int] = []
    v: List[int] = []
    e: List[int] = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            w = int(connections_matrix[i][j])
            if w > 0:
                u.append(i)
                v.append(j)
                e.append(w)
    return u, v, e


def _format_cfg_like_cpu_dram(
    chiplet_count: int,
    widths: List[float],
    heights: List[float],
    powers: List[float],
    connections_matrix: List[List[int]],
    path_value: str = "outputs/system/",
    target_reward: int = 0,
) -> str:
    """
    Format a cfg file strictly following /root/placement/thermal-placement/benchmark/config/cpu-dram.cfg style.
    """
    def fmt_list(nums: List[float]) -> str:
        return ",".join(str(float(x)).rstrip("0").rstrip(".") if float(x).is_integer() else str(float(x)) for x in nums)

    def fmt_int_list(nums: List[int]) -> str:
        return ", ".join(str(int(x)) for x in nums)

    lines: List[str] = []
    lines.append("[general]")
    lines.append("path = " + path_value)
    lines.append("")
    lines.append("[chiplets]")
    lines.append(f"chiplet_count = {chiplet_count}")
    lines.append("widths = \t" + ",\t".join(f"{w:g}" for w in widths))
    lines.append("heights = \t" + ",\t".join(f"{h:g}" for h in heights))
    lines.append("powers = \t" + ",\t".join(f"{p:g}" for p in powers))
    lines.append(f"target_reward = {int(target_reward)}")

    # connections matrix: each row ends with ';' except last row (matches cpu-dram.cfg)
    lines.append("connections = " + ",\t".join(str(int(x)) for x in connections_matrix[0]) + ";")
    for r in range(1, chiplet_count - 1):
        lines.append("\t\t\t" + ",\t".join(str(int(x)) for x in connections_matrix[r]) + ";")
    if chiplet_count > 1:
        lines.append("\t\t\t" + ",\t".join(str(int(x)) for x in connections_matrix[chiplet_count - 1]))
    else:
        # n==1 edge case
        lines[-1] = "connections = 0"

    u, v, e = _connection_matrix_to_uve(connections_matrix)
    lines.append(f"u =  {fmt_int_list(u)}")
    lines.append(f"v =  {fmt_int_list(v)}")
    lines.append(f"e =  {fmt_int_list(e)}")
    lines.append("x = " + ",".join("0" for _ in range(chiplet_count)))
    lines.append("y = " + ",".join("0" for _ in range(chiplet_count)))
    lines.append("")
    return "\n".join(lines)


def _next_system_cfg_path(config_dir: Path) -> Path:
    config_dir.mkdir(parents=True, exist_ok=True)
    existing = list(config_dir.glob("system_*.cfg"))
    used = set()
    for p in existing:
        m = re.match(r"system_(\d+)\.cfg$", p.name)
        if m:
            used.add(int(m.group(1)))
    i = 0
    while i in used:
        i += 1
    return config_dir / f"system_{i}.cfg"


def generate_random_connected_cfg(
    config_dir: Path,
    chiplet_count_range: Tuple[int, int] = (3, 20),
    width_range: Tuple[int, int] = (3, 30),
    height_range: Tuple[int, int] = (3, 30),
    ar_min: float = 0.8,
    ar_max: float = 1.25,
    extra_edge_prob: float = 0.25,
) -> Path:
    """
    Generate a random connected system cfg and save as ./config/system_i.cfg (auto-increment).
    """
    n = random.randint(chiplet_count_range[0], chiplet_count_range[1])
    widths, heights = _random_chiplet_dims(n, width_range, height_range, ar_min, ar_max)

    # cpu-dram.cfg requires powers; not specified by user, so we generate a reasonable random power per chiplet.
    powers = [float(random.randint(1, 200)) for _ in range(n)]

    edges = _generate_connected_graph_edges(n, extra_edge_prob=extra_edge_prob)
    conn_mat = _edges_to_connection_matrix(n, edges)

    cfg_text = _format_cfg_like_cpu_dram(
        chiplet_count=n,
        widths=widths,
        heights=heights,
        powers=powers,
        connections_matrix=conn_mat,
        path_value="outputs/system/",
        target_reward=0,
    )

    out_path = _next_system_cfg_path(config_dir)
    out_path.write_text(cfg_text, encoding="utf-8")
    return out_path


def main():
    """Main function to process all .cfg files."""
    import argparse
    script_dir = Path(__file__).parent.resolve()
    parser = argparse.ArgumentParser(
        description="Convert .cfg files to JSON, or generate random connected .cfg files."
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        default="config",
        help=".cfg 文件所在目录（相对本脚本所在目录，默认: config）",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="test_input",
        help="输出 JSON 的目录（相对本脚本所在目录，默认: test_input）",
    )
    parser.add_argument(
        "--generate-random-cfg",
        action="store_true",
        help="Generate one random connected system cfg and save to ./config/system_i.cfg (auto-numbered).",
    )
    parser.add_argument(
        "--generate-count",
        type=int,
        default=1,
        help="How many random .cfg files to generate (used with --generate-random-cfg).",
    )
    parser.add_argument(
        "--generate-out-dir",
        type=str,
        default="config",
        help="Output directory for generated .cfg files (relative to this script dir).",
    )
    args = parser.parse_args()
    
    config_dir = (script_dir / args.config_dir).resolve()
    output_dir = (script_dir / args.output_dir).resolve()

    if args.generate_random_cfg:
        out_dir = (script_dir / args.generate_out_dir)
        count = int(args.generate_count)
        if count < 1:
            raise SystemExit("--generate-count must be >= 1")
        last_path = None
        for _ in range(count):
            last_path = generate_random_connected_cfg(config_dir=out_dir)
        if last_path is not None:
            print(f"Generated {count} cfg file(s). Last: {last_path}")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    cfg_files = list(Path(config_dir).glob('*.cfg'))
    
    if not cfg_files:
        print(f"No .cfg files found in {config_dir}")
        return
    
    print(f"Found {len(cfg_files)} .cfg file(s) to process")
    print(f"EMIBType 默认: interfaceC\n")
    
    success_count = 0
    error_count = 0
    
    for cfg_path in sorted(cfg_files):
        try:
            cfg_to_json(str(cfg_path), output_dir)
            success_count += 1
        except Exception as e:
            print(f"ERROR processing {cfg_path}: {e}")
            error_count += 1
        print()
    
    # Summary
    print("=" * 60)
    print(f"Processing complete!")
    print(f"  Success: {success_count}")
    print(f"  Errors: {error_count}")
    print(f"  Output directory: {output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    # python input_preprocess.py --generate-random-cfg --generate-count 8000 --generate-out-dir config
    main()
