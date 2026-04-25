#!/usr/bin/env python3
"""
Convert .cfg files to JSON format for chiplet placement.
"""

import os
import re
import json
from pathlib import Path

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


def main():
    """Main function to process all .cfg files."""
    import argparse
    script_dir = Path(__file__).parent.resolve()
    parser = argparse.ArgumentParser(
        description="Convert .cfg files to JSON. connections 输出为 {node1, node2, wireCount, EMIBType} 格式，EMIBType 默认 interfaceC。"
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        default="../config",
        help=".cfg 文件所在目录（相对本脚本所在目录，默认: ../config）",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="input_test",
        help="输出 JSON 的目录（相对本脚本所在目录，默认: input_test）",
    )
    args = parser.parse_args()
    
    config_dir = (script_dir / args.config_dir).resolve()
    output_dir = (script_dir / args.output_dir).resolve()
    
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
            cfg_to_json(str(cfg_path), output_dir, default_emib_type="interfaceC")
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
    main()

#python3 input_preprocess.py --config-dir ../config --output-dir input_test