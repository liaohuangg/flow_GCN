import os
import shutil
import re
import glob

# Paths relative to this script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))
src_base = os.path.join(script_dir, "../output/thermal/hotspot_config")
dst_dir = os.path.join(script_dir, "../output/thermal/thermal_map/csv")

os.makedirs(dst_dir, exist_ok=True)

# Find all system_i_config/data directories
pattern = os.path.join(src_base, "system_*_config", "data")
data_dirs = glob.glob(pattern)

for data_dir in sorted(data_dirs):
    # Extract system index i from path like .../system_123_config/data
    config_name = os.path.basename(os.path.dirname(data_dir))  # system_123_config
    match = re.search(r"system_(\d+)_config", config_name)
    if not match:
        continue
    idx = match.group(1)

    # Copy and rename each .csv file: Edge.csv -> Edge_123.csv
    for csv_file in glob.glob(os.path.join(data_dir, "*.csv")):
        base_name = os.path.splitext(os.path.basename(csv_file))[0]  # e.g. "Edge"
        new_name = f"{base_name}_{idx}.csv"
        dst_path = os.path.join(dst_dir, new_name)
        shutil.copy2(csv_file, dst_path)

print(f"Done. Copied CSV files from {len(data_dirs)} systems to {os.path.relpath(dst_dir, script_dir)}")
