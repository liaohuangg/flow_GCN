# HotSpot Thermal Data Pipeline

This directory contains scripts and assets to generate a HotSpot-based thermal dataset from placement JSONs.

The pipeline supports **multiple power scenarios per layout**:
- For each layout directory `system_i_config/` (same floorplan), we generate multiple power traces `system_1.ptrace .. system_10.ptrace`.
- HotSpot is run once per ptrace, producing `Chiplet_{j}.grid.steady`.
- Downstream CSVs are named with `(i,j)` so `power_i_j` and `temp_i_j` **always match the same ptrace scenario**.

## Directory layout

Inputs (generated elsewhere):
- `../output/placement/system_i.json`
  - Placement and original chiplet powers (field `chiplets[*].power`).

Pipeline outputs:
- `../output/thermal/hotspot_config/system_i_config/`
  - `system.flp` (floorplan)
  - `system_sub.flp` (bbox helper floorplan)
  - `system.ptrace` (original powers from JSON)
  - `system_1.ptrace .. system_10.ptrace` (random powers, **guaranteed != JSON powers** per chiplet)
  - `hotspot.config` (generated from template and resized)
  - `Chiplet_{j}.grid.steady`, `Chiplet_{j}.steady` (HotSpot outputs; one per ptrace)
- `../output/thermal/thermal_map/powercsv/`
  - `system_power_{i}_{j}.csv` (grid power map; 2 columns: `grid_id,power`)
- `../output/thermal/thermal_map/totalpowercsv/`
  - `system_totalpower_{i}_{j}.csv` (avg power per grid cell; 2 columns)
- `../output/thermal/thermal_map/tempcsv/`
  - `system_temp_{i}_{j}.csv` (Layer-2 grid temperatures; 2 columns: `grid_id,temp`)

HotSpot assets in this directory:
- `HotSpot/hotspot` (HotSpot executable)
- `hotspot.config` (template)
- `Chiplet.lcf` (layer configuration for grid model)
- `example.materials` (materials database)

## One-command dataset generation

Use the wrapper script:

```bash
bash Dataset/dataset/hotspot/gen_thermal_data.sh START_ID END_ID
```

Example (generate for layouts 1..2):

```bash
bash Dataset/dataset/hotspot/gen_thermal_data.sh 1 2
```

## What `gen_thermal_data.sh` does

All paths are relative to `Dataset/dataset/hotspot/`.

### Step 1) Generate HotSpot input directories + multi-ptrace scenarios

Script: `gen_flp_trace.py`

```bash
python3 gen_flp_trace.py \
  --input_dir ../output/placement \
  --output_root ../output/thermal/hotspot_config \
  --start_id START_ID \
  --end_id END_ID
```

For each `system_i.json`, it creates `../output/thermal/hotspot_config/system_i_config/` and writes:
- `system.flp`
- `system_sub.flp`
- `system.ptrace` (powers from JSON)
- `system_1.ptrace .. system_10.ptrace`
  - each chiplet power is sampled uniformly from **1..200W**
  - **strong guarantee**: if a sampled value equals the original JSON power for that chiplet, it is **resampled** until different.

### Step 2) Generate grid power CSVs per (i,j)

Script: `gen_powercsv.py`

```bash
python3 gen_powercsv.py \
  --grid 64 \
  --startid START_ID \
  --endid END_ID
```

This reads `system.flp` and each `system_{j}.ptrace` (fallback: `system.ptrace` if no numbered ptrace exists) and outputs:
- `../output/thermal/thermal_map/powercsv/system_power_{i}_{j}.csv`
- `../output/thermal/thermal_map/totalpowercsv/system_totalpower_{i}_{j}.csv`

### Step 3) Run HotSpot per (i,j) and preserve outputs

Script: `run_hotspot.py`

```bash
python3 run_hotspot.py \
  --configs_root ../output/thermal/hotspot_config \
  --start_id START_ID \
  --end_id END_ID
```

Behavior:
- Ensures each `system_i_config/` has a `hotspot.config` generated from the template `Dataset/dataset/hotspot/hotspot.config`.
  - `-s_sink` and `-s_spreader` are set to `(Unit0_width_from_system_sub.flp + 0.002)`.
- For each `system_i_config/` it enumerates `system_*.ptrace` and runs HotSpot once per file.
- After each run, it renames outputs to avoid overwriting:
  - `Chiplet.grid.steady` → `Chiplet_{j}.grid.steady`
  - `Chiplet.steady` → `Chiplet_{j}.steady`

Optional plotting (disabled by default):
- `--plot` enables generating `../output/thermal/thermal_map/fig/system_{i}_thermal_{j}.png`.

### Step 4) Extract Layer-2 temperatures to temp CSVs

Script: `process_csvdata.py`

```bash
python3 process_csvdata.py
```

It scans `../output/thermal/hotspot_config/system_*_config/` and for each existing
`Chiplet_{j}.grid.steady` extracts `Layer 2` values and writes:
- `../output/thermal/thermal_map/tempcsv/system_temp_{i}_{j}.csv`

Notes:
- Missing steady files are skipped (the script does not fail the whole run).

## Output pairing rule (important)

For a fixed layout `i` and scenario index `j`:
- power map: `system_power_{i}_{j}.csv`
- temperature: `system_temp_{i}_{j}.csv`

They correspond to the **same power trace**:
- `system_{j}.ptrace` → `Chiplet_{j}.grid.steady` → `system_temp_{i}_{j}.csv`
- `system_{j}.ptrace` → `system_power_{i}_{j}.csv`
