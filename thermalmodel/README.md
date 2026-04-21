# Thermal Guidance Model (`guidance_model.py`)

This document describes the **inputs**, **model structure**, **outputs**, and the **design rationale** of `/root/placement/flow_GCN/thermalmodel/guidance_model.py`.

> Target task: learn a fast surrogate model that predicts a **64×64 temperature map** from a **power map** and a **layout mask**, so it can be used as a differentiable (or near-differentiable) *thermal guidance* signal in placement / diffusion-style optimization.

---

## 1. What the model predicts (Output)

**Output tensor:** `T_pred` with shape **(B, 1, 64, 64)**.

- The network internally works on a 128×128 grid, then downsamples to 64×64.
- In code:
  - `ThermalGuidanceNet.forward(...)` produces a 128×128 map via `self.head`.
  - Then it applies `avg_pool2d(kernel_size=2, stride=2)` to get 64×64.

**Why 64×64?**
- The dataset target temperature grid is 64×64 (`temp_grid_size=64` in `ThermalDataset`).
- Predicting at the same resolution as the ground truth avoids unnecessary upsampling artifacts and simplifies loss computation.

**Units:**
- In training, `temp` is typically **normalized** by the dataset (see dataset stats in checkpoints).
- At test time, if the checkpoint includes `stats = {temp_min, temp_max}`, the script can de-normalize back to **Kelvin** for visualization.

---

## 2. Model inputs (What goes into the network)

The model is a CoordConv-style CNN that consumes **four input channels** at **128×128** resolution:

### 2.1 `power_grid`
- **Shape:** (B, 1, 128, 128)
- **Meaning:** spatial distribution of power density / power mapped to a grid.

### 2.2 `layout_mask`
- **Shape:** (B, 1, 128, 128)
- **Meaning:** layout occupancy mask (usually 0/1), derived from floorplan (.flp) / chiplet placement.

### 2.3 `x_coord_map` and `y_coord_map`
- **Shape:** each is (B, 1, 128, 128)
- **Values:** linearly spaced in **[-1, 1]**
  - `x_coord_map`: constant across rows, increases left → right
  - `y_coord_map`: constant across cols, increases top → bottom

This makes the final input:

```
X = concat([power_grid, layout_mask, x_coord_map, y_coord_map], dim=1)
X.shape = (B, 4, 128, 128)
```

---

## 3. Architecture overview (Structure)

The network (`ThermalGuidanceNet`) is a lightweight encoder–decoder with skip connections (U-Net-like), built from **MobileNet-style inverted residual blocks** and **GroupNorm**:

### 3.1 Key design goals
- **Lightweight**: thermal inference should be fast for iterative guidance.
- **Stable on small batches**: GroupNorm avoids BatchNorm instability.
- **Quantization-friendly**: structure avoids BatchNorm and uses simple conv + pointwise + depthwise ops.

### 3.2 Blocks used

#### (A) `ConvGNAct`
- `Conv2d(bias=False) → GroupNorm → SiLU`

#### (B) `LiteInvertedResidual`
MobileNetV3-style block:
- optional expand (1×1)
- depthwise 3×3
- pointwise 1×1
- GroupNorm + SiLU
- residual connection if shape matches

### 3.3 Encoder–decoder stages

Input: (B, 4, 128, 128)

- **Stem:** 3×3 conv → (B, base, 128, 128)

**Encoder:**
- `e1` keeps 128×128
- `d1` downsamples 128→64 (channels base→2base)
- `e2` keeps 64×64
- `d2` downsamples 64→32 (2base→4base)
- `e3` keeps 32×32
- `d3` downsamples 32→16 (4base→8base)

**Bottleneck:**
- 16×16 at 8base channels

**Decoder:**
- nearest upsample (×2), reduce channels, concat skip, fuse conv
- 16→32 (concat with encoder’s 32×32)
- 32→64 (concat with encoder’s 64×64)
- 64→128 (concat with encoder’s 128×128)

**Head:** 1×1 conv → (B, 1, 128, 128)

Then **avg-pool** to 64×64 output.

---

## 4. Conditioning on total power (FiLM)

Besides the spatial inputs, the model also takes `total_power`:

- **Shape:** (B, 1)
- **Usage:** FiLM modulation on bottleneck features.

Mechanism:
- A small MLP maps `total_power` → `(gamma, beta)` for each bottleneck channel.
- Modulation: `z = z * (1 + gamma) + beta`

**Why this helps for this task**
- Thermal fields are strongly influenced by **global power scaling** and package-level boundary conditions.
- Two layouts with similar spatial patterns but different total power may have similar *shape* but different absolute magnitude.
- FiLM gives the network an explicit “global knob” to adjust feature magnitudes based on total power, improving generalization.

---

## 5. Why include coordinate maps? (Task adaptation)

A plain CNN is translation-equivariant: it tends to treat the same local pattern similarly everywhere.

But thermal behavior on a chip is **not** perfectly translation-invariant:
- Edges / corners can have different cooling behavior (boundary effects).
- The same power blob near an edge may create a different hotspot than the same blob in the center.

By injecting `(x, y)` coordinate channels, the network can learn **position-dependent responses** (boundary sensitivity) while staying fully convolutional.

---

## 6. Loss function (Why it fits thermal maps)

Training uses:

- **MSE** between predicted temperature and target temperature.
- plus **spatial gradient loss** (Sobel-based) with weight `grad_w`.

Why the gradient term helps:
- Thermal maps are smooth fields but important for placement is often the **shape and location of hotspots / gradients**.
- A pure MSE can over-smooth peaks or misplace edges; matching spatial gradients encourages correct hotspot boundaries and spatial structure.

---

## 7. Quantization support (QAT) and why it matters

The model includes `QuantStub` and `DeQuantStub` and can optionally run **FX graph-mode QAT** (`--qat`).

Why this is beneficial here:
- Guidance is often called many times during sampling/optimization.
- INT8 inference can reduce CPU latency and memory bandwidth.
- The architecture avoids BatchNorm and uses simple conv blocks, which is generally friendlier to quantization.

Notes:
- Without `--qat`, training is standard floating-point.
- With `--qat`, `prepare_qat_fx(...)` inserts fake-quant modules during training.
- At checkpoint time, `convert_fx(...)` can export an INT8-converted model state dict.

---

## 8. Summary: Why this design is a good fit

- **Inputs reflect physics & placement constraints**: power + occupancy mask + coordinates + total power.
- **U-Net-like multi-scale processing**: captures both local hotspots and global diffusion.
- **CoordConv + FiLM**: explicitly models boundary effects and global power scaling.
- **Gradient-aware loss**: preserves hotspot structure, not just per-pixel values.
- **QAT-ready**: supports fast deployment for iterative guidance loops.

---

## 9. Quick reference (I/O)

**Forward signature**

```python
pred = model(power_grid, layout_mask, total_power)
```

**Shapes**
- `power_grid`: (B, 1, 128, 128)
- `layout_mask`: (B, 1, 128, 128)
- `total_power`: (B, 1)
- `pred`: (B, 1, 64, 64)

