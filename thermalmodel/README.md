# Thermal Guidance Model (`guidance_model.py`)

本文档描述 `/root/placement/flow_GCN/thermalmodel/guidance_model.py` 当前版本的 **输入/输出**、**网络结构**、**损失函数**、**训练与测试流程** 以及 **checkpoint 续训机制**。

> 目标任务：训练一个快速 surrogate model，从 **power map + layout mask** 预测 **64×64 温度场**，用于 placement / diffusion guidance。

---

## 1) 数据与归一化（ThermalDataset）

数据加载在 `thermalmodel/dataLoader.py`。

- `power` 与 `temp` 都做 **min-max 归一化到 [0,1]**（使用全数据集标量 min/max）：
  - `p01 = (p - power_min) / (power_max - power_min)`
  - `t01 = (t - temp_min) / (temp_max - temp_min)`
- 因此 `ThermalDataset.__getitem__()` 返回的：
  - `batch["power"]`：归一化后 `(B,1,128,128)`
  - `batch["layout"]`：mask `(B,1,128,128)`（0/1）
  - `batch["temp"]`：归一化后 `(B,1,64,64)`

关键实现：`thermalmodel/dataLoader.py:214-220`。

### 反归一化（回到 Kelvin）
若要把网络输出/GT 从 [0,1] 还原到 Kelvin（K）：

```python
T = t01 * (temp_max - temp_min) + temp_min
```

在当前 `guidance_model.py` 的 `test` 中：
- **反归一化使用 `dataset.stats.temp_min/temp_max`**（与数据加载一致）。

---

## 2) 模型输入/输出

### 输入
模型采用 CoordConv 思路，最终输入通道为 4：

- `power_grid`: `(B,1,128,128)`
- `layout_mask`: `(B,1,128,128)`
- `x_coord_map`: `(B,1,128,128)`，线性分布在 [-1,1]
- `y_coord_map`: `(B,1,128,128)`，线性分布在 [-1,1]

拼接后：

```text
X = concat([power_grid, layout_mask, xmap, ymap], dim=1)
X.shape = (B,4,128,128)
```

### 输出
- `T_pred`: `(B,1,64,64)`
- 训练时输出与标签 `temp` 在同一尺度（归一化 [0,1]）。
- 测试/可视化时可以反归一化到 K。

---

## 3) 网络结构：ThermalGuidanceNet（当前版本）

当前 `ThermalGuidanceNet` 是对称 U-Net 风格结构，核心目标是增强热扩散任务所需的全局感受野。

### 关键模块

- `ConvGNAct`: Conv2d + GroupNorm + GELU
- `LargeKernelBlock`: 7×7 depthwise conv 的残差块（扩大感受野，用于扩散/全局耦合）
- `ASPP`: Atrous Spatial Pyramid Pooling（dilation=2/4/6）捕捉多尺度上下文

### 分辨率路径（对齐 64×64 目标网格）

- **Stem**：`128×128 → 64×64`（stride=2），从一开始就对齐温度标签分辨率
- **Encoder**：`64→32→16→8`
- **Bottleneck**：LargeKernel + ASPP + LargeKernel
- **Decoder**：`8→16→32→64`，与 encoder 同尺度 skip concat 融合
- **Head**：输出 1 通道 `64×64`

---

## 4) 损失函数：guidance_loss

用于回归温度场，并同时强调热点与物理平滑性。

组成：

1. **Hotspot-aware weighted L1**（主损失）
2. **Gradient loss（Sobel）**：约束一阶梯度，使热点边界更准确
3. **Laplacian loss（physics-informed）**：约束二阶导（拉普拉斯），贴近热传导/泊松方程先验

总损失：

```text
loss = weighted_l1 + grad_w * grad_loss + 0.05 * laplacian_loss
```

训练时会打印/记录：`l1, mse(监控), grad, lap, loss`。

---

## 5) 训练（train）

- Optimizer：`AdamW(lr=args.lr, weight_decay=1e-4)`
- Scheduler：`CosineAnnealingLR(T_max=args.epochs)`

### 从头训练示例（100 epochs）

```bash
python /root/placement/flow_GCN/thermalmodel/guidance_model.py train \
  --epochs 100 --batch_size 8 --lr 5e-4 --base 64 --grad_w 0.1 \
  --ckpt_every 10 --print_every 10
```

---

## 6) Checkpoint 设计与续训练（resume）

训练保存的 `.pth` checkpoint（在 `thermalmodel/checkpoints/`）包含：

- `epoch`: 当前 epoch
- `model`: `state_dict`
- `opt`: optimizer state
- `scheduler`: scheduler state（**必须有，用于严格续训**）
- `stats`: 数据集 min/max（用于 test 阶段反归一化到 K）
- `train_args`: 记录训练参数（含 epochs/lr/base/grad_w 等）

### 严格续训
当前代码 **不兼容缺少 `scheduler` 字段的旧 checkpoint**：
- 若 `--resume_ckpt` 指向的 `.pth` 没有 `scheduler`，会直接报错。

续训示例：

```bash
python /root/placement/flow_GCN/thermalmodel/guidance_model.py train \
  --resume_ckpt /root/placement/flow_GCN/thermalmodel/checkpoints/xxx.pth \
  --epochs 200 --batch_size 8 --lr 5e-4 --base 64 --grad_w 0.1
```

---

## 7) 测试与绘图（test）

测试流程：

1. 推理得到 `pred_norm`（[0,1]）与 `gt_norm`（[0,1]）
2. 使用 `dataset.stats.temp_min/temp_max` 反归一化得到 `pred_K/gt_K`
3. 在 **K 尺度**上计算并打印：
   - `max_rmse / min_rmse / mean_rmse`
   - `max_ae`
   - `mean_rmspe`（%）
4. 选取 RMSE 最差样本，绘制 pred/gt 热图（单位 K）

示例：

```bash
python /root/placement/flow_GCN/thermalmodel/guidance_model.py test \
  --ckpt /root/placement/flow_GCN/thermalmodel/checkpoints/xxx.pth \
  --batch_size 8 \
  --out_fig_dir /root/placement/flow_GCN/thermalmodel/test_result/test_guidance_fig
```

绘图函数：`thermalmodel/draw_thermal_fig.py:plot_thermal_grid_overlay()`。
