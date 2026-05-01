# thermalmodel (Thermal Guidance Model)

这个目录主要包含 **ThermalGuidanceNet** 的训练 / 评测 / 消融与量化相关脚本。

> 运行方式：这些脚本大多允许你在 `thermalmodel/` 目录内直接运行（代码会把项目根目录加入 `sys.path`）。

## 1. 模型结构：`ThermalGuidanceNet`（`guidance_model.py`）

核心模型定义在：`thermalmodel/guidance_model.py:107`。

**输入/输出张量约定（见 `ThermalGuidanceNet` docstring 与 forward）：**

- 输入
  - `power_grid`: `(B, 1, 128, 128)`，来自 `ThermalDataset` 的 power map（min-max 归一化到 [0,1]）
  - `layout_mask`: `(B, 1, 128, 128)`，从 `system.flp` 解析得到的 0/1 布局 mask（chiplet 区域为 1）
  - `total_power`: `(B, 1)`，每个 case 的总功耗标量（数据集内已做 scaling+minmax）
- 输出
  - `temp_grid`: `(B, 1, 64, 64)`，预测温度场（训练/评测时通常是归一化值；评测可反归一化回 Kelvin）
  - `avg_temp`: `(B, 1)`，预测的平均温度标量（同样是归一化值；用于辅助 loss 以及推理时做全局均值校准）

**网络结构（高层概览）：**

- 输入通道拼接：`[power, layout, x_coord, y_coord]` 共 4 通道（`guidance_model.py:199-205`）
- 编码器：多层轻量 inverted residual block（MobileNet 风格）逐步下采样 128→64→32→16（`d1/d2/d3`）
- bottleneck：在最低分辨率处提特征（`bottleneck`）
- 条件注入：用 **FiLM** 把 `total_power` 注入到 bottleneck 特征（`guidance_model.py:162-222`）
- 解码器：双线性上采样 + skip concat + conv 融合，回到 64×64（`guidance_model.py:227-237`）
- 平均温度 head：对 bottleneck 做全局池化 + MLP 输出 `avg_temp`（`guidance_model.py:175-226`）
- 全局校准：最后把 `temp_grid` 整体平移，使 `mean(temp_grid)` 与 `avg_temp` 一致（`guidance_model.py:241-245`）

> 备注：模型里包含 `QuantStub/DeQuantStub` 与 QAT 相关配置，用于可选的量化训练流程；FP32 训练时不需要开启。

## 2. 数据集与 HotSpot 仿真数据（ground truth）

数据读取由 `thermalmodel/dataLoader.py:ThermalDataset` 完成。

默认数据目录（相对项目根目录）：

- thermal map 数据：`Dataset/dataset/output/thermal/thermal_map/`
  - `powercsv/system_power_{i}_{j}.csv`
  - `tempcsv/system_temp_{i}_{j}.csv`  ← **HotSpot 热仿真温度场（ground truth）**
  - `totalpowercsv/system_totalpower_{i}_{j}.csv`
  - `avgtempcsv/system_avgtemp_{i}_{j}.csv`

- floorplan：`Dataset/dataset/output/thermal/hotspot_config/system_{i}_config/system.flp`

`ThermalDataset.__getitem__` 会返回 `{i, j, power, layout, temp, total_power, avg_temp}`（见 `dataLoader.py:219-256`），其中 `temp` 来源就是 `tempcsv/system_temp_{i}_{j}.csv`。

训练保存 checkpoint 时，会把数据集的 min/max（尤其是 `temp_min/temp_max`）写入 `ckpt["stats"]`，用于评测时把归一化温度反归一化回 **Kelvin**。

## 3. FP32 训练方式（`guidance_model.py train`）

训练入口在：`thermalmodel/guidance_model.py:337 main_train()`，命令行子命令是：`guidance_model.py train`（见 `guidance_model.py:732+`）。

### 3.1 训练/验证划分

- `ThermalDataset(...)` 全量样本
- `torch.utils.data.random_split` 按 **90% train / 10% val** 划分
- split 的随机性由 `--seed` 控制（`guidance_model.py:348-356`）

### 3.2 Loss（监督学习目标）

loss 在 `guidance_model.py:268 guidance_loss()`：

- per-pixel MSE（并对热点区域做加权，强调 hotspot 部分）
- + `grad_w * spatial_gradient_loss`（Sobel 梯度一致性）
- + 可选的 `avg_w * MSE(avg_temp)`（平均温度 head 的监督）
- + 可选的 mean consistency（鼓励 `mean(temp_grid)` 与 `avg_temp` 一致）

### 3.3 FP32 训练命令示例

在 `thermalmodel/` 目录下：

```bash
python guidance_model.py train \
  --epochs 200 --batch_size 8 --lr 2e-4 --base 96 \
  --grad_w 0.1 --avg_w 0.1 --mean_consistency_w 0.1 \
  --ckpt_every 5 --print_every 10 \
  --seed 0 \
  --ckpt_tag b96_lr2e-4_totalp_avg_resume \
  --out_dir checkpoints/guidance_b96_lr2e-4_ep200_20260430_resume
```

> 如果要混合精度训练（AMP）可以加：`--amp --amp_dtype fp16|bf16`（见 `guidance_model.py:420+`）。

## 4. FP32 模型评估（与 HotSpot 仿真对比）

评估脚本推荐使用：`thermalmodel/eval_guidance_ckpt.py`。

它会：
- 用与训练同样的 `ThermalDataset` 和相同 90/10 split 逻辑生成 test set（`eval_guidance_ckpt.py:76-89`）
- 前向得到 `pred temp_grid`
- 若 ckpt 内含 `stats(temp_min/temp_max)`，则将 pred 与 gt 从 [0,1] 反归一化回 **Kelvin**（`eval_guidance_ckpt.py:124-129`）
- 计算 per-sample RMSE 等指标，并找 best/worst case（或 `--topk K` 时输出 top-K best/worst）

### 4.1 指标（单位：K）

`eval_guidance_ckpt.py` 输出的关键指标：
- RMSE（每个样本在全像素上的 RMSE）
- max absolute error
- RMSPE (%)
- mean_grad（Sobel 梯度差）

这些指标的 **ground truth** 直接来自 HotSpot 仿真输出 `tempcsv/system_temp_{i}_{j}.csv`。

### 4.2 评估命令示例

```bash
python eval_guidance_ckpt.py \
  --ckpt /path/to/fp32_ckpt.pth \
  --eval_bs 8 --seed 0 --device cuda
```

如果要导出 best/worst（或 top-k）预测 vs 仿真图：

```bash
python eval_guidance_ckpt.py \
  --ckpt /path/to/fp32_ckpt.pth \
  --eval_bs 8 --seed 0 --device cuda \
  --out_fig_dir /root/placement/flow_GCN/thermalmodel/test_result/fig \
  --topk 50
```

导出的图片文件名会包含 case 编号 `(i,j)`，并区分：
- `*_pred_*.png`：模型预测温度图
- `*_sim.png`：HotSpot 热仿真温度图（ground truth）

每张图里都会标注 Max / AVG 温度数值。

## 目录内容速览

- `guidance_model.py`
  - **核心**：模型结构 `ThermalGuidanceNet` + CLI（`train`/`test`）
  - 训练会保存 checkpoint（`.pth`），并在 `test` 时可输出 worst-case 图

- `dataLoader.py`
  - `ThermalDataset`：从 `Dataset/dataset/output/thermal/...` 读取 power/temp/totalpower/avgtemp
  - 同时从 `hotspot_config/.../system.flp` 解析布局，生成 layout mask
  - 做 min-max 归一化，并在 ckpt 内保存 `stats` 便于评测时反归一化回 **Kelvin**

- `eval_guidance_ckpt.py`
  - 对单个 checkpoint 做评测（RMSE / grad / max AE / RMSPE 等），并可导出 best/worst（或 top-k）样本图
  - 可选 `--bench` 做纯模型前向 latency benchmark

- `ablation_eval.py`
  - 给定 fp32/qat checkpoint 目录：
    1) 遍历评测并按 `mean_rmse` 选 best fp32 与 best qat
    2) 从 best fp32 导出 **PTQ-fp16**（权重 half）
    3) 对三种变体（fp32/qat/ptq-fp16）评测 + benchmark
    4) 可选写出 JSON 汇总

- `guidance_pipeline.py`
  - 一个“管道式”包装：提供 `train_fp32` / `train_qat` / `export_ptq` 子命令
  - 其中训练子命令内部是 `os.system(...)` 去调用 `guidance_model.py train ...`
  - `ablation_eval.py` 会直接 import 这里的 `export_fp16_from_fp32()`

- `draw_thermal_fig.py`
  - `plot_thermal_grid_overlay()`：把 2D thermal grid 画成图，并叠加 `system.flp` 的 chiplet 矩形框

- `auto_train_guidance.sh`
  - 一键：fp32 训练 + QAT 训练 + ablation（选 best + 导出 fp16 + benchmark）

- `adjust_parameter.sh`
  - fp32 超参 sweep：训练多组 (BASE, LR, GRAD_W)，并对每个保存的 ckpt（每 5 epoch）做评测、写日志和 worst-case 图

- `checkpoints/`
  - 保存训练产物（按脚本可能在 `checkpoints/fp32`、`checkpoints/qat`、`checkpoints/ptq`、`checkpoints/fp32_sweep/...` 等）

- `eval_logs/` / `logs/` / `worst_figs/`
  - 评测/训练日志与 worst-case 可视化输出（具体由脚本决定输出目录）

## 文件调用关系（依赖 / 调用链）

下面是“谁调用谁”的主链路（从入口脚本到核心库）：

```
auto_train_guidance.sh
  -> guidance_model.py train (fp32)
  -> guidance_model.py train --qat
  -> ablation_eval.py
       -> guidance_model.ThermalGuidanceNet
       -> dataLoader.ThermalDataset
       -> guidance_pipeline.export_fp16_from_fp32

adjust_parameter.sh
  -> guidance_model.py train
  -> eval_guidance_ckpt.py
       -> guidance_model.ThermalGuidanceNet
       -> dataLoader.ThermalDataset
       -> draw_thermal_fig.plot_thermal_grid_overlay

(guidance_model.py test)
  -> draw_thermal_fig.plot_thermal_grid_overlay
  -> dataLoader.ThermalDataset
```

更细的 Python import 依赖：

- `guidance_model.py` 依赖：`dataLoader.py`、`draw_thermal_fig.py`
- `eval_guidance_ckpt.py` 依赖：`dataLoader.py`、`guidance_model.py`、`draw_thermal_fig.py`
- `ablation_eval.py` 依赖：`dataLoader.py`、`guidance_model.py`、`guidance_pipeline.py`
- `guidance_pipeline.py` 依赖：`guidance_model.py`（通过命令行调用）

## 数据目录约定（非常重要）

`ThermalDataset` 默认读取（相对项目根目录，即 `thermalmodel/..`）：

- thermal map 数据：`Dataset/dataset/output/thermal/thermal_map/`
  - `powercsv/system_power_{i}_{j}.csv`
  - `tempcsv/system_temp_{i}_{j}.csv`
  - `totalpowercsv/system_totalpower_{i}_{j}.csv`
  - `avgtempcsv/system_avgtemp_{i}_{j}.csv`

- floorplan：`Dataset/dataset/output/thermal/hotspot_config/system_{i}_config/system.flp`

如果你数据不在这个位置，需要修改 `dataLoader.py:ThermalDataset(...)` 里的 `thermal_map_rel/hotspot_cfg_rel` 参数。

## 常用使用方式

下面命令都以在 `thermalmodel/` 目录下执行为例。

### 1) 一键训练 + QAT + 消融（推荐）

```bash
./auto_train_guidance.sh
```

常用：跳过 fp32（如果已经有 fp32 ckpt）：

```bash
./auto_train_guidance.sh --no-fp32
```

脚本默认用 `conda run -n chipdiffusion python -u ...`。如果你环境名不同：

```bash
./auto_train_guidance.sh --conda-env <YOUR_ENV>
```

### 2) 直接训练（Python CLI）

FP32：

```bash
python guidance_model.py train \
  --epochs 200 --batch_size 8 --lr 5e-4 --base 64 --grad_w 0.1 \
  --ckpt_every 5 --print_every 10 \
  --ckpt_tag fp32 --out_dir checkpoints/fp32
```

QAT：

```bash
python guidance_model.py train \
  --epochs 200 --batch_size 8 --lr 5e-4 --base 64 --grad_w 0.1 \
  --ckpt_every 5 --print_every 10 \
  --qat --ckpt_tag qat --out_dir checkpoints/qat
```

从 checkpoint 继续训练：

```bash
python guidance_model.py train --resume_ckpt <path/to/ckpt.pth --epochs 200 ...
```

（可选）CUDA AMP：

```bash
python guidance_model.py train --amp --amp_dtype fp16 ...
```

### 3) 评测单个 checkpoint

```bash
python eval_guidance_ckpt.py \
  --ckpt <path/to/ckpt.pth> \
  --eval_bs 8 --seed 0 --device cuda \
  --out_fig_dir worst_figs/single_ckpt
```

带 benchmark：

```bash
python eval_guidance_ckpt.py --ckpt <ckpt.pth> --bench --bench_bs 8 --device cuda
```

### 4) 消融评测（选 best + 导出 fp16 + benchmark + JSON）

```bash
python ablation_eval.py \
  --fp32_dir checkpoints/fp32 \
  --qat_dir checkpoints/qat \
  --ptq_dir checkpoints/ptq \
  --eval_bs 8 --bench_bs 8 --device cuda \
  --out_json logs/ablation.json
```

### 5) 超参 sweep（fp32）

```bash
./adjust_parameter.sh
```

可用环境变量覆盖默认值（见脚本顶部的 defaults）：

```bash
CONDA_ENV=chipdiffusion EPOCHS=200 BS=8 SEED=0 ./adjust_parameter.sh
```

## Checkpoint 命名/字段说明（训练输出）

`guidance_model.py` 保存的 ckpt（`.pth`）通常包含：

- `model`: `state_dict`
- `opt`: optimizer state（用于 resume）
- `epoch`
- `stats`: 数据 min/max（用于把预测从 norm 反归一化回 Kelvin）
- 训练超参：`base/batch_size/lr/grad_w/seed/ckpt_tag` 等

文件名形如：

```
guidance_net_{ckpt_tag}_epXXXX_seed{seed}_bs{bs}_lr{lr_tok}_base{base}_gw{gw_tok}.pth
```

## 备注

- `guidance_pipeline.py` 的 `train_fp32/train_qat` 是包装命令（内部 `os.system` 调用），功能上与直接运行 `guidance_model.py train ...` 等价。
- `eval_guidance_ckpt.py` 的日志输出是单行 `metrics ...`，适合用脚本批量 grep/汇总。

## 训练参数记录（resume from 80 -> 200, seed0）

对应日志：`eval_logs/train_resume_20260430_to200_from80_seed0.log`

- resume_ckpt:
  `/root/placement/flow_GCN/thermalmodel/checkpoints/guidance_b96_lr2e-4_ep200_20260430_resume/guidance_net_b96_lr2e-4_totalp_avg_resume_ep0080_seed0_bs8_lr2e-04_base96_gw0p1.pth`
- start_epoch: 81
- target_epochs: 200
- device: cuda
- batch_size: 8
- seed: 0
- amp: False（这次训练不是 fp16 AMP 训练）
- ckpt_tag: `b96_lr2e-4_totalp_avg_resume`
- out_dir:
  `/root/placement/flow_GCN/thermalmodel/checkpoints/guidance_b96_lr2e-4_ep200_20260430_resume`

从 ckpt 文件名与保存目录可读到的关键超参：
- base: 96
- lr: 2e-4
- grad_w: 0.1

如果你要做“训练时 fp16（AMP）”的同参数训练：
- 在 `guidance_model.py train ...` 的同一组参数上额外加：`--amp --amp_dtype fp16`
