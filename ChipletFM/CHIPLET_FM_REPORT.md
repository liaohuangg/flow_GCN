# Chiplet Flow Matching Integration Report

## 1. 背景与目标

本次工作的目标是把 `flow_GCN/gcn_thermal` 中已经准备好的 chiplet 布局数据，接入 `chipdiffusion` 的代码结构，并使用其中已经存在的 `FlowMatchingModel` 完成：

1. 数据集格式转换
2. Flow Matching 训练配置接入
3. 最小训练链路验证
4. 最小 eval 推理验证
5. 为 `FlowMatchingModel` 接入 legality guidance

本次工作聚焦于“先把流程打通”，即：

- 数据能被 `chipdiffusion` 正常读取
- `FlowMatchingModel` 能训练
- `eval.py` 能生成样本和指标
- legality guidance 在 FM 采样时真正生效


## 2. 本次修改了哪些代码

### 2.1 新增数据转换脚本

文件：
[scripts/convert_flow_gcn_dataset.py](/Users/mac/WORK/duffusion/chipdiffusion/scripts/convert_flow_gcn_dataset.py:1)

作用：

- 读取 `flow_GCN/gcn_thermal/dataset/input_test/*.json`
- 读取 `flow_GCN/gcn_thermal/dataset/output/placement/*.json`
- 做输入输出配对
- 转成 `chipdiffusion` 训练所需的数据目录格式：
  - `graph{i}.pickle`
  - `output{i}.pickle`
  - `config.yaml`
  - `index_map.json`

输出目录默认是：

- [datasets/graph/chiplet_flow_gcn](/Users/mac/WORK/duffusion/chipdiffusion/datasets/graph/chiplet_flow_gcn)

这个脚本构造的 `torch_geometric.data.Data` 里包含：

- `x`: 节点尺寸 `(width, height)`
- `edge_index`: 双向边
- `edge_attr`: 4 维 pin offset，占位为全零，表示 center-to-center 连接
- `is_ports`: 全 False
- `is_macros`: 全 True
- `chip_size`: 从真实 placement 推断得到的画布边界
- 额外保留了：
  - `node_power`
  - `edge_weight`
  - `edge_type`
  - `edge_emib_length`
  - `edge_emib_max_width`
  - `edge_emib_bump_width`
  - `system_id`


### 2.2 新增专用 FM 训练配置

文件：
[diffusion/configs/config_graph_chiplet_fm.yaml](/Users/mac/WORK/duffusion/chipdiffusion/diffusion/configs/config_graph_chiplet_fm.yaml:1)

作用：

- 为 chiplet 数据集提供单独的 Flow Matching 训练配置
- 默认使用：
  - `family: flow_matching`
  - `task: chiplet_flow_gcn`
  - `backbone: att_gnn`
  - `model/size: small`
  - `logger.wandb: false`

这份配置用于：

- 最小训练 smoke test
- 后续正式训练的基础模板


### 2.3 修改 FlowMatchingModel，使其支持 legality guidance

文件：
[diffusion/models.py](/Users/mac/WORK/duffusion/chipdiffusion/diffusion/models.py:1570)

改动内容：

#### 2.3.1 在 `FlowMatchingModel.__init__()` 中接入 guidance 相关参数

新增支持的参数包括：

- `legality_guidance_weight`
- `hpwl_guidance_weight`
- `grad_descent_steps`
- `grad_descent_rate`
- `alpha_init`
- `alpha_lr`
- `alpha_critical_factor`
- `legality_potential_target`
- `use_adam`
- `guidance_mode`
- `legality_softmax_factor_min`
- `legality_softmax_factor_max`
- `legality_softmax_critical_factor`

这些参数此前虽然在 eval 配置层可见，但对 `FlowMatchingModel` 本身并没有实际作用。

#### 2.3.2 在 `reverse_samples()` 中真正接入 guided sampling

此前 FM 的采样逻辑是：

```text
x_next = x - dt * velocity
```

现在变成：

```text
x_next = x - dt * velocity + guidance_force
```

其中 `guidance_force` 由 `guidance_mode` 决定：

- `sgd` 模式：调用 `reverse_guidance_force()`
- `opt` 模式：调用 `reverse_guidance_opt_force()`

#### 2.3.3 新增 FM 版本的 guidance 实现

新增方法：

- `reverse_guidance_force()`
- `reverse_guidance_opt_force()`
- `get_legality_softmax_factor()`
- `reset_guidance_state()`

设计原则：

- 复用 `ContinuousDiffusionModel` 的思想
- 但适配 FM 的 ODE 采样语义

关键区别：

- 对 diffusion 来说，guidance 作用对象更像 `x0_hat`
- 对 flow matching 来说，guidance 直接作用于当前状态 `x_current`

也就是说，本次实现中，FM 的 legality guidance 是在“当前布局状态”上做几步优化，再把这个位移量叠加回 ODE 更新。


## 3. 本次实现了哪些功能

### 3.1 已实现：flow_GCN 数据到 chipdiffusion 格式的转换

已经生成的数据目录：

- [datasets/graph/chiplet_flow_gcn](/Users/mac/WORK/duffusion/chipdiffusion/datasets/graph/chiplet_flow_gcn)

当前生成结果：

- `graph*.pickle`: 4154 个
- `output*.pickle`: 4154 个
- `config.yaml`: 已生成
- `index_map.json`: 已生成

切分结果：

- `train_samples: 3739`
- `val_samples: 415`


### 3.2 已实现：chipdiffusion 读取 chiplet 数据集

已经实际验证：

- `chipdiffusion` 的 `load_graph_data('chiplet_flow_gcn')` 可以正常读取数据
- 第一条样本张量结构正常

验证结果示例：

- `x_shape = (5, 2)`
- `cond.x = (5, 2)`
- `edge_index = (2, 12)`
- `edge_attr = (12, 4)`


### 3.3 已实现：Flow Matching 最小训练

已经实际验证：

- `train_graph.py` 能使用 `family=flow_matching`
- 使用 `task=chiplet_flow_gcn` 可以完成最小训练
- 前向、loss、反向传播、checkpoint 保存都正常

已生成训练目录：

- [logs/chiplet_fm/chiplet_flow_gcn.chiplet_fm.61](/Users/mac/WORK/duffusion/chipdiffusion/logs/chiplet_fm/chiplet_flow_gcn.chiplet_fm.61)


### 3.4 已实现：Flow Matching 最小 eval 推理

已经实际验证：

- `eval.py` 能加载 FM checkpoint
- 能生成样本 `.pkl`
- 能生成可视化 `.png`
- 能输出 `metrics.csv`

已生成目录：

- [chiplet_flow_gcn.chiplet_fm_eval_smoke.61](/Users/mac/WORK/duffusion/chipdiffusion/logs/chiplet_fm/chiplet_flow_gcn.chiplet_fm_eval_smoke.61)


### 3.5 已实现：FM legality guidance 生效

这是本次最重要的功能性修复。

此前现象：

- 在 eval 时即使打开 `guidance@_global_=sgd`
- 指标和无 guidance 的结果完全一样
- 说明参数虽然传到了配置层，但 `FlowMatchingModel` 没有实际使用 guidance

本次改动后再次验证：

- guidance 开启后，指标发生明显变化
- legality 提升，说明 legality guidance 已经真正影响了 FM 采样过程

实测对比：

无 guidance：

- `legality_2 = 0.429`
- `hpwl_rescaled = 338.27`

有 legality guidance：

- `legality_2 = 0.897`
- `hpwl_rescaled = 665.05`

这说明：

- legality guidance 已经生效
- 当前会明显提升合法性
- 同时会恶化线长
- 后续需要调参平衡 legality 和 HPWL


## 4. 当前 FM 训练的总流程是什么

### 4.1 数据准备阶段

原始数据来源：

- `flow_GCN/gcn_thermal/dataset/input_test/*.json`
- `flow_GCN/gcn_thermal/dataset/output/placement/*.json`

转换工具：

- [convert_flow_gcn_dataset.py](/Users/mac/WORK/duffusion/chipdiffusion/scripts/convert_flow_gcn_dataset.py:1)

转换后格式：

- `chipdiffusion/datasets/graph/chiplet_flow_gcn/graph{i}.pickle`
- `chipdiffusion/datasets/graph/chiplet_flow_gcn/output{i}.pickle`


### 4.2 数据加载阶段

入口：

- [diffusion/utils.py](/Users/mac/WORK/duffusion/chipdiffusion/diffusion/utils.py:348)
- [diffusion/utils.py](/Users/mac/WORK/duffusion/chipdiffusion/diffusion/utils.py:393)

主要逻辑：

1. 读取 `graph{i}.pickle`
2. 读取 `output{i}.pickle`
3. 根据 `config.yaml` 切分 train / val
4. 调 `preprocess_graph()` 做归一化：
   - 节点尺寸归一化
   - edge_attr 归一化
   - placement 从原始坐标转成归一化中心坐标


### 4.3 训练阶段

训练入口：

- [diffusion/train_graph.py](/Users/mac/WORK/duffusion/chipdiffusion/diffusion/train_graph.py:11)

训练过程：

1. `load_graph_data(task)` 读入数据
2. 用 `GraphDataLoader` 组织 `(x, cond)`
3. 根据 `family=flow_matching` 构造 `FlowMatchingModel`
4. 每个 step：
   - 采样时间 `t ~ Uniform(fm_t_min, 1)`
   - 采样噪声 `z`
   - 构造插值状态：
     ```text
     x_t = (1 - t) * x + t * z
     ```
   - 目标速度：
     ```text
     v_target = z - x
     ```
   - 网络预测：
     ```text
     v_pred = model(x_t, cond, t)
     ```
   - 用 MSE 训练：
     ```text
     loss = ||v_pred - v_target||^2
     ```


### 4.4 推理阶段

eval 入口：

- [diffusion/eval.py](/Users/mac/WORK/duffusion/chipdiffusion/diffusion/eval.py:1)

推理采样过程：

1. 从高斯噪声初始化 `x`
2. 对 `t = 1 -> 0` 做离散 ODE 积分
3. 每一步计算：
   ```text
   velocity = model(x, cond, t)
   x_next = x - dt * velocity
   ```
4. 如果开启 guidance：
   ```text
   x_next = x - dt * velocity + guidance_force
   ```
5. 输出最终 placement
6. 保存：
   - `sample*.pkl`
   - `placed*.png`
   - `metrics.csv`


## 5. 当前训练与 eval 的配置方法

### 5.1 运行环境

当前在本机上的可运行方式是：

```bash
cd /Users/mac/WORK/duffusion/chipdiffusion
PYTHONPATH=.:./diffusion conda run -n torch-env python ...
```

注意：

- 这里使用了 `PYTHONPATH=.:./diffusion`
- 原因是项目内部存在顶层导入，如 `import orientations`
- 如果不这样设置，某些脚本会因为导入路径失败


### 5.2 数据转换命令

```bash
cd /Users/mac/WORK/duffusion
PYTHONPATH=.:./diffusion conda run -n torch-env python \
  chipdiffusion/scripts/convert_flow_gcn_dataset.py
```

如需自定义目录：

```bash
cd /Users/mac/WORK/duffusion
PYTHONPATH=.:./diffusion conda run -n torch-env python \
  chipdiffusion/scripts/convert_flow_gcn_dataset.py \
  --input-dir flow_GCN/gcn_thermal/dataset/input_test \
  --placement-dir flow_GCN/gcn_thermal/dataset/output/placement \
  --output-dir chipdiffusion/datasets/graph/chiplet_flow_gcn \
  --val-ratio 0.1
```


## 6. 当前训练指令如何配置

### 6.1 最基本训练

```bash
cd /Users/mac/WORK/duffusion/chipdiffusion
PYTHONPATH=.:./diffusion conda run -n torch-env python \
  diffusion/train_graph.py \
  --config-name config_graph_chiplet_fm
```

这会使用：

- `task=chiplet_flow_gcn`
- `family=flow_matching`
- `backbone=att_gnn`
- `small` 模型尺寸


### 6.2 最小 smoke test 训练

```bash
cd /Users/mac/WORK/duffusion/chipdiffusion
PYTHONPATH=.:./diffusion conda run -n torch-env python \
  diffusion/train_graph.py \
  --config-name config_graph_chiplet_fm \
  train_steps=2 \
  print_every=2 \
  train_data_limit=8 \
  val_data_limit=4 \
  batch_size=2 \
  val_batch_size=2 \
  display_examples=1 \
  logger.wandb=False
```


### 6.3 正式训练建议起点

```bash
cd /Users/mac/WORK/duffusion/chipdiffusion
PYTHONPATH=.:./diffusion conda run -n torch-env python \
  diffusion/train_graph.py \
  --config-name config_graph_chiplet_fm \
  train_steps=100000 \
  print_every=500 \
  batch_size=16 \
  val_batch_size=16 \
  logger.wandb=False
```

如果要改模型容量：

```bash
model/size@model.backbone_params=small
```

或替换成其他 size 配置。


## 7. 当前 eval 指令如何配置

### 7.1 最小无 guidance 推理

```bash
cd /Users/mac/WORK/duffusion/chipdiffusion
PYTHONPATH=.:./diffusion conda run -n torch-env python \
  diffusion/eval.py \
  family=flow_matching \
  task=chiplet_flow_gcn \
  method=chiplet_fm_eval_smoke \
  seed=61 \
  log_dir=logs/chiplet_fm \
  from_checkpoint=chiplet_flow_gcn.chiplet_fm.61/latest.ckpt \
  model/size@model.backbone_params=small \
  logger.wandb=False \
  guidance@_global_=none \
  legalizer@_global_=none \
  num_output_samples=1 \
  val_data_limit=1 \
  eval_samples=0 \
  show_intermediate_every=20 \
  model.max_diffusion_steps=200
```

建议：

- `model.max_diffusion_steps` 最好和训练时一致
- 当前训练配置里默认是 `200`


### 7.2 开启 legality guidance 的最小推理

```bash
cd /Users/mac/WORK/duffusion/chipdiffusion
PYTHONPATH=.:./diffusion conda run -n torch-env python \
  diffusion/eval.py \
  family=flow_matching \
  task=chiplet_flow_gcn \
  method=chiplet_fm_eval_guidance_smoke_v2 \
  seed=61 \
  log_dir=logs/chiplet_fm \
  from_checkpoint=chiplet_flow_gcn.chiplet_fm.61/latest.ckpt \
  model/size@model.backbone_params=small \
  logger.wandb=False \
  guidance@_global_=sgd \
  legalizer@_global_=none \
  num_output_samples=1 \
  val_data_limit=1 \
  eval_samples=0 \
  show_intermediate_every=20 \
  model.max_diffusion_steps=200 \
  model.legality_guidance_weight=3.0 \
  model.hpwl_guidance_weight=0.0 \
  model.grad_descent_steps=5 \
  model.grad_descent_rate=0.1
```


### 7.3 开启 opt guidance 的模板

```bash
cd /Users/mac/WORK/duffusion/chipdiffusion
PYTHONPATH=.:./diffusion conda run -n torch-env python \
  diffusion/eval.py \
  family=flow_matching \
  task=chiplet_flow_gcn \
  method=chiplet_fm_eval_opt \
  seed=61 \
  log_dir=logs/chiplet_fm \
  from_checkpoint=chiplet_flow_gcn.chiplet_fm.61/latest.ckpt \
  model/size@model.backbone_params=small \
  logger.wandb=False \
  guidance@_global_=opt \
  legalizer@_global_=none \
  num_output_samples=1 \
  val_data_limit=1 \
  eval_samples=0 \
  model.max_diffusion_steps=200
```

如果使用 `opt`，建议显式调以下参数：

- `model.alpha_init`
- `model.alpha_lr`
- `model.legality_potential_target`
- `model.grad_descent_steps`
- `model.grad_descent_rate`


## 8. 当前系统的已知限制

### 8.1 `edge_attr` 目前还是占位形式

当前转换脚本使用：

```text
edge_attr = [0, 0, 0, 0]
```

表示连接被近似成“节点中心到节点中心”。

这意味着：

- HPWL 只是代理指标
- 不是基于真实 pin offset 的严格物理线长
- guidance 里的 HPWL 也会受到这个近似影响


### 8.2 训练和 eval 目前仍复用了 diffusion 项目的 eval 配置框架

虽然现在可以跑，但 `config_eval.yaml` 原本是为 diffusion 设计的，因此：

- 里面仍保留了 `noise_schedule`、`beta_1`、`beta_T` 等对 FM 无实际作用的字段
- 后续最好单独写一份 FM 专用 eval 配置


### 8.3 legality guidance 已经生效，但还没调到最优

当前观察到：

- legality 明显上升
- 但 HPWL 明显恶化

所以后续工作重点会是参数平衡，而不是再打通流程。


## 9. 推荐的下一步工作

建议按下面顺序继续：

1. 单独写一份 `config_eval_chiplet_fm.yaml`
2. 对 guidance 参数做小范围 sweep：
   - `legality_guidance_weight`
   - `grad_descent_steps`
   - `grad_descent_rate`
   - `hpwl_guidance_weight`
3. 重新设计 `edge_attr`
   - 把 `wireCount`
   - `EMIB_length`
   - `EMIB_max_width`
   - `EMIB_bump_width`
   更直接地并入模型使用的边特征
4. 如果要进一步提升任务适配性，可以考虑：
   - 在 node feature 中加入 `power`
   - 在 loss 或 eval 中加入更贴近 chiplet 布局的目标函数


## 10. 总结

到目前为止，chiplet Flow Matching 的第一阶段已经完成：

- 数据已接入
- 训练已打通
- eval 已打通
- legality guidance 已在 FM 中真正生效

当前系统已经具备了“可训练、可推理、可调 guidance”的基础能力。后续主要工作将从“打通流程”转向“提升建模质量与调参效果”。
