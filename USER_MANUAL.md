# LVD-Surv 1.3.0 完整使用手册（零基础版）

> 适用对象：第一次接触 Python、机器学习、C-MAPSS 数据或本项目的用户。
>
> 本文按“先能运行，再理解原理，最后进行高级配置”的顺序编写。若你只想尽快使用程序，请优先阅读第 1～8 章。

---

## 1. 项目是什么

LVD-Surv 是一个面向 NASA C-MAPSS 航空发动机退化数据的可靠性预测系统。它读取发动机在多个运行周期中的工况参数和传感器数据，学习设备退化规律，并输出：

- 设备未来每个周期的失效风险（hazard）；
- 设备未来每个周期仍然正常运行的概率（reliability）；
- 预计剩余使用寿命（Expected RUL）；
- 潜在故障模式概率；
- 潜在健康状态；
- 传感器特征重要性；
- SHAP 全局解释结果；
- 混合寿命分布与可靠度先验。

它不仅是一个模型脚本，还包含完整工作流：

```text
加载配置
→ 检查环境和数据
→ 特征分析
→ 混合寿命分布拟合
→ 模型训练
→ 可靠度预测
→ 结果验证
→ 模型解释
```

项目遵循严格原则：正式模块失败时直接停止并说明原因，不会静默切换到简化算法、备用特征、备用分布或跳过 SHAP。

---

## 2. 你需要准备什么

### 2.1 电脑要求

最低建议：

- 64 位 Windows、Linux 或 macOS；
- Python 3.9 或更高版本；
- 8 GB 内存以上；
- 至少 5 GB 可用磁盘空间；
- 完整训练建议 16 GB 内存以上。

GPU 不是必须的。没有 GPU 时，PyTorch 会使用 CPU，训练和 SHAP 计算可能更慢。

### 2.2 安装 Python

如果电脑尚未安装 Python，建议安装 Python 3.10 或 3.11。安装后打开终端并输入：

```bash
python --version
```

若系统使用 `python3` 命令，则输入：

```bash
python3 --version
```

应看到类似：

```text
Python 3.10.13
```

### 2.3 什么是“终端”

- Windows：PowerShell 或“命令提示符”；
- macOS：Terminal（终端）；
- Linux：Terminal。

虽然项目提供桌面窗口，但首次安装依赖通常仍需在终端执行一次。

---

## 3. 解压项目并进入目录

把压缩包解压到一个路径简单、可读写的位置，例如：

```text
D:\Projects\lvd_surv_project
```

或：

```text
/Users/yourname/Projects/lvd_surv_project
```

不要直接在 ZIP 压缩包内部运行程序。

终端进入项目目录：

Windows：

```powershell
cd D:\Projects\lvd_surv_project
```

macOS/Linux：

```bash
cd /Users/yourname/Projects/lvd_surv_project
```

确认目录中能看到：

```text
main.py
README.md
configs/
lvd_surv/
datastream/
```

---

## 4. 创建独立 Python 环境
> **可选步骤**：独立环境用于版本隔离、防止依赖冲突，不执行也可直接使用本机全局 Python 环境运行项目

独立环境可以避免本项目依赖与其他 Python 项目冲突。(注意: 该步骤非必要步骤)

### 4.1 创建环境（可选）

```bash
python -m venv .venv
```

### 4.2 激活环境（可选，仅创建虚拟环境后才需要执行）

Windows PowerShell：

```powershell
.venv\Scripts\Activate.ps1
```

Windows 命令提示符：

```cmd
.venv\Scripts\activate.bat
```

macOS/Linux：

```bash
source .venv/bin/activate
```

激活后，终端行首通常会出现：

```text
(.venv)
```

### 4.3 安装项目与正式依赖

```bash
python -m pip install --upgrade pip
pip install -e .
```

该命令会安装：

- NumPy；
- Pandas；
- scikit-learn；
- Matplotlib；
- SciPy；
- tqdm；
- PyTorch；
- PyYAML；
- seaborn；
- dcor；
- xgboost；
- SHAP。

---

## 5. 第一次启动前检查

安装完成后执行：

```bash
lvd release-check --config configs/default.yaml
```

它会检查：

- 项目必要文件；
- 配置格式；
- 数据文件是否存在；
- 正式依赖是否能导入；
- 输出目录是否可写；
- 是否残留已淘汰入口。

若看到“发布检查通过”，说明基础环境可用。

若 `lvd` 命令不存在，可改用：

```bash
python -m lvd_surv.cli release-check --config configs/default.yaml
```

---

## 6. 启动桌面工作台

推荐普通用户使用桌面工作台。

### 6.1 通用启动方式

```bash
python main.py
```

### 6.2 Windows 双击启动

双击：

```text
start_lvd.bat
```

### 6.3 Linux/macOS

首次需要给予执行权限：

```bash
chmod +x start_lvd.sh
```

然后运行：

```bash
./start_lvd.sh
```

### 6.4 安装后的快捷入口

```bash
lvd-gui
```

---

## 7. 认识桌面窗口

桌面窗口分为三个主要区域：

```text
┌──────────────────────────────────────────────┐
│ 输出区：日志、进度、警告、错误、结果路径      │
│                                              │
├──────────────────────────────────────────────┤
│ 输入区：输入短命令并按 Enter                  │
├──────────────────────────────────────────────┤
│ 状态栏：配置、数据集、模型、任务状态          │
└──────────────────────────────────────────────┘
```

特征分析、训练、预测和 SHAP 会在后台工作线程中运行。界面会持续刷新，不应出现长时间黑屏。

### 7.1 第一个命令

输入：

```text
help
```

输出区会显示全部可用命令。

### 7.2 加载配置

输入：

```text
load configs/default.yaml
```

也可以只输入：

```text
load
```

随后在弹出的文件选择窗口中选择 `configs/default.yaml`。

加载后，后续 `feature`、`prior`、`train`、`test`、`validate`、`explain` 都会自动使用当前配置。

---

## 8. 最推荐的完整使用流程

### 8.1 查看当前状态

```text
show
```

### 8.2 检查依赖、配置和数据

```text
check
```

### 8.3 单独运行特征分析

```text
feature
```

### 8.4 单独拟合混合寿命先验

```text
prior
```

### 8.5 训练模型

```text
train
```

训练完成后，最佳 checkpoint 会自动绑定为当前模型。

### 8.6 预测可靠度

```text
test
```

默认使用 `current` 绘图模式，即每台设备重点显示最新观测时刻的可靠度曲线。

### 8.7 验证预测结果

```text
validate
```

### 8.8 解释模型

```text
explain
```

该命令同时生成：

- permutation importance；
- SHAP GradientExplainer 结果。

### 8.9 一次执行全部流程

```text
run all
```

执行顺序：

```text
检查 → 特征分析 → 混合先验 → 训练 → 预测 → 验证 → 解释
```

首次运行可能耗时很长。建议先用小规模测试配置确认流程。

---

## 9. 桌面短命令详细说明

### 9.1 `help`

```text
help
```

显示命令帮助。

### 9.2 `load`

```text
load configs/default.yaml
```

加载配置。重新加载另一配置时，系统会刷新数据路径，并解除不兼容的模型和预测结果绑定。

### 9.3 `reload`

```text
reload
```

重新读取当前配置文件。适合你在文本编辑器中修改 YAML 后刷新。

### 9.4 `show`

```text
show
show config
show data
show model
show settings
show output
show records
```

分别查看整体状态、配置路径、数据设置、当前模型、会话设置、输出目录和运行记录。

### 9.5 `set`

```text
set
```

查看当前会话设置。

常用设置：

```text
set log quiet
set log normal
set log verbose
set log debug
set plots on
set plots off
set reports on
set reports off
set plot-mode current
set plot-mode history
set plot-mode snapshot
set samples 20
set shap 128
```

`set samples` 控制预测的 Monte Carlo 采样数；`set shap` 控制 SHAP 样本上限。

恢复配置默认值：

```text
set reset
```

### 9.6 `check`

```text
check
check dependencies
check config
check data
check model
check all
```

检查依赖、配置、数据和当前模型兼容性。

### 9.7 `release-check`

```text
release-check
```

执行更完整的发布结构和运行环境检查。

### 9.8 `feature`

```text
feature
feature full
feature quiet
feature rebuild
feature show
```

- `feature`：正常运行或读取有效缓存；
- `feature full`：本次显示详细日志并保存报告和图片；
- `feature quiet`：本次减少普通日志；
- `feature rebuild`：强制重新计算；
- `feature show`：显示最近特征产物。

`full` 和 `quiet` 只影响本次输出，不会永久修改会话设置，也不会改变算法。

### 9.9 `prior`

```text
prior
prior full
prior quiet
prior rebuild
prior show
```

含义与 `feature` 相同，但作用对象是混合寿命分布。

### 9.10 `train`

```text
train
```

使用当前配置执行完整训练。系统会检查或生成正式特征契约和混合寿命先验。

### 9.11 `model`

```text
model
```

弹出 checkpoint 选择窗口。

也可指定路径：

```text
model outputs/FD004/checkpoints/best_model.pt
```

系统会严格检查数据集、窗口长度、预测范围、特征契约和 scaler 维度。

### 9.12 `test`

```text
test
test current
test history
test snapshot
```

- `current`：每台设备突出最后观测时刻；
- `history`：对比多个历史观测时刻；
- `snapshot`：按时刻输出快照。

### 9.13 `validate`

```text
validate
validate config
validate model
validate prediction
validate all
```

不带参数时默认验证最近一次预测结果。

### 9.14 `explain`

```text
explain
```

生成置换重要性和正式 SHAP 结果。必须先有当前 checkpoint。

### 9.15 `run`

```text
run train
run evaluate
run all
```

- `run train`：检查后训练；
- `run evaluate`：预测、验证和解释当前模型；
- `run all`：完整端到端流程。

### 9.16 `results`

```text
results
```

显示当前会话最近生成的模型、预测、解释、特征和先验产物。

### 9.17 `open`

```text
open output
open plots
open reports
open model
open logs
```

使用系统文件管理器打开对应目录或文件。

### 9.18 `status`

```text
status
```

查看当前任务和会话状态。

### 9.19 `stop`

```text
stop
```

请求在最近的安全检查点取消任务。部分底层单次计算无法立即强杀，因此可能需要短暂等待。

### 9.20 `clear`、`history`、`reset`、`exit`

```text
clear
history
reset
exit
```

- `clear`：清空输出窗口；
- `history`：显示命令历史；
- `reset`：解除当前模型和结果绑定，但保留配置；
- `exit`：关闭程序。

---

## 10. 配置文件逐项说明

默认配置位于：

```text
configs/default.yaml
```

YAML 使用空格缩进，不能使用 Tab。冒号后要有空格。

### 10.1 `project`

```yaml
project:
  output_dir: outputs/FD004
```

`output_dir` 是该配置所有产物的根目录。建议不同实验使用不同目录，避免覆盖。

### 10.2 `data`

```yaml
data:
  dataset: FD004
  root: datastream/CMAPSSData
  train_file: train_FD004.txt
  test_file: test_FD004.txt
  rul_file: RUL_FD004.txt
  known_condition: true
  condition_columns: [setting_1, setting_2, setting_3]
  selected_sensor_columns: null
  window_size: 50
  stride: 1
  max_horizon: 150
  train_split: 0.85
  random_seed: 42
```

- `dataset`：FD001、FD002、FD003 或 FD004；
- `root`：数据目录；
- `train_file`：训练数据文件名；
- `test_file`：测试数据文件名；
- `rul_file`：测试设备末端真实 RUL 文件；
- `known_condition`：是否使用已知工况标签；
- `condition_columns`：工况列；
- `selected_sensor_columns`：手工指定传感器，`null` 表示由正式特征分析决定；
- `window_size`：每个样本包含的历史周期数；
- `stride`：滑动窗口步长；
- `max_horizon`：向未来预测的最大周期数；
- `train_split`：按设备划分训练集比例；
- `random_seed`：随机种子。

不要随意修改 `window_size` 或 `max_horizon` 后继续使用旧 checkpoint，因为模型结构和输出维度可能不兼容。

### 10.3 `features`

```yaml
features:
  enabled: true
  cache_policy: auto
  default_training_feature_mode: raw
  residual_condition_columns: [setting_1, setting_2, setting_3]
  include_condition_dummies: true
  residual_suffix: _resid
  full_validation: true
  enable_deconfounding: false
```

- `enabled`：是否执行正式特征分析；
- `cache_policy`：`auto`、`force`、`readonly`、`off`；
- `default_training_feature_mode`：`raw`、`residual`、`hybrid`；
- `residual_condition_columns`：残差特征去除工况影响时使用的列；
- `include_condition_dummies`：是否加入工况哑变量；
- `residual_suffix`：残差列后缀；
- `full_validation`：是否执行完整特征验证；
- `enable_deconfounding`：是否启用正式去混杂流程。

缓存策略：

- `auto`：有有效缓存则读取，否则重算；
- `force`：强制重算；
- `readonly`：只能读取，缺少或失效就报错；
- `off`：本次不使用缓存。

### 10.4 `lifetime`

```yaml
lifetime:
  enabled: true
  cache_policy: auto
  max_components: 4
  use_in_inference: true
  blend_weight: 0.25
```

- `enabled`：是否拟合正式混合寿命分布；
- `cache_policy`：缓存策略；
- `max_components`：最多尝试的混合成分数；
- `use_in_inference`：预测时是否融合寿命先验；
- `blend_weight`：先验融合权重。

### 10.5 `model`

```yaml
model:
  encoder_type: tcn
  hidden_dim: 96
  latent_dim: 16
  num_modes: 6
  num_tcn_layers: 4
  tcn_kernel_size: 3
  transformer_layers: 2
  transformer_heads: 4
  dropout: 0.1
```

这些参数控制模型结构。修改后不能直接使用旧 checkpoint。

### 10.6 `training`

```yaml
training:
  batch_size: 128
  epochs: 60
  learning_rate: 0.001
  weight_decay: 0.0001
  beta_kl_final: 0.01
  kl_anneal_epochs: 20
  lambda_mono: 0.02
  lambda_cond: 0.2
  lambda_orth: 0.01
  gradient_clip_norm: 5.0
  early_stop_patience: 12
  num_workers: 0
```

- `batch_size`：每批样本数；
- `epochs`：最大训练轮数；
- `learning_rate`：学习率；
- `weight_decay`：权重衰减；
- `beta_kl_final`：KL 项最终权重；
- `kl_anneal_epochs`：KL 退火轮数；
- `lambda_mono`：健康状态单调性损失权重；
- `lambda_cond`：工况约束权重；
- `lambda_orth`：故障模式正交约束权重；
- `gradient_clip_norm`：梯度裁剪阈值；
- `early_stop_patience`：早停耐心轮数；
- `num_workers`：DataLoader 子进程数。

### 10.7 `inference`

```yaml
inference:
  mc_samples: 20
  plot_mode: current
  plot_max_curves_per_device: 8
```

- `mc_samples`：变分推理 Monte Carlo 采样次数；
- `plot_mode`：`current`、`history`、`snapshot`；
- `plot_max_curves_per_device`：历史模式每台设备最多绘制曲线数。

### 10.8 `explanation`

```yaml
explanation:
  permutation_repeats: 3
  shap_sample_size: 128
```

- `permutation_repeats`：置换重要性重复次数；
- `shap_sample_size`：SHAP 样本上限。

### 10.9 `runtime`

```yaml
runtime:
  verbosity: normal
  save_reports: false
  save_plots: false
```

- `verbosity`：`quiet`、`normal`、`verbose`、`debug`；
- `save_reports`：是否保存详细报告；
- `save_plots`：是否保存分析图片。

---

## 11. 数据文件格式

标准 C-MAPSS 训练和测试文件每行包括：

```text
设备编号
周期
3 个运行工况参数
21 个传感器值
```

程序内部会把列规范化为：

```text
unit_id
cycle
setting_1 ～ setting_3
sensor_1 ～ sensor_21
```

训练文件包含设备完整运行至失效的历史；测试文件通常在失效前截断；`RUL_FDxxx.txt` 提供每台测试设备在最后观测周期后的真实剩余寿命。

不要手工删除列、调整列顺序或添加表头，除非同时修改正式数据读取代码并完成验证。

---

## 12. 输出目录和文件怎么看

以 `outputs/FD004` 为例：

```text
outputs/FD004/
├── artifacts/
│   ├── features/
│   └── lifetime/
├── checkpoints/
├── predictions/
├── explanations/
├── reports/
├── plots/
└── logs/
```

### 12.1 特征产物

```text
artifacts/features/decision.json
artifacts/features/analysis.pkl
artifacts/features/transformer.pkl
artifacts/features/manifest.json
```

- `decision.json`：最终特征契约；
- `analysis.pkl`：完整分析对象；
- `transformer.pkl`：residual/hybrid 变换器；
- `manifest.json`：数据、配置和代码指纹。

### 12.2 寿命先验产物

```text
artifacts/lifetime/prior.json
artifacts/lifetime/model.pkl
artifacts/lifetime/manifest.json
```

### 12.3 模型

```text
checkpoints/best_model.pt
checkpoints/last_model.pt
```

通常预测和解释使用 `best_model.pt`。

### 12.4 预测

常见结果：

```text
predictions/all_reliability_values.csv
predictions/reliability_values/device_001_reliability.csv
predictions/reliability_curves/device_001_reliability.png
predictions/states/all_mode_health_states.csv
```

关键列：

- `unit_id`：设备编号；
- `current_time`：作出预测时的当前周期；
- `future_step`：未来第几个周期；
- `hazard`：该未来周期失效风险；
- `failure_probability`：该未来周期失效概率质量；
- `reliability`：运行到该未来周期仍未失效的概率；
- `expected_rul_from_current`：当前时刻预计剩余寿命。

### 12.5 解释

```text
explanations/global_feature_importance.csv
explanations/global_feature_importance.png
explanations/global_shap_importance.csv
explanations/global_shap_importance.png
```

### 12.6 日志和运行记录

```text
logs/desktop_session_*.log
logs/resolved_config_*.yaml
logs/workflow_runs.jsonl
```

`resolved_config` 记录本次实际使用的完整配置，是复现实验的重要文件。

---

## 13. 高级命令行用法

### 13.1 查看帮助

```bash
lvd --help
```

### 13.2 训练

```bash
lvd train --config configs/default.yaml
```

### 13.3 特征分析

```bash
lvd feature analyze --config configs/default.yaml
```

详细输出并保存报告与图：

```bash
lvd feature analyze --config configs/default.yaml --verbosity verbose --save-reports --save-plots
```

### 13.4 混合先验

```bash
lvd prior analyze --config configs/default.yaml
```

### 13.5 预测

```bash
lvd predict --checkpoint outputs/FD004/checkpoints/best_model.pt
```

显式指定测试文件：

```bash
lvd predict \
  --checkpoint outputs/FD004/checkpoints/best_model.pt \
  --test-file datastream/CMAPSSData/test_FD004.txt \
  --rul-file datastream/CMAPSSData/RUL_FD004.txt \
  --output-dir outputs/FD004/predictions \
  --mc-samples 20 \
  --plot-mode current
```

### 13.6 解释

```bash
lvd explain --checkpoint outputs/FD004/checkpoints/best_model.pt
```

### 13.7 配置检查

```bash
lvd config validate --config configs/default.yaml
```

### 13.8 依赖检查

```bash
lvd doctor
```

### 13.9 发布检查

```bash
lvd release-check --config configs/default.yaml
```

---

## 14. 推荐的测试方法

### 14.1 流程测试配置

复制配置：

```text
configs/default.yaml
→ configs/test.yaml
```

修改：

```yaml
project:
  output_dir: outputs/FD004_test
training:
  epochs: 1
```

该配置只用于验证流程能否运行，不能用于评价最终精度。

### 14.2 自动测试

```bash
python -m pytest -q
```

### 14.3 Python 编译检查

```bash
python -m compileall -q lvd_surv scripts tests main.py
```

---

## 15. 常见问题与处理方法

### 15.1 `ModuleNotFoundError`

原因通常是依赖未安装或虚拟环境未激活。

处理：

```bash
source .venv/bin/activate
pip install -e .
```

Windows 使用对应激活命令。

### 15.2 `lvd` 命令不存在

使用：

```bash
python -m lvd_surv.cli --help
```

并确认已经执行：

```bash
pip install -e .
```

### 15.3 数据文件不存在

检查：

```text
datastream/CMAPSSData/
```

以及 YAML 中 `data.root` 和文件名是否正确。

### 15.4 checkpoint 不兼容

常见原因：

- 配置数据集不同；
- `window_size` 不同；
- `max_horizon` 不同；
- 特征契约不同；
- scaler 维度不同。

不要强行绕过检查。应使用与训练时一致的配置，或重新训练。

### 15.5 SHAP 很慢

先减小：

```text
set shap 16
```

确认流程后再提高到 128 或更大。

### 15.6 界面日志太多

```text
set log quiet
```

完整日志仍会写入日志文件。

### 15.7 任务停止不立即生效

`stop` 是协作式取消。程序会在安全检查点退出，不会强行中断正在进行的底层数值运算，以避免产物损坏。

### 15.8 图中为什么有多条可靠度曲线

`history` 模式会绘制多个观测时刻的预测。只想查看当前可靠度时使用：

```text
set plot-mode current
test
```

### 15.9 缓存为什么不生效

缓存有效性会结合数据、配置和代码指纹判断。数据或关键配置发生变化时，旧缓存会被视为无效，这是为了防止误用。

---

## 16. 使用安全建议

1. 正式实验前复制配置，不要反复覆盖默认配置。
2. 每次实验使用独立输出目录。
3. 保留 `resolved_config_*.yaml` 和日志。
4. 不要手工修改 checkpoint、PKL 或 manifest。
5. 不要删除特征变换器后继续使用 residual/hybrid 模型。
6. 不要把其他数据集的 checkpoint 绑定到当前配置。
7. 模型预测只能作为工程分析参考，不应替代实际维护规范和安全决策。

---

## 17. 程序设计总结

LVD-Surv 1.3.0 将复杂的可靠性分析整合为一个状态化桌面工作台。其主要优势是：

- 用户加载一次配置即可连续完成全部流程；
- GUI 与 CLI 共用唯一正式工作流；
- 训练与推理严格共享特征契约和 scaler；
- 正式混合寿命先验可以独立分析，也可用于预测；
- 支持 raw、residual 和 hybrid 特征模式；
- 同时提供置换重要性和 SHAP；
- 所有关键步骤都有缓存、指纹、校验和运行记录；
- 不采用静默 fallback，避免用户误判。

对于零基础用户，推荐固定操作顺序：

```text
安装 → release-check → python main.py → load → check → run all
```

对于正式研究和工程验证，建议分别运行特征、先验、训练、预测和解释，并保存每次实际配置与日志。
