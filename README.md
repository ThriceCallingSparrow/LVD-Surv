# LVD-Surv

> **Latent Variational Distribution Survival Modeling for Reliability Prediction**
> 
> 面向可靠性预测的潜变量变分分布生存建模框架
> 
> 当前版本：v1.3.0

## 项目简介

**LVD-Surv** 是一个面向设备剩余寿命预测与可靠性分析的开源探索项目，起源于传统 RUL（Remaining Useful Life）预测任务，并尝试从单一寿命点预测进一步扩展到概率化可靠性预测。

项目希望将寿命分布拟合、传感器特征筛选、深度学习时序建模与变分推断思想结合起来，不仅预测设备可能还能运行多久，也尝试输出未来不同时间点的失效风险、可靠度变化和期望剩余寿命。相比直接回归 RUL 数值的常见方法，本项目更关注可靠度曲线、多失效机制、寿命先验融合以及预测结果的可解释性。

目前项目仍处于初版阶段，模型精度、代码结构和工程稳定性都还有较大优化空间。现阶段主要目标是打通从数据处理、寿命分布拟合、特征筛选、模型训练、可靠性预测到结果可视化的完整流程，并提供一个简单的交互式终端界面，方便进行基础实验配置和结果查看。

这个项目并不是一个成熟的工业级可靠性预测工具，而是一次面向可靠性建模与深度学习结合方向的开源探索，也承载了作者本科四年对专业知识与深度学习方法进行融合实践的阶段性总结。

希望它能为同样关注质量时间序列、寿命预测、可靠性分析和预测性维护的研究者或工程实践者提供一些参考和启发。如果你也在关注相关方向，欢迎参考、讨论或基于本项目继续扩展。


## 详细文档

- [零基础完整使用手册](USER_MANUAL.md)
- [项目结构与每个文件作用说明](PROJECT_FILE_GUIDE.md)


## 1. 安装与启动

```bash
cd lvd_surv_project
python -m venv .venv
```

Windows：

```powershell
.venv\Scripts\Activate.ps1
```

Linux/macOS：

```bash
source .venv/bin/activate
```

安装正式依赖：

```bash
pip install -e .
```

启动桌面工作台：

```bash
python main.py
```

也可以双击 `start_lvd.bat`，或在 Linux/macOS 运行 `./start_lvd.sh`。安装后还可使用 `lvd-gui`。

## 2. 桌面工作台

窗口上方是实时输出区，下方是命令输入区，底部显示当前配置、数据集、模型和任务状态。特征分析、混合分布、训练、预测、验证、置换重要性和 SHAP 均通过同一套 `workflows` 调用正式算法。

首次加载配置：

```text
load configs/default.yaml
```

加载后可连续执行：

```text
check
feature
prior
train
test
validate
explain
```

训练完成后自动绑定最佳 checkpoint；预测完成后 `validate` 自动使用最近一次预测结果。

## 3. 常用短命令

```text
help
load [配置文件]
reload
show [config|data|model|settings|output|records]
set <项目> <值>
set reset
check [dependencies|config|data|model|all]
feature [full|quiet|rebuild|show]
prior [full|quiet|rebuild|show]
train
model [checkpoint]
test [current|history|snapshot]
validate [config|model|prediction|all]
explain
run train
run evaluate
run all
results
open [output|plots|reports|model|logs]
status
stop
clear
history
reset
exit
```

常用会话设置：

```text
set log quiet
set plots on
set reports on
set plot-mode current
set samples 20
set shap 128
```

这些设置只影响当前会话，不修改磁盘 YAML。

## 4. 简化后的配置

用户配置只保留真正需要控制的内容：

```yaml
project:
  output_dir: outputs/FD004
data:
  dataset: FD004
  root: datastream/CMAPSSData
  train_file: train_FD004.txt
  test_file: test_FD004.txt
  rul_file: RUL_FD004.txt
  window_size: 50
  stride: 1
  max_horizon: 150
  train_split: 0.85

features:
  enabled: true
  cache_policy: auto
  default_training_feature_mode: raw

lifetime:
  enabled: true
  cache_policy: auto
  max_components: 4
  use_in_inference: true
  blend_weight: 0.25

model: {}
training: {}
inference: {}
explanation: {}
runtime: {}
```

### 已删除的公开配置项

以下内部路径不能再由用户设置：

```text
features.decision_path
features.bundle_path
features.transformer_path
features.manifest_path
features.analysis_output_dir
lifetime_prior.prior_json
lifetime_prior.prior_pkl
lifetime_prior.manifest_path
```

系统根据 `project.output_dir` 自动生成。若配置中仍包含这些字段，加载会明确失败并提示删除，避免同一运行中读取不同目录的产物。

用户配置也不再需要 `data.fd`、`data.output_dir`、`pipeline.mode`、`features.backend` 或 `lifetime_prior.backend`。数据集只保留 `data.dataset`，输出目录只保留 `project.output_dir`。

## 5. 自动产物目录

以 FD004 为例：

```text
outputs/FD004/
├── artifacts/
│   ├── features/
│   │   ├── decision.json
│   │   ├── analysis.pkl
│   │   ├── transformer.pkl
│   │   └── manifest.json
│   └── lifetime/
│       ├── prior.json
│       ├── model.pkl
│       └── manifest.json
├── checkpoints/
├── predictions/
├── explanations/
├── reports/
├── plots/
└── logs/
```

内部产物名称和位置由 `lvd_surv.core.artifacts.build_artifact_paths()` 唯一决定。

## 6. 正式架构

```text
lvd_surv/
├── app/                # 桌面界面、短命令、会话和任务
├── workflows/          # GUI/CLI 共用的唯一流程编排
├── data/               # C-MAPSS 数据、数据集和加载器
├── features/           # 正式特征分析、时序分析和变换器
├── lifetime/           # 正式混合寿命分布和推理先验
├── modeling/           # 模型、损失、指标、训练和推理
├── interpretation/     # 置换重要性和正式 SHAP
├── reporting/          # 可靠度和状态绘图
├── core/               # 配置、路径、产物、契约、校验和错误
└── runtime/            # 取消、输出策略和运行上下文
```

## 7. 高级 CLI

```bash
lvd --help
lvd train --config configs/default.yaml
lvd feature analyze --config configs/default.yaml
lvd prior analyze --config configs/default.yaml
```

预测和解释仍可直接指定 checkpoint：

```bash
lvd predict --checkpoint outputs/FD004/checkpoints/best_model.pt
lvd explain --checkpoint outputs/FD004/checkpoints/best_model.pt
```

普通用户建议优先使用桌面工作台。

## 8. 测试

```bash
python -m pytest -q
python -m compileall -q lvd_surv scripts tests main.py
```
