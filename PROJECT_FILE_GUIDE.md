# LVD-Surv 1.3.0 项目结构与每个文件作用说明

> 本文逐项解释稳定版项目中每个目录和文件的用途。
>
> 标记说明：
>
> - **用户常用**：普通用户可能直接打开或运行；
> - **开发维护**：主要由开发人员维护；
> - **自动生成**：运行程序后产生，不应手工编辑；
> - **只读资料**：用于参考，不参与主运行逻辑。

---

## 1. 顶层目录总览

```text
lvd_surv_project/
├── main.py
├── start_lvd.bat
├── start_lvd.sh
├── README.md
├── docs/
├── configs/
├── datastream/
├── lvd_surv/
├── scripts/
├── tests/
├── pyproject.toml
├── requirements.txt
├── environment.yaml
└── 发布说明与测试记录文件
```

核心分层：

```text
app            用户界面、短命令、会话、后台任务
workflows      GUI/CLI 共用的正式流程编排
data           C-MAPSS 数据读取与样本生成
features       特征分析与特征变换
lifetime       混合寿命分布与寿命先验
modeling       模型、损失、训练、推理、指标
interpretation 置换重要性和 SHAP
reporting      绘图
core           配置、路径、产物、契约、校验、错误
runtime        取消、日志和运行上下文
```

---

## 2. 顶层文件

### `main.py`

- 类型：**用户常用**。
- 作用：桌面工作台主入口。
- 使用：

```bash
python main.py
```

- 内部行为：调用 `lvd_surv.app.desktop.launch()`。
- 用户是否应修改：通常不应修改。

### `start_lvd.bat`

- 类型：**用户常用，Windows**。
- 作用：方便 Windows 用户双击启动桌面工作台。
- 前提：项目环境和依赖已经安装。
- 用户是否应修改：一般不需要。

### `start_lvd.sh`

- 类型：**用户常用，Linux/macOS**。
- 作用：启动桌面工作台。
- 首次可能需要：

```bash
chmod +x start_lvd.sh
```

### `README.md`

- 类型：**用户常用**。
- 作用：项目首页、快速安装、核心命令、配置结构和稳定版状态说明。
- 与 `docs/USER_MANUAL_CN.md` 的区别：README 是快速入口，完整操作细节在用户手册中。

### `pyproject.toml`

- 类型：**开发维护**。
- 作用：定义包名称、版本、Python 要求、依赖和命令入口。
- 注册命令：

```text
lvd     → lvd_surv.cli:main
lvd-gui → lvd_surv.app.desktop:launch
```

- 不建议普通用户随意修改依赖版本。

### `requirements.txt`

- 类型：**用户/部署人员常用**。
- 作用：列出 pip 依赖，适合某些部署环境使用：

```bash
pip install -r requirements.txt
```

正式推荐仍是：

```bash
pip install -e .
```

### `environment.yaml`

- 类型：**用户/部署人员可用**。
- 作用：Conda 环境描述文件。
- 用法通常是：

```bash
conda env create -f environment.yaml
```

### `FINAL_RELEASE_NOTES.md`

- 类型：**只读资料**。
- 作用：1.3.0 稳定版最终发布摘要、完成事项和版本结论。

### `IMPLEMENTATION_STATUS.md`

- 类型：**只读资料/维护记录**。
- 作用：记录多阶段重构实施状态和已完成范围。

### `STAGE4_RELEASE_NOTES.md`

- 类型：**历史只读资料**。
- 作用：第四阶段配置和路径收敛说明。

### `STAGE5_RELEASE_NOTES.md`

- 类型：**历史只读资料**。
- 作用：第五阶段发布候选检查和收尾说明。

### `TEST_RESULTS_FINAL.txt`

- 类型：**只读测试记录**。
- 作用：最终稳定版自动测试和检查摘要。

### `TEST_RESULTS_STAGE4.txt`

- 类型：**历史测试记录**。
- 作用：第四阶段测试结果。

### `TEST_RESULTS_STAGE5.txt`

- 类型：**历史测试记录**。
- 作用：第五阶段测试结果。

### `USER_TEST_CHECKLIST.md`

- 类型：**用户/维护人员常用**。
- 作用：部署、升级、回归和维护测试清单。

---

## 3. `docs/` 文档目录

### `docs/USER_MANUAL_CN.md`

- 类型：**用户常用**。
- 作用：面向零基础用户的完整中文使用手册。
- 内容：安装、环境、GUI、短命令、配置、数据、输出、CLI、测试和故障排查。

### `docs/PROJECT_FILE_GUIDE_CN.md`

- 类型：**用户/开发人员常用**。
- 作用：本文，逐项解释所有文件和目录。

---

## 4. `configs/` 配置目录

### `configs/default.yaml`

- 类型：**用户常用**。
- 作用：正式默认配置。
- 控制：数据集、特征、寿命先验、模型、训练、推理、解释和运行输出。
- 建议：复制后修改，不要直接把正式默认配置改成临时测试配置。

### `configs/schema.yaml`

- 类型：**开发维护/校验参考**。
- 作用：描述允许的配置区块、字段和结构，用于配置规范说明与校验维护。
- 普通用户通常不需修改。

---

## 5. `datastream/CMAPSSData/` 数据目录

### 训练数据文件

- `train_FD001.txt`：FD001 完整训练轨迹。
- `train_FD002.txt`：FD002 完整训练轨迹。
- `train_FD003.txt`：FD003 完整训练轨迹。
- `train_FD004.txt`：FD004 完整训练轨迹，也是默认配置使用的训练文件。

这些文件均为**正式输入数据**，每行包含设备编号、周期、工况和传感器值。

### 测试数据文件

- `test_FD001.txt`：FD001 测试轨迹。
- `test_FD002.txt`：FD002 测试轨迹。
- `test_FD003.txt`：FD003 测试轨迹。
- `test_FD004.txt`：FD004 测试轨迹，也是默认配置使用的测试文件。

测试轨迹通常在设备失效前截断。

### 测试集真实 RUL 文件

- `RUL_FD001.txt`：FD001 各测试设备最后观测点之后的真实 RUL。
- `RUL_FD002.txt`：FD002 各测试设备最后观测点之后的真实 RUL。
- `RUL_FD003.txt`：FD003 各测试设备最后观测点之后的真实 RUL。
- `RUL_FD004.txt`：FD004 各测试设备最后观测点之后的真实 RUL。

这些文件均为**正式输入数据**，用于为测试轨迹构造真实失效时间并评价预测。

### `readme.txt`

- 类型：**只读资料**。
- 作用：C-MAPSS 数据集原始说明、列结构和子数据集描述。

### `Damage Propagation Modeling.pdf`

- 类型：**只读资料**。
- 作用：与 NASA C-MAPSS 损伤传播模拟相关的参考文献。
- 不参与程序运行。

---

## 6. `lvd_surv/` 主程序包

### `lvd_surv/__init__.py`

- 类型：**开发维护**。
- 作用：标记 `lvd_surv` 为 Python 包，并提供项目级说明和版本相关入口。

### `lvd_surv/cli.py`

- 类型：**用户/开发人员常用**。
- 作用：高级 CLI 和文本交互 Shell。
- 定义命令解析器、`lvd` 命令、`doctor`、`release-check`、训练、预测、解释等入口。
- 桌面工作台和 CLI 最终都调用正式工作流，而不是各自实现算法。

### `lvd_surv/utils.py`

- 类型：**开发维护**。
- 作用：通用小工具，包括随机种子、目录创建、JSON 保存、YAML 加载和模型参数统计。
- 不包含主算法。

---

## 7. `lvd_surv/app/` 桌面应用层

### `lvd_surv/app/__init__.py`

- 标记 `app` 为包。

### `lvd_surv/app/commands.py`

- 类型：**核心控制层**。
- 作用：解析桌面短命令并维护命令语义。
- 主要类：`CommandEngine`。
- 负责：
  - `load`、`show`、`set`、`check`；
  - `feature`、`prior`、`train`、`test`、`explain`；
  - `run all` 等复合命令；
  - 运行记录和配置快照登记；
  - 结果路径注册。
- 不负责特征、模型或 SHAP 的数学计算。

### `lvd_surv/app/desktop.py`

- 类型：**核心桌面界面**。
- 作用：建立 Tkinter 窗口、输出区、输入区、状态栏和按钮。
- 主要类：
  - `QueueWriter`：把 stdout/stderr 写入消息队列；
  - `DesktopApp`：主窗口。
- 重要稳定性机制：日志批量刷新、队列限流、文本限长、后台任务、安全停止、Agg 绘图后端。

### `lvd_surv/app/session.py`

- 类型：**核心会话状态**。
- 主要类：`SessionContext`。
- 保存：
  - 当前配置；
  - 当前数据集；
  - 当前 checkpoint；
  - 最近特征、先验、预测和解释结果；
  - 会话设置；
  - 最近命令和状态。

### `lvd_surv/app/tasks.py`

- 类型：**核心任务控制**。
- 主要类：`TaskRunner`。
- 作用：在工作线程中运行耗时任务，通过队列向 GUI 传递日志、状态、结果和错误，并管理取消事件。

---

## 8. `lvd_surv/core/` 基础设施层

### `lvd_surv/core/__init__.py`

- 标记核心基础设施包。

### `lvd_surv/core/artifacts.py`

- 作用：统一保存和读取 JSON、PKL、manifest、指纹和运行状态。
- 关键能力：
  - 数据指纹；
  - 配置指纹；
  - 脚本指纹；
  - 缓存有效性判断；
  - 标准产物路径生成。
- 用户不应手工修改其生成的 manifest。

### `lvd_surv/core/config.py`

- 作用：唯一正式配置加载与规范化入口。
- 负责：
  - 读取 YAML；
  - 检查区块类型；
  - 拒绝过时配置；
  - 解析相对路径；
  - 应用会话临时设置。

### `lvd_surv/core/contracts.py`

- 作用：定义和应用正式契约。
- 主要处理：
  - 特征决策契约；
  - 特征排序与选择；
  - residual/hybrid 变换；
  - 混合寿命先验契约。
- 这是保证训练和推理一致性的关键文件。

### `lvd_surv/core/errors.py`

- 主要类：`LVDWorkflowError`。
- 作用：提供结构化错误，包含模块、阶段、原因和处理建议。

### `lvd_surv/core/paths.py`

- 主要类：`ProjectPaths`。
- 作用：集中解析项目根目录和输出目录，生成 checkpoints、predictions、explanations、reports、plots、logs 路径。

### `lvd_surv/core/run_records.py`

- 作用：保存实际解析配置快照和追加式工作流运行记录。
- 产物：
  - `resolved_config_*.yaml`；
  - `workflow_runs.jsonl`。

### `lvd_surv/core/validators.py`

- 作用：验证特征契约、混合先验、checkpoint 结构和预测 DataFrame。
- 防止缺少关键字段或数值范围错误的产物进入后续流程。

---

## 9. `lvd_surv/data/` 数据层

### `lvd_surv/data/__init__.py`

- 标记数据领域包。

### `lvd_surv/data/cmapss.py`

- 作用：C-MAPSS 文本读取、列规范化、RUL 构造、特征缩放、样本目标和数据集划分。
- 主要类：
  - `DataSpec`：数据和缩放信息描述；
  - `SlidingWindowSurvivalDataset`：把设备时间序列转换成滑动窗口生存样本。
- 关键函数：
  - `read_cmapss_txt`；
  - `read_rul_txt`；
  - `add_train_rul`；
  - `add_test_rul`；
  - `make_survival_target`；
  - `split_by_unit`。

### `lvd_surv/data/loader.py`

- 主要类：`CMAPSSLoader`。
- 作用：按数据集名称加载训练、测试和 RUL 文件，并提供统一数据 bundle。

---

## 10. `lvd_surv/features/` 特征领域

### `lvd_surv/features/__init__.py`

- 标记特征包。

### `lvd_surv/features/analysis.py`

- 作用：正式特征分析流程入口和缓存管理。
- 负责：
  - 调用时序分析；
  - 建立特征决策；
  - 拟合 residual/hybrid 变换器；
  - 保存契约、分析对象和 manifest；
  - 严格处理缓存策略。

### `lvd_surv/features/time_series.py`

- 作用：完整时序特征研究与验证实现。
- 包含：
  - 设备内/跨设备相关性；
  - 距离相关性；
  - 生命周期分段；
  - 异常值处理；
  - 统计验证；
  - 可视化验证；
  - 交互特征；
  - 模型验证；
  - SHAP 分级；
  - 工况去混杂；
  - residual 特征评估；
  - 报告生成。

### `lvd_surv/features/transformer.py`

- 主要类：`LinearResidualFeatureTransformer`。
- 作用：学习工况对传感器的线性影响并生成 residual 特征。
- 训练和推理必须使用同一个已保存变换器。

---

## 11. `lvd_surv/lifetime/` 寿命分布领域

### `lvd_surv/lifetime/__init__.py`

- 标记寿命领域包。

### `lvd_surv/lifetime/analysis.py`

- 作用：正式混合寿命分布拟合与完整评估。
- 主要类：`MixtureDistributionEM`。
- 包含：
  - 候选单分布拟合；
  - 混合分布组合生成；
  - EM；
  - 最优模型选择；
  - EDF 拟合优度；
  - P-P/Q-Q 图；
  - hazard 对比；
  - Bootstrap；
  - 交叉验证；
  - 寿命指标；
  - 缓存与契约输出。

### `lvd_surv/lifetime/prior.py`

- 主要类：`GenericMixturePrior`。
- 作用：从 JSON 契约重建通用混合寿命先验，并将先验与模型预测尾部融合。
- 主要函数：
  - `build_prior_from_contract`；
  - `blend_tail_with_prior`。

---

## 12. `lvd_surv/modeling/` 模型领域

### `lvd_surv/modeling/__init__.py`

- 标记模型包。

### `lvd_surv/modeling/model.py`

- 作用：正式 LVD-Surv 神经网络结构。
- 主要类：
  - `Chomp1d`；
  - `TemporalBlock`；
  - `TCNEncoder`；
  - `CausalTransformerEncoder`；
  - `GaussianLatent`；
  - `ModelConfig`；
  - `LVDSurvModel`。
- 输出 hazard、reliability、mode probability、health score 等。

### `lvd_surv/modeling/losses.py`

- 作用：正式损失函数和可靠度数学工具。
- 包含：
  - 离散生存负对数似然；
  - 健康状态单调性；
  - 模式熵；
  - 正交约束；
  - 总损失；
  - hazard 转 failure PMF；
  - expected RUL。

### `lvd_surv/modeling/metrics.py`

- 作用：可靠度结果有效性指标和报告。
- 主要函数：`reliability_validity_report`。

### `lvd_surv/modeling/trainer.py`

- 作用：正式训练和验证循环。
- 包含：
  - batch 搬移；
  - 单 epoch 训练；
  - 验证；
  - 数据准备；
  - checkpoint 保存；
  - early stopping；
  - 协作式取消。

### `lvd_surv/modeling/inference.py`

- 作用：checkpoint 加载、推理数据准备、严格特征约束、scaler 应用、MC 推理、CSV 和图片输出。
- 主要函数：`predict_dataset`。

---

## 13. `lvd_surv/interpretation/` 解释领域

### `lvd_surv/interpretation/__init__.py`

- 标记解释包。

### `lvd_surv/interpretation/explain.py`

- 作用：正式模型解释。
- 包含：
  - permutation importance；
  - SHAP GradientExplainer；
  - CSV 保存；
  - 重要性图片保存。
- SHAP 是正式功能，缺失依赖时直接失败。

---

## 14. `lvd_surv/reporting/` 绘图层

### `lvd_surv/reporting/__init__.py`

- 标记报告包。

### `lvd_surv/reporting/plots.py`

- 作用：可靠度曲线和潜在健康/模式状态图。
- 主要函数：
  - `plot_device_reliability`；
  - `plot_mode_health`。
- 支持 current、history、snapshot 绘图语义。

---

## 15. `lvd_surv/runtime/` 运行控制层

### `lvd_surv/runtime/__init__.py`

- 标记运行控制包。

### `lvd_surv/runtime/cancellation.py`

- 主要异常：`TaskCancelledError`。
- 作用：保存当前取消事件、检查用户取消，并在安全检查点终止长任务。

### `lvd_surv/runtime/context.py`

- 主要类：
  - `OutputPolicy`；
  - `RuntimeContext`。
- 作用：让 CLI、桌面、脚本共用同一套配置、路径和输出策略。

### `lvd_surv/runtime/logging.py`

- 主要类：`RuntimeLogger`。
- 作用：根据 quiet、normal、verbose、debug 控制输出。

---

## 16. `lvd_surv/workflows/` 正式工作流层

工作流只负责编排，不重新实现算法。GUI 与 CLI 都调用这些文件。

### `lvd_surv/workflows/__init__.py`

- 标记正式工作流包。

### `lvd_surv/workflows/feature.py`

- 作用：加载训练数据、准备路径并调用正式特征分析。

### `lvd_surv/workflows/prior.py`

- 作用：加载训练寿命并调用正式混合分布拟合。

### `lvd_surv/workflows/training.py`

- 作用：编排训练完整流程：数据、特征、先验、正式 trainer、checkpoint。

### `lvd_surv/workflows/prediction.py`

- 作用：编排预测，调用正式 `predict_dataset`，登记预测产物。

### `lvd_surv/workflows/explanation.py`

- 作用：准备训练解释数据并统一运行 permutation 和 SHAP。

### `lvd_surv/workflows/validation.py`

- 作用：依赖、配置、数据、模型、预测、项目结构和发布就绪检查。

---

## 17. `scripts/` 工具脚本

### `scripts/cli.py`

- 作用：从项目源码目录直接启动统一 CLI 的薄入口。
- 使用：

```bash
python scripts/cli.py --help
```

### `scripts/release_check.py`

- 作用：独立执行稳定版发布检查。
- 适合打包、部署或 CI 使用。

### `scripts/validate_outputs.py`

- 作用：验证已有 checkpoint、预测结果或其他产物结构。
- 主要用于维护和回归检查。

### `scripts/check_documentation.py`

- 作用：扫描公开模块、类和函数的 docstring 覆盖率。
- 用于确保项目注释和文档覆盖达到维护要求。

---

## 18. `tests/` 自动测试目录

这些测试不会参与正式训练，但用于证明架构、严格行为和界面控制没有回退。

### `tests/test_cleanup_and_shap.py`

- 验证旧 fallback 和辅助模块已清理；
- 验证 SHAP 为正式依赖和正式输出。

### `tests/test_desktop_session.py`

- 验证配置、checkpoint、设置和最近产物在桌面会话中的继承与刷新。

### `tests/test_final_stable_release.py`

- 验证 1.3.0 稳定版状态、必要文档、版本和最终发布结构。

### `tests/test_gui_stability.py`

- 验证 GUI 日志限流、批量刷新、队列处理、Agg 后端和稳定性机制。

### `tests/test_stage2_state_and_records.py`

- 验证第二阶段的状态继承、配置快照和工作流记录。

### `tests/test_stage3_architecture.py`

- 验证领域目录迁移、唯一工作流和旧重复业务脚本删除。

### `tests/test_stage4_config_and_paths.py`

- 验证简化配置、自动产物路径和过时键拒绝。

### `tests/test_stage5_release_readiness.py`

- 验证发布检查命令、依赖和目录写权限检查。

### `tests/test_strict_contracts.py`

- 验证特征契约、先验契约和 checkpoint 的严格失败行为。

### `tests/test_workflow_adapters.py`

- 验证 GUI、CLI 和工作流适配层调用同一正式实现，没有复制算法。

---

## 19. 运行后自动生成但压缩包中默认不存在的目录

### `outputs/`

- 类型：**自动生成**。
- 包含模型、预测、解释、报告、图和日志。
- 正式发布包通常不携带旧输出，避免用户误认为旧结果属于当前运行。

### `.venv/`

- 类型：**用户本机生成**。
- Python 虚拟环境，不应打入项目发布包。

### `__pycache__/`、`.pytest_cache/`、`*.pyc`

- 类型：**自动缓存**。
- 不属于项目源码，发布打包时应清除。

---

## 20. 哪些文件可以改，哪些不应改

### 普通用户可修改

```text
configs/default.yaml 的副本
自己的数据文件
README 或个人实验说明
```

### 普通用户不应手工修改

```text
*.pt checkpoint
*.pkl 分析和变换器
manifest.json
feature decision 契约
mixture prior 契约
workflow_runs.jsonl
```

### 开发人员修改后必须完整回归

```text
lvd_surv/data/
lvd_surv/features/
lvd_surv/lifetime/
lvd_surv/modeling/
lvd_surv/interpretation/
lvd_surv/core/contracts.py
```

这些文件可能直接影响数值结果、模型精度或训练推理一致性。

---

## 21. 项目主流程调用关系

```text
main.py
  → app.desktop
    → app.commands
      → workflows.*
        → data / features / lifetime / modeling / interpretation
          → core / runtime / reporting
```

训练：

```text
workflows.training
→ data.loader / data.cmapss
→ features.analysis
→ lifetime.analysis
→ modeling.trainer
→ checkpoints
```

预测：

```text
workflows.prediction
→ modeling.inference
→ feature contract + transformer + scaler
→ model forward + MC sampling
→ lifetime prior blending
→ reporting.plots
```

解释：

```text
workflows.explanation
→ checkpoint feature contract
→ interpretation.explain
→ permutation importance + SHAP
```

---

## 22. 总结

这个项目已经把研究型算法和普通用户操作界面分离：

- 用户主要接触 `main.py`、配置和输出目录；
- `app` 管理交互；
- `workflows` 管理流程；
- 各领域模块保留完整正式算法；
- `core` 和 `runtime` 保证路径、契约、缓存、记录和错误可追踪；
- 测试文件确保后续维护不会重新引入静默 fallback 或重复主逻辑。

理解项目时，推荐阅读顺序：

```text
README.md
→ docs/USER_MANUAL_CN.md
→ configs/default.yaml
→ docs/PROJECT_FILE_GUIDE_CN.md
→ lvd_surv/workflows/
→ 对应算法领域目录
```
