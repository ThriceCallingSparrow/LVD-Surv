# LVD-Surv 1.3.0 维护测试清单

此清单用于新机器部署、依赖升级、配置修改或后续代码维护后的回归检查。

## A. 自动回归

```bash
python -m pytest -q
python -m compileall -q lvd_surv scripts tests main.py
```

## B. 桌面工作台冒烟测试

```bash
python main.py
```

依次执行：

```text
load configs/default.yaml
show
check
release-check
```

确认窗口不黑屏、不冻结，状态栏与输出正常。

## C. 短流程测试

使用独立测试配置和 1–2 epoch：

```text
feature
prior
train
test current
validate prediction
set shap 16
explain
```

检查配置、checkpoint 和预测结果能够自动衔接，SHAP 不得被跳过。

## D. 故障反馈材料

出现问题时请保留：

- `logs/resolved_config_*.yaml`；
- `logs/workflow_runs.jsonl`；
- 桌面 session 日志；
- 完整错误文本；
- Python、PyTorch、CUDA 和操作系统信息；
- 修改前后的配置和 checkpoint。
