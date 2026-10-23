# MFP-Nova

独立的治疗肽多标签预测科研项目。当前专注 **`data/therapeutic_2025/`**，任务为 `major15` 和 `fine39`；同类论文的数据集仅作后续补充基准准备。

项目统一维护四个研究模型：**TriRelay、GeoFactor、ProtoRoute、HelixSet**。前三模型已完成两任务各五 Seed 的首轮训练及独立 Test；HelixSet 两层几何上下文版已完成 major15、seed42 的 Train/Val，当前四层几何上下文配置待运行。

## 四模型入口

| 模型 | 源码 | 配置 | 主要机制 |
|---|---|---|---|
| TriRelay | [models/trirelay.py](src/mfp_nova/models/trirelay.py) | [trirelay.yaml](configs/trirelay.yaml) | 多模态交互、标签反馈、共享/私有证据 |
| GeoFactor | [models/geofactor.py](src/mfp_nova/models/geofactor.py) | [geofactor.yaml](configs/geofactor.yaml) | 异构图与高阶因子推断 |
| ProtoRoute | [models/protoroute.py](src/mfp_nova/models/protoroute.py) | [protoroute.yaml](configs/protoroute.yaml) | 标签条件专家路由与多原型 |
| HelixSet | [models/helixset.py](src/mfp_nova/models/helixset.py) | [helixset.yaml](configs/helixset.yaml) | 四层几何上下文、质量调制残基对、多尺度平衡记忆与标签条件解码 |

四个模型共用 `research.py` 的训练、恢复、评价和推理流程，使用同一套数据加载与基础神经网络层。`sequence` 仍是独立的轻量对照基线。

- 单次训练：`mfp_nova.cli train --model` 支持上述四个名称，`--task` 选择 `major15/fine39`。
- 评价与推理：统一使用 `evaluate`、`predict`、`summarize`。
- 多 Seed：`mfp_nova.experiments` 与 `scripts/run_multiseed.ps1` 已纳入四模型；`all` 为四模型 × 两任务 × 五 Seed，共 40 组候选任务。运行脚本才会执行，文件整理不会启动训练。
- GPU 诊断：`gpu-check --models` 支持四模型，HelixSet 的关键模块检查点已列入源码；本轮没有执行诊断。

四模型都支持 `--metric-suite traditional5|task_specific`。HelixSet 当前默认传统五指标，前三模型保留原 AP 选模协议；批量计划会记录模式。跨模型正式比较时须明确统一的选模和阈值协议，不能将不同模式混成同一组结果。

原三模型的历史五 Seed 批量设置仍为 FP32、batch64、无累积；HelixSet 批量设置沿用其配置的 FP32、microbatch16 × 累积4、EMA0.999。两层几何上下文版在 RTX 5070、batch32×累积2 下峰值 allocated 为7.49 GiB、EMA阶段约249.7秒/轮；当前四层版的资源与性能待实测。

## 项目结构

```text
configs/                      # 四模型、序列基线及设备环境配置
src/mfp_nova/
  models/                     # 四个研究模型，各自独立实现
  research_layers.py          # 共享网络层与研究模型基础损失
  research.py                 # 四模型共享训练/恢复/评价/预测
  data.py, multimodal.py      # 只读数据、模态输入与 Train 统计
  encoder.py                  # 冻结 ESM 与残基特征缓存
  metrics.py, metric_protocol.py
  cli.py, experiments.py      # 命令行与固定 Seed 批量入口
  model.py, trainer.py         # 轻量 sequence 对照
scripts/                      # 训练辅助、结果导出及论文数据整理
tests/                        # 相关正确性检查；本轮未运行
docs/
  MFP-Next-Generation-Architecture.md
  MFP-Metrics-Guide.md         # 指标与 Loss 定义
  MFP-Real-Data-GPU-Validation.md
  experiment-history.md       # 原 README 的完整实验与开发记录
  references/                 # 同类论文包、原数据论文
  legacy/                     # 历史 TriFusion 技术蓝图
data/
  therapeutic_2025/           # 当前主数据集，只读
  archives/                   # 迁移/数据压缩包
  literature_benchmarks/      # 按论文保存的原始下载数据
  literature_consolidated/    # 按数据集归并的备用基准
features/                     # ESM 缓存，不提交 Git
runs/                         # 训练输出、权重和分析，不提交 Git
```

## 当前数据与实验约定

默认使用原样本、明确骨架队列，保留既有修复划分和监督 mask：

| 任务 | 标签数 | Train | Validation | Test |
|---|---:|---:|---:|---:|
| major15 | 15 | 39,478 | 8,534 | 8,187 |
| fine39 | 39 | 39,455 | 8,529 | 8,180 |

0 表示未标注，fine39 的 `__unknown` 表示具体亚类未知。Train 学参数，Validation 选模型与阈值，Test 作独立最终评价。主数据、既有 ESM 缓存与历史运行目录不因源码归类而重建。

环境按设备适配：V100 保留 Conda 克隆环境中的 PyTorch 2.4.1+cu121；RTX 5070 使用 [environment-rtx5070.yaml](configs/environment-rtx5070.yaml) 对应的 Blackwell 构建。主要依赖和实验协议保持一致。

## 文档导航

- [模型架构、四模型差异及项目结果](docs/MFP-Next-Generation-Architecture.md)：HelixSet 详见第 9 节。
- [指标与 Loss 说明](docs/MFP-Metrics-Guide.md)：包括四模型监督目标及两种评价模式。
- [实验与开发记录](docs/experiment-history.md)：保留原 README 的安装、两平台命令、历次改进和结果；历史数字按当时实验范围解释。
- [真实 GPU 验证记录](docs/MFP-Real-Data-GPU-Validation.md)：已取得的检查与性能记录。
- [主数据统计](docs/data_summary.json)、[原始论文数据说明](data/literature_benchmarks/README.md)、[按数据集归并说明](data/literature_consolidated/README.md)。

HelixSet 当前采用 geometry 4 / pair 4 / latent 6 / decoder 3 层。2026-10-23 的两层版 major15、seed42 结果为 Val Acc63.49%、Macro-AP55.18%，第25轮EMA最佳、第40轮早停；第11轮Macro-AP最高57.44%。四层版用于检验几何上下文深度，以microbatch16×累积4从头训练，不能恢复两层版checkpoint进行深度对照。本次深度调整仅做静态检查，未启动训练或测试。
