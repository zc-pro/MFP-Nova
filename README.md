# MFP-Nova

独立的治疗肽多标签预测科研项目。当前专注 **`data/therapeutic_2025/`**，任务为 `major15` 和 `fine39`；同类论文的数据集仅作后续补充基准准备。

项目统一维护四个研究模型：**TriRelay、GeoFactor、ProtoRoute、HelixSet**。前三模型已完成两任务各五 Seed 的首轮训练及独立 Test；HelixSet 两层、四层几何上下文版均已完成 major15、seed42 的 Train/Val。最新缺失值修复及诊断输出尚未训练验证。

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

原三模型的历史五 Seed 批量设置仍为 FP32、batch64、无累积；HelixSet 批量设置沿用其配置的 FP32、microbatch16 × 累积4、EMA0.999。两层版在 RTX 5070、batch32×累积2 下峰值 allocated 为7.49 GiB、EMA阶段约249.7秒/轮；四层版在batch16×累积4下为4.59 GiB、约438.5秒/轮。时间差也包含microbatch变化，不能完全归因于深度。

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

HelixSet 当前采用 geometry 4 / pair 4 / latent 6 / decoder 3 层。2026-10-23 的两层版 major15、seed42 结果为 Val Acc63.49%、Macro-AP55.18%，第25轮EMA最佳、第40轮早停；四层版为Acc64.05%、Macro-AP55.38%、AT54.28%，第27轮最佳、第42轮早停。全程最高Macro-AP分别为57.44%/57.55%，不支持继续加深作为优先方向。这些是缺失值修复前的Validation成绩。

## HelixSet 下一轮受控实验

先运行修复后的 `baseline`。新训练在Train统计中记录 `phys.zero_missing=true`，理化缺失值在训练、验证、外部输入推理时统一标准化后置零，mask保持不变。所有新训练的研究模型均使用此修复；旧checkpoint没有该字段，继续采用原来的填充值，便于复现。恢复旧训练不会获得这项修复，应从头训练建立新参照。

HelixSet默认保存主协议最优的 `best.pt`，同时保存另一目标的候选：traditional5主协议对应 `best_ap.pt`，task_specific主协议对应 `best_accuracy.pt`。额外候选不控制早停、不独立训练；两者均来自每轮实际用于主选择的online/EMA权重。`last.pt`保存候选状态用于恢复；旧checkpoint恢复后只能从恢复轮次开始追踪额外候选，不能补出历史AP最优权重。候选权重存于CPU，增加约一份模型权重的内存和checkpoint写盘开销；验收ZIP不包含这些权重。

主模型和额外候选自动输出 `*_validation.npz` 与 `*_diagnostics.json`，包含对齐的row_id、ID、cluster_id、标签、目标、概率、阈值，以及空预测、逐类TP/观测FP/FN和多功能漏检统计。`results.json`记录两者指标与分组，`history.jsonl`记录额外候选最佳轮次。无需再跑一次模型来生成这些新增输出；NPZ可直接交给已有 `mfp_nova.decisions` 进行验证集诊断。观测FP不表示实验确认无活性，验证阈值拟合结果不作为独立Test成绩。

所有对照保持major15、seed42、FP32、microbatch16×累积4、相同EMA/损失/主选模协议；每次只改变下表一项，不批量启动。先查看baseline，再决定执行哪个对照。

| Variant | 相对baseline的变化 | 解释 |
|---|---|---|
| `baseline` | 无 | 缺失值修复后的四层参照，cosine计划仍为100轮 |
| `no_structure` | `--ablate structure_evidence` | 同时关闭局部、全局及pair中的显式结构输入；保留序列派生理化特征，不是纯序列模型 |
| `no_pair_read` | `--ablate pair_interactions` | 关闭标签解码的显式pair状态贡献，保留端点证据与上游pair主干 |
| `no_triangles` | `--ablate triangles` | 关闭学习/几何两路三角更新，保留其他pair运算 |
| `latent3` | `--latent-depth 3` | 潜变量融合6层减为3层，其他深度不变 |
| `decay30` | `--lr-decay-epochs 30` | 最大100轮不变，cosine在30个调度周期后到达5%学习率下限并保持 |

结构与pair开关保留参数，不把关闭计算的模型宣称为等计算量或精简参数模型。`decay30`和`latent3`是待检验候选，不是已证实的改进；不改变未标注标签语义，不解冻ESM。

### RTX 5070：PowerShell

在RTX机器项目目录执行，确认已经同步本次代码。下面每次只运行选定的一组；完成后执行打包段。

```powershell
Set-Location 'D:\Data\ChatGPT\mfp\MFP-Nova'
$mfpPython = 'D:\App\Code\Anaconda\envs\MFP-Nova\python.exe'
$Variant = 'baseline'
$Variants = @{
  baseline = @()
  no_structure = @('--ablate', 'structure_evidence')
  no_pair_read = @('--ablate', 'pair_interactions')
  no_triangles = @('--ablate', 'triangles')
  latent3 = @('--latent-depth', '3')
  decay30 = @('--lr-decay-epochs', '30')
}
if (-not $Variants.ContainsKey($Variant)) { throw "Unknown variant: $Variant" }
$Extra = $Variants[$Variant]
$env:CUDA_VISIBLE_DEVICES = '0'
$env:OMP_NUM_THREADS = '4'
$env:MKL_NUM_THREADS = '4'
$RunName = "helixset_major15_seed42_rtx5070_${Variant}_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
& $mfpPython -u -m mfp_nova.cli train --model helixset --config configs/helixset.yaml `
  --task major15 --esm 650m --seed 42 --device cuda:0 --precision fp32 `
  --batch-size 16 --accumulation-steps 4 --ema-decay 0.999 --ema-start-epoch 6 `
  --metric-suite traditional5 --threshold-strategy global --epochs 100 --name $RunName @Extra
if ($LASTEXITCODE -ne 0) { throw 'Training failed; inspect the error before packaging.' }

# 完成后的验收包：JSON、JSONL、NPZ；checkpoint留在训练机器。
$RunDir = Join-Path 'runs' $RunName
$PackFiles = Get-ChildItem -LiteralPath $RunDir -File | Where-Object { $_.Extension -in '.json', '.jsonl', '.npz' }
$ZipPath = Join-Path ([Environment]::GetFolderPath('Desktop')) "${RunName}_results_$(Get-Date -Format 'yyyyMMdd_HHmmss').zip"
Compress-Archive -LiteralPath $PackFiles.FullName -DestinationPath $ZipPath
Write-Host $ZipPath
```

### V100：Ubuntu Bash

使用相同FP32、batch及协议，以隔离设备外的变化。GPU与CPU线程可选；`nohup`使训练在SSH断开后继续运行。下面不是自动多组训练。

```bash
cd ~/wuzhencheng/projects/MFP-Nova
conda activate MFP-Nova
mfpPython="$(command -v python)"
variant=baseline
case "$variant" in
  baseline) extra=() ;;
  no_structure) extra=(--ablate structure_evidence) ;;
  no_pair_read) extra=(--ablate pair_interactions) ;;
  no_triangles) extra=(--ablate triangles) ;;
  latent3) extra=(--latent-depth 3) ;;
  decay30) extra=(--lr-decay-epochs 30) ;;
  *) echo "Unknown variant: $variant"; exit 1 ;;
esac
gpu=0
threads=4
export CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS="$threads" MKL_NUM_THREADS="$threads"
run_name="helixset_major15_seed42_v100_${variant}_$(date +%Y%m%d_%H%M%S)"
mkdir -p runs
nohup "$mfpPython" -u -m mfp_nova.cli train --model helixset --config configs/helixset.yaml \
  --task major15 --esm 650m --seed 42 --device cuda:0 --precision fp32 \
  --batch-size 16 --accumulation-steps 4 --ema-decay 0.999 --ema-start-epoch 6 \
  --metric-suite traditional5 --threshold-strategy global --epochs 100 --name "$run_name" \
  "${extra[@]}" > "runs/${run_name}.log" 2>&1 < /dev/null &
echo "PID=$! Run=$run_name"
```

训练完成后打包；重新登录时先恢复 `run_name` 为上面打印的完整名称，在项目目录、MFP-Nova环境中执行：

```bash
python - "$run_name" <<'PY'
from datetime import datetime
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import sys
run = Path('runs') / sys.argv[1]
if not (run / 'results.json').is_file():
    raise SystemExit('results.json not found: training has not finalized')
archive = run.parent / f'{run.name}_results_{datetime.now():%Y%m%d_%H%M%S}.zip'
with ZipFile(archive, 'x', ZIP_DEFLATED) as out:
    for path in run.iterdir():
        if path.is_file() and path.suffix in ('.json', '.jsonl', '.npz'):
            out.write(path, path.name)
    log = run.with_suffix('.log')
    if log.is_file():
        out.write(log, 'console.log')
print(archive.resolve())
PY
```
