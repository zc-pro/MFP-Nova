# MFP-Nova

独立的治疗肽多标签预测科研项目。当前专注 **`data/therapeutic_2025/`**，任务为 `major15` 和 `fine39`；同类论文的数据集仅作后续补充基准准备。

项目统一维护四个研究模型：**TriRelay、GeoFactor、ProtoRoute、HelixSet**。前三模型已完成两任务各五 Seed 的首轮训练及独立 Test；HelixSet 原版、decay30及精简版均已完成major15/seed42 Train/Val。2026-10-25按性能优先恢复原版主干，加入ESM末四层LoRA；新方案尚未运行模型或训练验证。

## 四模型入口

| 模型 | 源码 | 配置 | 主要机制 |
|---|---|---|---|
| TriRelay | [models/trirelay.py](src/mfp_nova/models/trirelay.py) | [trirelay.yaml](configs/trirelay.yaml) | 多模态交互、标签反馈、共享/私有证据 |
| GeoFactor | [models/geofactor.py](src/mfp_nova/models/geofactor.py) | [geofactor.yaml](configs/geofactor.yaml) | 异构图与高阶因子推断 |
| ProtoRoute | [models/protoroute.py](src/mfp_nova/models/protoroute.py) | [protoroute.yaml](configs/protoroute.yaml) | 标签条件专家路由与多原型 |
| HelixSet | [models/helixset.py](src/mfp_nova/models/helixset.py) | [helixset.yaml](configs/helixset.yaml) | 原版多尺度片段、残基对/潜变量/集合解码，ESM末四层任务适配 |

四个模型共用 `research.py` 的训练、恢复、评价和推理流程，使用同一套数据加载与基础神经网络层。`sequence` 仍是独立的轻量对照基线。

- 单次训练：`mfp_nova.cli train --model` 支持上述四个名称，`--task` 选择 `major15/fine39`。
- 评价与推理：统一使用 `evaluate`、`predict`、`summarize`。
- 多 Seed：`mfp_nova.experiments` 与 `scripts/run_multiseed.ps1` 已纳入四模型；`all` 为四模型 × 两任务 × 五 Seed，共 40 组候选任务。运行脚本才会执行，文件整理不会启动训练。
- GPU 诊断：`gpu-check --models` 支持四模型，HelixSet 的关键模块检查点已列入源码；本轮没有执行诊断。

四模型都支持 `--metric-suite traditional5|task_specific`。HelixSet 当前默认传统五指标，前三模型保留原 AP 选模协议；批量计划会记录模式。跨模型正式比较时须明确统一的选模和阈值协议，不能将不同模式混成同一组结果。

原三模型的历史五 Seed 批量设置仍为 FP32、batch64、无累积；HelixSet 当前配置为 FP32、microbatch16 × 累积4、EMA0.999。原版在RTX5070/b32×a2下峰值allocated约7.73GiB；精简版为3.69GiB、约149秒/轮，但训练91轮总耗时224.7分钟，原版32轮为140.5分钟。ESM适配方案的显存和耗时尚未测量。

## 项目结构

```text
configs/                      # 四模型、序列基线及设备环境配置
src/mfp_nova/
  models/                     # 四个研究模型，各自独立实现
  research_layers.py          # 共享网络层与研究模型基础损失
  research.py                 # 四模型共享训练/恢复/评价/预测
  data.py, multimodal.py      # 只读数据、模态输入与 Train 统计
  encoder.py                  # 冻结 ESM 与残基特征缓存
  esm_tail.py                 # 冻结前缀缓存之后的 ESM LoRA 适配
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

`bb3aca4`原版在第17轮达到Val Acc63.87%、AT53.89%；`021734d` decay30为63.26%、52.60%。`0bf2435`精简版第76轮为63.45%、52.85%，最佳Macro-AP56.52%（第13轮）也低于原版57.33%（第12轮）；它没有解决过拟合，不作为性能主线。它的代码和权重仍在Git历史及原运行目录保留，评价需使用`0bf2435`版本。

## HelixSet 性能优先：原版主干与ESM任务适配

恢复`bb3aca4`原版sequence3/property2/geometry4/pair4/latent6/decoder3及其分类头、几何融合和辅助监督；不再沿着精简版逐层加回。ESM650M前29层保持冻结并一次缓存其输出，末四层Q/K/V及attention输出投影加入rank8/alpha16/dropout0.05的LoRA，与完整HelixSet共同学习。ESM预训练权重仍冻结，**序列表征会随LoRA任务学习而改变**；没有原始ESM直接分类旁路。

新缓存目录为`features/esm2_650m_layer29`，保留原最终层缓存。它包含未经过最终LayerNorm的第29层残基状态及CLS/EOS状态；训练时还原各条序列自己的EOS位置，执行末四层及原最终LayerNorm。不能从已有最终层缓存反推出中间层，也不能缺少CLS/EOS直接拼接使用。提取只读取序列，不使用标签；Train/Val先提取，Test按独立评价需要再提取。固定同一HuggingFace权重revision，两平台保持一致。

首轮从头训练，主干LR2e-4、LoRA LR1e-5，二者共用原5轮warmup/100轮cosine；FP32、batch16×累积4（有效64）、EMA0.999/第6轮起、ASL未标注权重0.3、direct0.2/branch0.1、traditional5/global。保留原数据划分和指标定义。**不使用精简版checkpoint恢复，不叠加decay30、伪标签或新损失。** 本次是原版任务适配方案比较，microbatch变化也影响随机轨迹，不能宣称全部变化由LoRA单独引起。

末四层只常驻其冻结预训练权重和LoRA，不将前29层放入训练图。LoRA按650M的1280维计算新增327,680个可训练参数。checkpoint保存HelixSet、LoRA和必要buffer，省略可按固定revision重新加载的冻结尾部参数；模型加载和EMA恢复处理这部分省略，不忽略其他缺失权重。首次运行需能访问本机HuggingFace缓存或下载对应预训练权重。

HelixSet默认保存主协议最优的 `best.pt`，同时保存另一目标的候选：traditional5主协议对应 `best_ap.pt`，task_specific主协议对应 `best_accuracy.pt`。额外候选不控制早停、不独立训练；两者均来自每轮实际用于主选择的online/EMA权重。`last.pt`保存候选状态用于恢复；旧checkpoint恢复后只能从恢复轮次开始追踪额外候选，不能补出历史AP最优权重。候选权重存于CPU，增加约一份模型权重的内存和checkpoint写盘开销；验收ZIP不包含这些权重。

主模型和额外候选自动输出 `*_validation.npz` 与 `*_diagnostics.json`，包含对齐的row_id、ID、cluster_id、标签、目标、概率、阈值，以及空预测、逐类TP/观测FP/FN和多功能漏检统计。`results.json`记录两者指标与分组，`history.jsonl`记录额外候选最佳轮次。无需再跑一次模型来生成这些新增输出；NPZ可直接交给已有 `mfp_nova.decisions` 进行验证集诊断。观测FP不表示实验确认无活性，验证阈值拟合结果不作为独立Test成绩。

两平台命令使用同一配置，结果分别记录。`--latent-depth`恢复为原版潜变量融合深度。本轮不启动消融，不改变未标注语义。ESM低秩适配有既有研究依据，不宣称其为原创；是否提升本数据成绩仍需实际训练。本机仅做静态检查，不运行特征提取、forward/backward或训练。

### RTX 5070：PowerShell

先进入PowerShell（提示符应包含`PS`，Anaconda Prompt/CMD中先执行`powershell -NoLogo -NoProfile`）。同步并首次提取新缓存：

```powershell
Set-Location 'D:\Data\ChatGPT\mfp\MFP-Nova'
git switch main
if ($LASTEXITCODE -ne 0) { throw '分支切换失败。' }
git pull --ff-only origin main
if ($LASTEXITCODE -ne 0) { throw 'Git同步失败，请处理后再训练。' }
git log -1 --oneline
$mfpPython = 'D:\App\Code\Anaconda\envs\MFP-Nova\python.exe'
$env:PYTHONPATH = Join-Path $PWD.Path 'src'
$Revision = git rev-parse --short HEAD
$env:CUDA_VISIBLE_DEVICES = '0'
$env:OMP_NUM_THREADS = '4'
$env:MKL_NUM_THREADS = '4'
& $mfpPython -u -m mfp_nova.cli features --config configs/helixset.yaml `
  --esm 650m --device cuda:0 --splits train val --batch-size 8
if ($LASTEXITCODE -ne 0) { throw '前缀特征提取失败，请查看错误。' }
```

缓存完整后运行下面的前台训练与自动打包；已有新缓存时跳过提取段，缓存不会静默覆盖。中断提取沿用现有`features --resume-from`，按最后完成行数及实际中断split继续。

```powershell
Set-Location 'D:\Data\ChatGPT\mfp\MFP-Nova'
$mfpPython = 'D:\App\Code\Anaconda\envs\MFP-Nova\python.exe'
$env:PYTHONPATH = Join-Path $PWD.Path 'src'
$env:CUDA_VISIBLE_DEVICES = '0'
$env:OMP_NUM_THREADS = '4'
$env:MKL_NUM_THREADS = '4'
$Revision = git rev-parse --short HEAD
$RunName = "helixset_major15_seed42_rtx5070_${Revision}_esmtail_b16_a4_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
New-Item -ItemType Directory -Path 'runs' -Force | Out-Null
$LogPath = Join-Path 'runs' "${RunName}.log"
& $mfpPython -u -m mfp_nova.cli train --model helixset --config configs/helixset.yaml `
  --task major15 --esm 650m --seed 42 --device cuda:0 --precision fp32 `
  --batch-size 16 --accumulation-steps 4 --ema-decay 0.999 --ema-start-epoch 6 `
  --metric-suite traditional5 --threshold-strategy global --epochs 100 --name $RunName `
  2>&1 | Tee-Object -FilePath $LogPath
if ($LASTEXITCODE -ne 0) { throw 'Training failed; inspect the error before packaging.' }

# 完成后的验收包：JSON、JSONL、NPZ；checkpoint留在训练机器。
$RunDir = Join-Path 'runs' $RunName
if (-not (Test-Path -LiteralPath (Join-Path $RunDir 'results.json'))) { throw '训练尚未正常完成，未找到results.json。' }
$PackFiles = @(Get-ChildItem -LiteralPath $RunDir -File | Where-Object { $_.Extension -in '.json', '.jsonl', '.npz' } | Select-Object -ExpandProperty FullName)
$PackFiles += $LogPath
$ZipPath = Join-Path ([Environment]::GetFolderPath('Desktop')) "${RunName}_results_$(Get-Date -Format 'yyyyMMdd_HHmmss').zip"
Compress-Archive -LiteralPath $PackFiles -DestinationPath $ZipPath
Write-Host $ZipPath
```

### V100：Ubuntu Bash

使用相同FP32、batch及协议，以隔离设备外的变化。GPU与CPU线程可选；`nohup`使训练在SSH断开后继续运行。下面不是自动多组训练。

```bash
cd ~/wuzhencheng/projects/MFP-Nova
git switch main || exit 1
git pull --ff-only origin main || exit 1
git log -1 --oneline
conda activate MFP-Nova
mfpPython="$(command -v python)"
revision="$(git rev-parse --short HEAD)"
gpu=0
threads=4
export CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS="$threads" MKL_NUM_THREADS="$threads"
# 首次提取；已有完整layer29缓存时跳过此条。提取期间保持SSH连接。
"$mfpPython" -u -m mfp_nova.cli features --config configs/helixset.yaml \
  --esm 650m --device cuda:0 --splits train val --batch-size 8 || exit 1

run_name="helixset_major15_seed42_v100_${revision}_esmtail_b16_a4_$(date +%Y%m%d_%H%M%S)"
mkdir -p runs
nohup "$mfpPython" -u -m mfp_nova.cli train --model helixset --config configs/helixset.yaml \
  --task major15 --esm 650m --seed 42 --device cuda:0 --precision fp32 \
  --batch-size 16 --accumulation-steps 4 --ema-decay 0.999 --ema-start-epoch 6 \
  --metric-suite traditional5 --threshold-strategy global --epochs 100 --name "$run_name" \
  > "runs/${run_name}.log" 2>&1 < /dev/null &
echo "PID=$! Run=$run_name"
tail -n 40 -f "runs/${run_name}.log"
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

## HelixSet 验证集阈值研究

对结果ZIP中的 `best_validation.npz` 和 `best_ap_validation.npz` 做CPU分析，不需要加载权重或安装PyTorch。也可传入一份NPZ。当前支持major15；按发布数据核对完整Val样本、标签和簇，拒绝Test输入。

```powershell
$env:PYTHONPATH = "$PWD\src"
python -m mfp_nova.threshold_study --predictions 'C:\path\results.zip' `
  --data-root data/therapeutic_2025 --output runs/analysis/helixset_thresholds
```

```bash
PYTHONPATH=src python -m mfp_nova.threshold_study --predictions /path/results.zip \
  --data-root data/therapeutic_2025 --output runs/analysis/helixset_thresholds
```

四种方法为全局Jaccard阈值、逐类别F1阈值、向全局收缩的类别阈值、集合联合阈值。默认 `kappa=20`、最大偏移 `radius=0.10`；正例覆盖的序列簇数 `g` 决定收缩强度 `g/(g+20)`。联合方法在收缩范围内进行最多3轮单类别搜索，单类别停滞时尝试拟合集错误互换最多的6对类别，每维7个网格位置并保留当前值。按Acc、AT、Macro-F1、较小偏移依次比较，拟合集约束为Acc/AT/Macro-F1不低于全局、AF不高于全局。约束不保证留出数据不退步。所有方法使用相同的按簇两折划分，每折重新拟合全部阈值和收缩统计；没有根据留出折搜索超参数。

输出 `summary.csv`、每个checkpoint的JSON、联合阈值候选 `*_decision.json` 和交叉拟合预测。候选decision可供现有研究模型 `evaluate --decision` / `predict --decision` 显式使用，不修改checkpoint或默认训练策略。原始概率不变，因此AP不变。全Val拟合成绩有乐观性；交叉拟合只诊断新增阈值选择，checkpoint已经完整Val选轮，不能作为独立Test成绩。0仍为未标注。

2026-10-24，`bb3aca4`/major15/seed42/b32×a2，传统最佳epoch17、AP最佳epoch12。默认设置的簇交叉拟合结果（百分数）：

| checkpoint | 方法 | Acc↑ | AT↑ | Macro-F1↑ | AF↓ | MFP-F1↑ |
|---|---|---:|---:|---:|---:|---:|
| epoch17 | global | 63.79 | 53.37 | 55.33 | 5.89 | 54.90 |
| epoch17 | per_label | 62.77 | 50.90 | 56.11 | 6.05 | 53.86 |
| epoch17 | shrunk | 63.78 | 53.18 | 55.86 | 5.88 | 54.07 |
| epoch17 | joint | 64.27 | 53.81 | 55.92 | 5.78 | 53.37 |
| epoch12 | global | 62.71 | 51.44 | 54.50 | 6.03 | 52.90 |
| epoch12 | per_label | 62.44 | 49.63 | 55.82 | 6.13 | 53.02 |
| epoch12 | shrunk | 62.78 | 50.57 | 55.47 | 6.07 | 52.82 |
| epoch12 | joint | 63.36 | 52.04 | 55.30 | 5.92 | 52.83 |

联合方法在两个checkpoint均改善交叉拟合Acc/AT/Macro-F1/AF，但epoch17的MFP-F1下降约1.53个百分点；尚未实现所有目标共同改善，不替换默认global。此次没有接受任何类别对联合更新，因此尚无类别对搜索的收益证据。原始分析文件位于 `runs/analysis/helixset_thresholds_bb3aca4_20261024/`，不提交运行输出。

后续探索增加 `--protect-functions`：在原拟合约束上，要求独立多功能识别F1、至少恢复两个已标注独立功能的比例均不低于全局基准。抗微生物通用标签与具体抗菌/抗真菌/抗寄生虫/抗病毒证据按项目定义去重，不能用简单标签数代替。该选项输出 `joint_functions`，其余搜索设置保持相同，不更改模型训练。使用新的输出目录，例如 `runs/analysis/helixset_thresholds_functions_bb3aca4_20261024`。

| checkpoint | 方法 | Acc↑ | AT↑ | Macro-F1↑ | AF↓ | MFP-F1↑ | 恢复≥2真实功能的条数↑ |
|---|---|---:|---:|---:|---:|---:|---:|
| epoch17 | global | 63.79 | 53.37 | 55.33 | 5.89 | 54.90 | 973 |
| epoch17 | joint_functions | 64.15 | 53.27 | 55.82 | 5.87 | 53.84 | 940 |
| epoch12 | global | 62.71 | 51.44 | 54.50 | 6.03 | 52.90 | 941 |
| epoch12 | joint_functions | 63.22 | 51.54 | 55.42 | 6.02 | 53.43 | 937 |

以上为同一Val的簇交叉拟合探索结果。新约束是在查看上一项研究结果后提出，不能称为全新盲验证；拟合集满足约束不代表留出集满足。epoch17仍损害多功能表现；epoch12虽改善多项指标，真实多功能恢复仍减少4条。因此不默认采用、不继续在该Val上叠加约束调优，也不凭本轮分析改选checkpoint。

`decay30`已于2026-10-24完成：29轮、传统最佳14轮，Acc63.26%、AT52.60%、AF5.91%、MFP-F1 53.50%；相对原日程Acc63.87%、AT53.89%、AF5.79%、MFP-F1 54.38%退步。最佳Macro-AP为57.51%，原日程57.33%，小幅上涨主要由37个Val阳性的溶栓类别带动，其他14类平均AP由60.36%降至60.18%。相同epoch29的Train AP99.83%/Val AP52.16%/Val Loss0.11806，对照98.34%/54.80%/0.07361，未缓解过拟合。单seed不推出所有短日程均无效，但当前不采用decay30。随后精简版也未改善性能；当前转为原版主干加ESM尾部适配，保持有效batch64，具体配置以上文为准。跨设备结果分开报告，主要对照沿用RTX5070。
