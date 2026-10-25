# MFP-Nova

独立的治疗肽多标签预测科研项目。当前专注 **`data/therapeutic_2025/`**，任务为 `major15` 和 `fine39`；同类论文的数据集仅作后续补充基准准备。

项目统一维护四个研究模型：**TriRelay、GeoFactor、ProtoRoute、HelixSet**。前三模型已完成两任务各五 Seed 的首轮训练及独立 Test；HelixSet 原版、decay30、精简版及ESM尾部LoRA均已完成major15/seed42 Train/Val。2026-10-25当前推进完整主干上的早期类别取证、深浅融合与标注集合辅助监督；本轮仅静态检查，尚未执行模型测试或训练。

## 四模型入口

| 模型 | 源码 | 配置 | 主要机制 |
|---|---|---|---|
| TriRelay | [models/trirelay.py](src/mfp_nova/models/trirelay.py) | [trirelay.yaml](configs/trirelay.yaml) | 多模态交互、标签反馈、共享/私有证据 |
| GeoFactor | [models/geofactor.py](src/mfp_nova/models/geofactor.py) | [geofactor.yaml](configs/geofactor.yaml) | 异构图与高阶因子推断 |
| ProtoRoute | [models/protoroute.py](src/mfp_nova/models/protoroute.py) | [protoroute.yaml](configs/protoroute.yaml) | 标签条件专家路由与多原型 |
| HelixSet | [models/helixset.py](src/mfp_nova/models/helixset.py) | [helixset.yaml](configs/helixset.yaml) | 完整残基对/潜变量/集合解码，早期类别取证与深浅融合 |

四个模型共用 `research.py` 的训练、恢复、评价和推理流程，使用同一套数据加载与基础神经网络层。`sequence` 仍是独立的轻量对照基线。

- 单次训练：`mfp_nova.cli train --model` 支持上述四个名称，`--task` 选择 `major15/fine39`。
- 评价与推理：统一使用 `evaluate`、`predict`、`summarize`。
- 多 Seed：`mfp_nova.experiments` 与 `scripts/run_multiseed.ps1` 已纳入四模型；`all` 为四模型 × 两任务 × 五 Seed，共 40 组候选任务。运行脚本才会执行，文件整理不会启动训练。
- GPU 诊断：`gpu-check --models` 支持四模型，HelixSet 的关键模块检查点已列入源码；本轮没有执行诊断。

四模型都支持 `--metric-suite traditional5|task_specific`。HelixSet 当前默认传统五指标，前三模型保留原 AP 选模协议；批量计划会记录模式。跨模型正式比较时须明确统一的选模和阈值协议，不能将不同模式混成同一组结果。

原三模型的历史五 Seed 批量设置仍为 FP32、batch64、无累积；HelixSet 当前为 FP32、batch32 × 累积2、EMA0.999。RTX5070实测原版约7.73GiB/262秒每轮，ESM尾部LoRA约9.44GiB/421秒每轮；当前新增取证方案的显存和性能待实测，不能直接沿用历史峰值。

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

## HelixSet：完整主干、早期类别取证与集合监督

`756b707` ESM尾部LoRA真实实验完成29轮，传统最佳epoch14：Acc63.43%、AT53.39%、Macro-AP56.75%、Micro-AP72.64%；原版`bb3aca4`为63.87%、53.89%、56.90%、71.59%。最佳Macro-AP57.16%也未超过原版57.33%。LoRA每轮慢约61%，本次不继续扩大适配容量。2–16aa验证样本合计损失75条完全匹配，17–32aa增加33条，推动以下针对证据传递的整体改动；这些分组为探索证据，不能替代独立Test确认。

当前保留sequence3/property2/geometry4/pair4/latent6/decoder3、d384和32个latent。在几何/pair融合前，各类别分别读取序列编码器输出（包括修饰、手性和化学token）与理化tokens，按样本和类别融合。这份早期证据初始化深层解码，并通过共享类别打分器形成独立浅层logits。最终`logits = shallow + sigmoid(gate) * (deep - shallow)`；gate读取两路类别状态、长度、结构覆盖/置信度、理化覆盖和D残基比例，初始深浅各0.5。浅层分支不依赖显式结构输入，没有原始ESM直接分类头，也不硬编码短肽的路由方向。

损失保留ASL主项、0.2×潜变量辅助分类、0.1×中间解码监督，增加0.1×深浅两路ASL均值及0.05×逐样本标注集合soft-Jaccard，辅助项沿用5轮warmup。集合项对未标注预测使用0.3弱惩罚，fine39未知亚类沿用额外降权；它优化已观测标签集合，不是实际生物学功能数监督，不强行限定预测标签数，也不改变传统指标定义。

**下一轮从头训练，复用旧最终层缓存`features/esm2_650m`，不使用layer29缓存。** 默认冻结ESM，已有最终层Train/Val缓存时不需再加载或下载预训练权重。LoRA代码和旧checkpoint加载支持保留，旧LoRA实验恢复时仍按checkpoint保存配置使用layer29与预训练权重。精简版checkpoint仍须在`0bf2435`使用。

保持major15/seed42、FP32、batch32×累积2、LR2e-4、5轮warmup/100轮cosine、EMA0.999第6轮起、traditional5/global，耐心15轮。`--metric-suite task_specific`仍可切换；`best.pt`为主协议候选，`best_ap.pt`为传统模式下的AP候选，不混用两个checkpoint的指标。结果自动导出NPZ、逐类诊断和分组统计。当前预期参数约7333万，实际以运行metadata为准；测试代码已补充但尚未执行，不承诺指标提升。

### RTX 5070：PowerShell，前台训练与打包

在当前目录无运行中的训练时同步。提示符应为`PS`，Anaconda Prompt/CMD中先进入`powershell -NoLogo -NoProfile`。以下复用已完成的最终层缓存。

```powershell
Set-Location 'D:\Data\ChatGPT\mfp\MFP-Nova'
git switch main
if ($LASTEXITCODE -ne 0) { throw '分支切换失败。' }
git pull --ff-only origin main
if ($LASTEXITCODE -ne 0) { throw '代码同步失败，请查看错误。' }
git log -1 --oneline
$mfpPython = 'D:\App\Code\Anaconda\envs\MFP-Nova\python.exe'
$env:PYTHONPATH = Join-Path $PWD.Path 'src'
$env:CUDA_VISIBLE_DEVICES = '0'
$env:OMP_NUM_THREADS = '4'
$env:MKL_NUM_THREADS = '4'
$Revision = git rev-parse --short HEAD
$RunName = "helixset_major15_seed42_rtx5070_${Revision}_earlyset_b32_a2_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
New-Item -ItemType Directory -Path 'runs' -Force | Out-Null
$LogPath = Join-Path 'runs' "${RunName}.log"
Write-Host "本次实验：$RunName"
& $mfpPython -u -m mfp_nova.cli train --model helixset --config configs/helixset.yaml `
  --task major15 --esm 650m --seed 42 --device cuda:0 --precision fp32 `
  --batch-size 32 --accumulation-steps 2 --num-workers 0 --ema-decay 0.999 --ema-start-epoch 6 `
  --metric-suite traditional5 --threshold-strategy global --epochs 100 --name $RunName `
  2>&1 | Tee-Object -FilePath $LogPath
if ($LASTEXITCODE -ne 0) { throw '训练异常退出，请查看日志。' }

# 正常完成后打包；权重及特征缓存留在训练机器。
$RunDir = Join-Path 'runs' $RunName
if (-not (Test-Path -LiteralPath (Join-Path $RunDir 'results.json'))) { throw '未找到results.json。' }
$PackFiles = @(Get-ChildItem -LiteralPath $RunDir -File | Where-Object { $_.Extension -in '.json', '.jsonl', '.npz', '.log' } | Select-Object -ExpandProperty FullName)
$PackFiles += $LogPath
$ZipPath = Join-Path ([Environment]::GetFolderPath('Desktop')) "${RunName}_results_$(Get-Date -Format 'yyyyMMdd_HHmmss').zip"
Compress-Archive -LiteralPath $PackFiles -DestinationPath $ZipPath
Write-Host "验收包：$ZipPath"
```

若要先测显存，在上述环境设置后自行执行（该命令会运行模型与短训练）：

```powershell
$GpuCheckDir = Join-Path 'runs' "gpucheck_helixset_earlyset_b32_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
& $mfpPython -u -m mfp_nova.cli gpu-check --models helixset --tasks major15 --esm 650m `
  --feature-root features/esm2_650m --device cuda:0 --batch-sizes 32 --precisions fp32 `
  --samples 128 --steps 16 --output $GpuCheckDir `
  2>&1 | Tee-Object -FilePath "${GpuCheckDir}.log"
```

### V100：Ubuntu Bash，后台训练与打包

保持原PyTorch 2.4.1+cu121环境，使用28个可用逻辑CPU中的16个计算线程与4个加载进程。只在该checkout无运行中训练时同步；已启动的旧LoRA实验可以先完成，运行期间不更新源码。

```bash
cd ~/wuzhencheng/projects/MFP-Nova
# 此前仅手工将num_workers从0改成4：恢复这一项，后面由命令行指定4。
# 如还有其他本地改动，保留并正常处理Git提示，不使用强制重置。
sed -i -E 's/^([[:space:]]*num_workers:)[[:space:]]*4[[:space:]]*$/\1 0/' configs/helixset.yaml
git switch main || exit 1
git pull --ff-only origin main || exit 1
git log -1 --oneline
conda activate MFP-Nova
mfpPython="$(command -v python)"
export PYTHONPATH="$PWD/src"
gpu=0
threads=16
export CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS="$threads" MKL_NUM_THREADS="$threads"
revision="$(git rev-parse --short HEAD)"
run_name="helixset_major15_seed42_v100_${revision}_earlyset_b32_a2_$(date +%Y%m%d_%H%M%S)"
mkdir -p runs
nohup "$mfpPython" -u -m mfp_nova.cli train --model helixset --config configs/helixset.yaml \
  --task major15 --esm 650m --seed 42 --device cuda:0 --precision fp32 \
  --batch-size 32 --accumulation-steps 2 --num-workers 4 --ema-decay 0.999 --ema-start-epoch 6 \
  --metric-suite traditional5 --threshold-strategy global --epochs 100 --name "$run_name" \
  > "runs/${run_name}.log" 2>&1 < /dev/null &
echo "PID=$! Run=$run_name"
tail -n 40 -f "runs/${run_name}.log"
```

正常进入epoch后可关闭SSH；`Ctrl+C`仅结束日志查看。训练完成后打包；重新登录时先进入项目和Conda环境，把`run_name`恢复为启动时打印的完整名称。

```bash
python - "$run_name" <<'PY'
from datetime import datetime
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import sys
run = Path('runs') / sys.argv[1]
if not (run / 'results.json').is_file():
    raise SystemExit('未找到results.json，请确认训练是否正常完成。')
archive = run.parent / f'{run.name}_results_{datetime.now():%Y%m%d_%H%M%S}.zip'
with ZipFile(archive, 'x', ZIP_DEFLATED) as out:
    for path in sorted(run.iterdir()):
        if path.is_file() and path.suffix in ('.json', '.jsonl', '.npz', '.log'):
            out.write(path, f'{run.name}/{path.name}')
    log = run.parent / f'{run.name}.log'
    if log.is_file():
        out.write(log, f'{run.name}/console.log')
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

`decay30`已于2026-10-24完成：29轮、传统最佳14轮，Acc63.26%、AT52.60%、AF5.91%、MFP-F1 53.50%；相对原日程Acc63.87%、AT53.89%、AF5.79%、MFP-F1 54.38%退步。最佳Macro-AP为57.51%，原日程57.33%，小幅上涨主要由37个Val阳性的溶栓类别带动，其他14类平均AP由60.36%降至60.18%。相同epoch29的Train AP99.83%/Val AP52.16%/Val Loss0.11806，对照98.34%/54.80%/0.07361，未缓解过拟合。单seed不推出所有短日程均无效，但当前不采用decay30。随后精简版也未改善性能；随后ESM尾部适配也未超越原版传统指标；当前转为完整主干上的早期类别取证与集合监督，保持有效batch64，具体配置以上文为准。跨设备结果分开报告，主要对照沿用RTX5070。
