# MFP-Nova

独立的治疗肽多标签预测科研项目。主数据 **`data/therapeutic_2025/`** 提供 `major15` 和 `fine39`；新增 `therapep49` 使用独立目录 **`data/therapep_ai/`** 中 TheraPep-AI 作者发布的 Train/Val/Test 与49维标签，作为同论文数据口径的补充比较。

项目统一维护五个研究模型：**TriRelay、GeoFactor、ProtoRoute、HelixSet、TheraSet**。前三模型已完成两任务各五 Seed 的首轮训练及独立 Test；HelixSet 原版、decay30、精简版、ESM尾部LoRA及早期取证方案均已完成major15/seed42 Train/Val。双表征R-Drop版已完成major15/seed42真实训练，传统最佳Val Acc64.96%/AT54.73%，AP候选Macro-AP58.43%。2026-10-26暂选此架构为汇报候选，准备fine39独立训练及两任务Test；实际Test尚未执行，命令见本文末节。TheraSet 于2026-10-28完成两任务seed42完整Train/Val：major15 Acc65.08%/AT54.51%，fine39 Acc59.58%/AT47.77%；复算与训练记录一致，尚无独立Test或多Seed结论，详见[实验记录](docs/experiment-history.md#2026-10-28theraset-两任务-seed42-正式-trainval-结果)。

## 五模型入口

| 模型 | 源码 | 配置 | 主要机制 |
|---|---|---|---|
| TriRelay | [models/trirelay.py](src/mfp_nova/models/trirelay.py) | [trirelay.yaml](configs/trirelay.yaml) | 多模态交互、标签反馈、共享/私有证据 |
| GeoFactor | [models/geofactor.py](src/mfp_nova/models/geofactor.py) | [geofactor.yaml](configs/geofactor.yaml) | 异构图与高阶因子推断 |
| ProtoRoute | [models/protoroute.py](src/mfp_nova/models/protoroute.py) | [protoroute.yaml](configs/protoroute.yaml) | 标签条件专家路由与多原型 |
| HelixSet | [models/helixset.py](src/mfp_nova/models/helixset.py) | [helixset.yaml](configs/helixset.yaml) | 双冻结序列表征、理化/结构证据与完整残基对/潜变量/集合解码 |
| TheraSet | [models/theraset.py](src/mfp_nova/models/theraset.py) | [theraset.yaml](configs/theraset.yaml) | TheraPep式两级CNN直接预测、共享多原型尺度/模态/pair读取与标签条件残差修正 |

五个模型共用 `research.py` 的训练、恢复、评价和推理流程，使用同一套数据加载与基础神经网络层。`sequence` 仍是独立的轻量对照基线。

- 单次训练：`mfp_nova.cli train --model` 支持上述五个名称，`--task` 选择 `major15/fine39/therapep49/therapep49_positive/therapep48`。
- 评价与推理：统一使用 `evaluate`、`predict`、`summarize`。
- 多 Seed：`mfp_nova.experiments` 与 `scripts/run_multiseed.ps1` 已纳入五模型；`all` 仍为五模型 × 原两任务 × 五 Seed，共 50 组候选任务。新基准通过 `--task therapep49` 显式选择。运行脚本才会执行。
- GPU 诊断：`gpu-check --models` 支持五模型，HelixSet/TheraSet 的关键模块检查点已列入源码；本轮仅做本地 CPU 检查。

五模型都支持 `--metric-suite traditional5|task_specific`。HelixSet/TheraSet 当前默认传统五指标，前三模型保留原 AP 选模协议；批量计划会记录模式。跨模型正式比较时须明确统一的选模和阈值协议，不能将不同模式混成同一组结果。

原三模型的历史五 Seed 批量设置仍为 FP32、batch64、无累积；HelixSet 当前为 FP32、batch16 × 累积4、EMA0.999，启用R-Drop后仍为64条不同样本/更新。双表征单路训练实测7.76GiB/263秒每轮，R-Drop双路16×4实测7.78GiB/763秒每轮；fine39仍需检查显存。

## 项目结构

```text
configs/                      # 五模型、序列基线及设备环境配置
src/mfp_nova/
  models/                     # 五个研究模型，各自独立实现
  research_layers.py          # 共享网络层与研究模型基础损失
  research.py                 # 五模型共享训练/恢复/评价/预测
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
| therapep49（独立作者发布基准） | 48功能＋1全局标志 | 131,947 | 16,457 | 16,537 |
| therapep49_positive（所有split去除合成背景） | 48功能＋恒正治疗标志 | 44,524 | 5,565 | 5,566 |
| therapep48（正样本纯功能） | 原前48功能列 | 44,524 | 5,565 | 5,566 |

0 表示未标注，fine39 的 `__unknown` 表示具体亚类未知。Train 学参数，Validation 选模型与阈值，Test 作独立最终评价。主数据、既有 ESM 缓存与历史运行目录不因源码归类而重建。

环境按设备适配：V100 保留 Conda 克隆环境中的 PyTorch 2.4.1+cu121；RTX 5070 使用 [environment-rtx5070.yaml](configs/environment-rtx5070.yaml) 对应的 Blackwell 构建。主要依赖和实验协议保持一致。

`therapep49` 保留作者 CSV 成员、49位标签顺序及合成背景，包含55,655条真实/恢复正样本与109,286条合成背景；采用作者发布划分，不具备主基准的簇隔离口径。运行 `python -m mfp_nova.cli download-therapep` 下载原CSV；`features --config configs/theraset.yaml --track therapep49` 创建独立缓存，`train --model theraset --task therapep49` 默认 Micro-F1/固定0.5选模。新基准对所有来源统一使用序列与ESM，避免仅正样本有结构/属性造成背景标签泄漏；原两任务保留完整模态。数据、缓存路径、两平台准备与训练命令见 [TheraSet文档](docs/TheraSet-Architecture.md#therapep-ai-作者发布基准therapep49)。

### HelixSet：三个TheraPep数据版本、两套选模协议

`python -m mfp_nova.cli prepare-therapep-variants` 从原CSV生成 `data/therapep49_positive/` 和 `data/therapep48/`，保留各split内相对顺序、重复记录与原始row_id/ID；已有目录直接读取，不覆盖。48维版本同时去掉CSV的 `is_therapeutic` 和标签向量最后一位。它不将最后的 `Tumor_homing` 当治疗标志；功能0仍按未标注处理。三个任务都采用相同的序列输入范围、双ESM、HelixSet、R-Drop/EMA。正样本版本通过原row_id/ID/骨架对齐复用 `features/therapep_ai/` 完整缓存，不复制缓存或重新编码。

RTX5070使用 [run_helixset_therapep.ps1](scripts/run_helixset_therapep.ps1)。默认三个任务×两种协议，共6次独立训练；FP32、seed42、batch16×累积4、EMA0.999、R-Drop0.1、100轮上限、patience15。`traditional5` 以Validation拟合全局阈值的样本Jaccard选模，AT打破平局；`micro_f1` 以Validation Micro-F1@0.5选模，阈值固定0.5。两种协议均输出完整传统五指标及AP/F1，不混取不同checkpoint的单项最佳值。

```powershell
Set-Location 'D:\Data\ChatGPT\mfp\MFP-Nova'
git pull --ff-only origin main
if ($LASTEXITCODE -ne 0) { throw '代码同步失败。' }
$RunRoot = Join-Path $PWD.Path 'runs/helixset_therapep3_seed42'
& .\scripts\run_helixset_therapep.ps1 -Action prepare -Output $RunRoot
& .\scripts\run_helixset_therapep.ps1 -Action train -Output $RunRoot
# 所有训练完成后，评价各自best.pt的Val/Test，Test不重新调阈值
& .\scripts\run_helixset_therapep.ps1 -Action evaluate -Output $RunRoot
& .\scripts\run_helixset_therapep.ps1 -Action pack -Output $RunRoot
```

RTX前台执行，请保持终端打开。训练时按Ctrl+C退出后，重复同一条 `train` 命令即可：跳过完成任务，从未完成任务的last.pt原位续训，日志追加；不新建resume目录。恢复模型、优化器、学习率日程、EMA、随机数状态和早停计数，history按checkpoint恢复。保存以完整epoch为单位，未完成的epoch会重跑，不是从中断的microbatch续跑；第1轮尚未保存时在原目录重跑第1轮。last.pt先写临时文件再替换，写入中断保留上一份完整checkpoint。

脚本默认固定Output为项目目录下的 `runs/helixset_therapep3_seed42`，可直接重复 `& .\scripts\run_helixset_therapep.ps1 -Action train`。相对Output按脚本所在项目的PowerShell路径解析，不使用可能仍指向C盘用户目录的.NET进程当前目录；启动时打印绝对Output。如果此前已使用带时间戳的Output，必须继续显式传入原路径，不改用默认目录，也不重新生成时间戳。仅跑一个协议时，给每条脚本命令加 `-Suites traditional5` 或 `-Suites micro_f1`；仅跑一个任务用 `-Tasks therapep48`。`prepare`提取缺少的Train/Val缓存，`evaluate`才提取缺少的Test缓存；缓存中断须按完成行数使用features的 `--resume-from`，不覆盖缓存目录。

新终端不会保留 `$RunRoot`、`$Common`。恢复固定批次建议直接使用完整路径：`& .\scripts\run_helixset_therapep.ps1 -Action train -Output 'D:\Data\ChatGPT\mfp\MFP-Nova\runs\helixset_therapep3_seed42'`。空Output或项目根目录会明确报错，禁止退化为在项目根目录重新建任务。启动时应显示 `Resume existing checkpoint: .../last.pt`；只显示另一目录的 `New experiment` 不代表恢复旧进度。历史误启动的根目录任务不会被自动移动、删除或合并。

打包文件在桌面，包含配置元数据、训练记录、Val/Test预测、日志和 `summary.csv`；不包含checkpoint或ESM缓存。汇总同时列出整体输出与正肽48功能指标。完整therapep49的空背景Jaccard计0，Val整体Accuracy上限约33.82%；正样本49维版含恒正治疗标志，整体指标包含其贡献，48维版无该列。三种不同队列的整体数字不能直接作净增益比较，应在共同正肽48功能视图下分析，且注明各自选模协议。当前变更只做静态和数据对齐检查，尚未执行这6次真实GPU训练。

V100使用 [run_helixset_therapep.sh](scripts/run_helixset_therapep.sh)，保持上述模型、seed、batch、R-Drop/EMA、选模协议不变；每任务6个CPU线程、2个DataLoader进程，可通过 `CPU_THREADS`、`NUM_WORKERS` 修改。三任务各占一张用户分配的GPU，不是DDP。固定目录 `runs/helixset_therapep3_seed42_v100/<task>/<suite>/`，重复同一命令跳过完成实验、原位续训，未完成epoch重跑。保留Conda MFP-Nova的PyTorch2.4.1+cu121，不升级环境。

```bash
cd ~/wuzhencheng/projects/MFP-Nova
conda activate MFP-Nova
git pull --ff-only origin main
# 确认同步成功后继续；共用缓存仅准备一次，完成后再启动三个队列
export PYTHON_BIN="$(command -v python)"
export CPU_THREADS=6 NUM_WORKERS=2
export RUN_ROOT="$PWD/runs/helixset_therapep3_seed42_v100"
mkdir -p "$RUN_ROOT"
GPU=0 bash scripts/run_helixset_therapep.sh prepare

nohup env GPU=0 bash scripts/run_helixset_therapep.sh train therapep49 \
  >> "$RUN_ROOT/therapep49.log" 2>&1 < /dev/null &
echo "therapep49 PID=$!"
nohup env GPU=1 bash scripts/run_helixset_therapep.sh train therapep49_positive \
  >> "$RUN_ROOT/therapep49_positive.log" 2>&1 < /dev/null &
echo "therapep49_positive PID=$!"
nohup env GPU=2 bash scripts/run_helixset_therapep.sh train therapep48 \
  >> "$RUN_ROOT/therapep48.log" 2>&1 < /dev/null &
echo "therapep48 PID=$!"
```

每个队列顺序执行traditional5和micro_f1两次独立训练，SSH断开后继续；GPU号按实际分配替换，映射后内部统一cuda:0。`tail -n 30 -f "$RUN_ROOT/therapep49.log"` 查看进度，Ctrl+C只退出日志查看。后台任务可用 `touch "$RUN_ROOT/therapep49/PAUSE"` 请求本轮结束后暂停；退出后 `rm "$RUN_ROOT/therapep49/PAUSE"`，重复该任务的nohup命令恢复。另两任务替换对应任务名，不能在进程尚未退出时启动同目录的另一个实例。仅运行一种协议时先 `export SUITES=micro_f1` 或 `export SUITES=traditional5`，评价和打包沿用同一SUITES。

CUDA训练用多个DataLoader worker时启用页锁定内存与异步输入复制；验证标签保存独立数组，避免长验证集持有worker共享张量的文件描述符（Linux `received 0 items of ancdata`）。已有V100训练可在完整epoch暂停、同步后，用同一目录和 `NUM_WORKERS=2` 恢复；worker数是执行参数，不改变checkpoint中的batch、精度或选模协议。每轮 `Timing` 的 `Train` 包含 `DataWait`（主进程取下一批的等待），`Val` 包含在线/EMA验证和分组指标；用相同R-Drop阶段的轮次比较速度。

所有训练完成后，共用Test缓存仅准备一次，再分GPU评价并合并结果。打包逻辑由两平台共用的 [pack_helixset_therapep.py](scripts/pack_helixset_therapep.py) 实现，不包含权重/缓存。

```bash
GPU=0 bash scripts/run_helixset_therapep.sh prepare-test
nohup env GPU=0 bash scripts/run_helixset_therapep.sh evaluate therapep49 \
  >> "$RUN_ROOT/therapep49_evaluate.log" 2>&1 < /dev/null &
nohup env GPU=1 bash scripts/run_helixset_therapep.sh evaluate therapep49_positive \
  >> "$RUN_ROOT/therapep49_positive_evaluate.log" 2>&1 < /dev/null &
nohup env GPU=2 bash scripts/run_helixset_therapep.sh evaluate therapep48 \
  >> "$RUN_ROOT/therapep48_evaluate.log" 2>&1 < /dev/null &
wait
# 确认各任务评价成功完成后执行，缺少结果会明确报错
bash scripts/run_helixset_therapep.sh pack
```

合并ZIP保存在固定RUN_ROOT目录内，包含6组Val/Test指标、预测、训练记录、日志与summary.csv。重新登录后恢复PYTHON_BIN、CPU_THREADS、NUM_WORKERS和同一个RUN_ROOT即可，不生成新时间戳运行目录。

## 文档导航

- [第五模型 TheraSet 架构设计分析](docs/TheraSet-Architecture.md)：基线定义、四模型取舍、创新假设、公式与维度、消融、验证及 RTX 5070/V100 训练命令。
- [同类论文模型重训练可行性分析](docs/literature-model-comparison-feasibility.md)：保留各论文处理与训练方法，在当前数据上比较；包括仓库缺口、候选优先级和实施建议。
- [模型架构、四模型差异及项目结果](docs/MFP-Next-Generation-Architecture.md)：HelixSet 详见第 9 节。
- [指标与 Loss 说明](docs/MFP-Metrics-Guide.md)：包括四模型监督目标及两种评价模式。
- [实验与开发记录](docs/experiment-history.md)：保留原 README 的安装、两平台命令、历次改进和结果；历史数字按当时实验范围解释。
- [真实 GPU 验证记录](docs/MFP-Real-Data-GPU-Validation.md)：已取得的检查与性能记录。
- [主数据统计](docs/data_summary.json)、[原始论文数据说明](data/literature_benchmarks/README.md)、[按数据集归并说明](data/literature_consolidated/README.md)。

## 第五模型 TheraSet

TheraSet 保留 TheraPep-AI 的 34 维残基编码、两级 3/5/7/9 CNN 与直接分类头，新增共享多原型查询、独立多源证据读取、因子化残基对及标签条件残差。major15/fine39 下游可训练参数分别为 10,376,854 / 10,404,206，另有约 800M 冻结编码器。默认双 ESM、batch16×累积4、FP32、R-Drop 和 EMA，与当前 HelixSet 的主要训练协议对齐。

`train --model theraset` 使用新配置；`--ablate evidence_decoder chemical_evidence` 是适配后的纯 CNN 对照，不能称为原论文完整复现。两任务已完成RTX5070/FP32/batch64×累积1/seed42完整Train/Val，均为EMA候选，39/33轮正常早停；主候选epoch24/18，阈值0.74/0.68，显存峰值4.20/7.50GiB。major15相对HelixSet的Jaccard差异仅+0.11个百分点，按簇配对bootstrap区间跨零；主要已证实收益是更低成本，尚未完成TheraPep适配CNN对照与模块消融。完整设计与两平台命令见 [TheraSet 文档](docs/TheraSet-Architecture.md)。

## HelixSet：双冻结表征与Bernoulli R-Drop

`bb3aca4`原版在第17轮达到Val Acc63.87%、AT53.89%；`021734d` decay30为63.26%、52.60%。`0bf2435`精简版第76轮为63.45%、52.85%，最佳Macro-AP56.52%（第13轮）也低于原版57.33%（第12轮）；它没有解决过拟合，不作为性能主线。它的代码和权重仍在Git历史及原运行目录保留，评价需使用`0bf2435`版本。

`756b707` ESM尾部LoRA真实实验完成29轮，传统最佳epoch14：Acc63.43%、AT53.39%、Macro-AP56.75%、Micro-AP72.64%；原版`bb3aca4`为63.87%、53.89%、56.90%、71.59%。最佳Macro-AP57.16%也未超过原版57.33%。LoRA每轮慢约61%，本次不继续扩大适配容量。2–16aa验证样本合计损失75条完全匹配，17–32aa增加33条，推动以下针对证据传递的整体改动；这些分组为探索证据，不能替代独立Test确认。

`a97957d`早期取证方案完成38轮、174.7分钟，传统最佳epoch23：Acc63.77%、AT53.01%，最佳Macro-AP56.75%（epoch10），未超过原版。保留该实现以加载旧checkpoint，当前默认关闭`evidence_readout`，恢复原类别查询；`evidence`及`annotation_set`损失权重归零。

当前保留sequence3/property2/geometry4/pair4/latent6/decoder3、d384和32个latent。新增[短肽ESM-150M-F](https://huggingface.co/jiahuizhang/esm-150m-peptide-fine-tune)冻结最终层表征：每个残基的650M/150M输出分别做无参数LayerNorm，拼接1280+640维后投影到384，再进入同一套序列、理化、结构及集合解码主干。没有额外ESM直接分类头，也没有复制主干。相较原完整模型增加245,760个投影权重，预计71,723,550个可训练参数，以实际metadata为准。

`d26f60c`已完成62轮，传统最佳47轮Acc64.50%、AT54.94%、AF5.82%；AP最佳12轮Macro-AP57.84%/Micro-AP73.07%。第12→62轮Train AP88.90%→99.79%、Val AP57.84%→51.76%，因此下一轮针对泛化增加R-Drop。上述均为单seed Val结果，独立Test尚未评价。

**下一轮从头训练。** 复用`features/esm2_650m`最终层Train/Val缓存，并新增`features/esm2_peptide_150m`；不使用layer29缓存。新编码器使用自己的固定revision，首次提取需下载约150M模型权重，但正式训练只读取两份缓存。两份缓存独立核对row_id、ID和骨架序列；双视图metadata及维度写入checkpoint。旧完整模型/LoRA/早期取证checkpoint仍按保存参数加载，精简版仍须在`0bf2435`使用。

保持major15/seed42、FP32、LR2e-4、5轮warmup/100轮cosine、EMA0.999第6轮起、traditional5/global，耐心15轮。默认改为batch16×累积4。第1–5轮仍单路监督；第6–10轮R-Drop权重0.02→0.1，之后0.1。同一输入两次独立网络dropout，ASL主项＋0.2×潜变量辅助分类＋0.1×中间解码监督取两路均值，再加逐标签Bernoulli对称KL；未标注权重仍为0.3。Train指标取第一路预测，Val为单次干净eval与主ASL，原始R和权重写入history。`--metric-suite task_specific`仍可切换；传统默认保存best.pt及best_ap.pt两个候选。参数量及推理计算不变，双路训练耗时/显存待实测，必须重新测显存；本机不执行。

`features --encoder-view primary|peptide`分别提取两份缓存；`--encoder-view peptide`不要同时传`--esm`。训练默认读取配置中的两份缓存，`--feature-root`和`--peptide-feature-root`可改路径。纯序列`predict`按checkpoint加载两套编码器，不要求输入结构文件；外部NPZ接口须额外提供`esm_peptide[B,L,640]`，归一化由模型完成。独立Test需要时再分别提取Test缓存，训练命令不访问Test。

### RTX 5070：PowerShell，前台训练与打包

在当前目录无运行中的训练时同步。提示符应为`PS`，Anaconda Prompt/CMD中先进入`powershell -NoLogo -NoProfile`。以下跳过已完成的缓存，先提取缺少的Train/Val表征再前台训练。若提取中断，同一view/split按日志使用`--resume-from`续提；不要删除原缓存或把不完整目录视为完成。

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
foreach ($View in @('primary', 'peptide')) {
  $FeatureRoot = if ($View -eq 'primary') { 'features/esm2_650m' } else { 'features/esm2_peptide_150m' }
  foreach ($Split in @('train', 'val')) {
    if (Test-Path -LiteralPath "$FeatureRoot/$Split/metadata.json") { continue }
    & $mfpPython -u -m mfp_nova.cli features --config configs/helixset.yaml `
      --encoder-view $View --splits $Split --device cuda:0 --batch-size 8
    if ($LASTEXITCODE -ne 0) { throw "特征提取失败：$View/$Split，请查看错误；中断缓存按实际行数续提。" }
  }
}
```

先完成上述环境及缓存准备，然后在正式训练前执行以下显存检查。profile直接启用完整R-Drop权重0.1与两份激活图；第1–5轮单路训练的显存不能代表正式双路峰值。该命令会执行真实前后向与短训练，本机未执行：

```powershell
$GpuCheckDir = Join-Path 'runs' "gpucheck_helixset_dualpep_rdrop_b16_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
& $mfpPython -u -m mfp_nova.cli gpu-check --models helixset --tasks major15 --esm 650m `
  --feature-root features/esm2_650m --device cuda:0 --batch-sizes 16 --precisions fp32 `
  --peptide-feature-root features/esm2_peptide_150m `
  --samples 128 --steps 16 --output $GpuCheckDir `
  2>&1 | Tee-Object -FilePath "${GpuCheckDir}.log"
if ($LASTEXITCODE -ne 0) { throw "R-Drop显存检查未通过，请查看${GpuCheckDir}.log。" }
```

显存检查通过后，在同一个窗口前台训练并打包：

```powershell
$Revision = git rev-parse --short HEAD
$RunName = "helixset_major15_seed42_rtx5070_${Revision}_dualpep_rdrop_b16_a4_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
New-Item -ItemType Directory -Path 'runs' -Force | Out-Null
$LogPath = Join-Path 'runs' "${RunName}.log"
Write-Host "本次实验：$RunName"
& $mfpPython -u -m mfp_nova.cli train --model helixset --config configs/helixset.yaml `
  --task major15 --seed 42 --device cuda:0 --precision fp32 `
  --batch-size 16 --accumulation-steps 4 --num-workers 0 --ema-decay 0.999 --ema-start-epoch 6 `
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
run_name="helixset_major15_seed42_v100_${revision}_dualpep_rdrop_b16_a4_$(date +%Y%m%d_%H%M%S)"
mkdir -p runs
nohup bash -c '
set -e
mfpPython="$1"
run_name="$2"
for view in primary peptide; do
  if [ "$view" = primary ]; then feature_root=features/esm2_650m; else feature_root=features/esm2_peptide_150m; fi
  for split in train val; do
    if [ -f "$feature_root/$split/metadata.json" ]; then continue; fi
    "$mfpPython" -u -m mfp_nova.cli features --config configs/helixset.yaml \
      --encoder-view "$view" --splits "$split" --device cuda:0 --batch-size 8
  done
done
"$mfpPython" -u -m mfp_nova.cli gpu-check --models helixset --tasks major15 --esm 650m \
  --feature-root features/esm2_650m --peptide-feature-root features/esm2_peptide_150m \
  --device cuda:0 --batch-sizes 16 --precisions fp32 --samples 128 --steps 16 \
  --output "runs/${run_name}_gpucheck"
exec "$mfpPython" -u -m mfp_nova.cli train --model helixset --config configs/helixset.yaml \
  --task major15 --seed 42 --device cuda:0 --precision fp32 \
  --batch-size 16 --accumulation-steps 4 --num-workers 4 --ema-decay 0.999 --ema-start-epoch 6 \
  --metric-suite traditional5 --threshold-strategy global --epochs 100 --name "$run_name"
' -- "$mfpPython" "$run_name" \
  > "runs/${run_name}.log" 2>&1 < /dev/null &
echo "PID=$! Run=$run_name"
tail -n 40 -f "runs/${run_name}.log"
```

缓存提取与训练都在后台，确认日志正常后可关闭SSH；`Ctrl+C`仅结束日志查看。提取失败会停止，不会继续启动训练。训练完成后打包；重新登录时先进入项目和Conda环境，把`run_name`恢复为启动时打印的完整名称。

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

`decay30`已于2026-10-24完成：29轮、传统最佳14轮，Acc63.26%、AT52.60%、AF5.91%、MFP-F1 53.50%；相对原日程Acc63.87%、AT53.89%、AF5.79%、MFP-F1 54.38%退步。最佳Macro-AP为57.51%，原日程57.33%，小幅上涨主要由37个Val阳性的溶栓类别带动，其他14类平均AP由60.36%降至60.18%。相同epoch29的Train AP99.83%/Val AP52.16%/Val Loss0.11806，对照98.34%/54.80%/0.07361，未缓解过拟合。单seed不推出所有短日程均无效，但当前不采用decay30。随后精简版也未改善性能；随后ESM尾部适配也未超越原版传统指标；早期取证随后也未改善整体表现，当前转为完整主干上的双冻结序列表征，保持有效batch64，具体配置以上文为准。跨设备结果分开报告，主要对照沿用RTX5070。


## 汇报候选：major15与fine39独立Test（2026-10-26）

当前候选为`7505336`双冻结表征＋R-Drop。major15固定使用RTX5070恢复实验的`best.pt`：epoch18、EMA、Val阈值0.73、Val Acc64.96%/AT54.73%。fine39沿用架构和超参数，独立从头训练39类模型，使用自己的Val选出`best.pt`及阈值。只评价事先选定的传统候选，不按Test选择checkpoint、阈值或设备上的不同训练结果。

已按发布数组确认fine39各split均为major15对应队列的子集，可共用两套最终层ESM缓存。major15 Train/Val/Test=39478/8534/8187，fine39=39455/8529/8180。Test各提取一次，评价加载checkpoint保存的Train预处理统计与Val阈值，不重新拟合。以下操作需由用户在训练机器执行；本机没有权重，未启动训练或Test，实际Test结果尚未产生。

### RTX5070汇报命令：major15 Test、fine39训练/Test、统一打包

PowerShell前台执行。确保其他训练已结束；复用现有Train/Val缓存，不重新训练major15，不使用`best_ap.pt`替换已经选定的模型。若输出`test.json`已存在，保留已有结果，不覆盖重测。

```powershell
Set-Location 'D:\Data\ChatGPT\mfp\MFP-Nova'
$mfpPython = 'D:\App\Code\Anaconda\envs\MFP-Nova\python.exe'
$env:PYTHONPATH = Join-Path $PWD.Path 'src'
$env:CUDA_VISIBLE_DEVICES = '0'
$env:OMP_NUM_THREADS = '4'
$env:MKL_NUM_THREADS = '4'
$MajorRun = 'helixset_major15_seed42_rtx5070_7505336_dualpep_rdrop_b16_a4_20261026_101951_resume_20261026_131448'
$MajorDir = Join-Path 'runs' $MajorRun

foreach ($View in @('primary', 'peptide')) {
    $Root = if ($View -eq 'primary') { 'features/esm2_650m' } else { 'features/esm2_peptide_150m' }
    if (Test-Path -LiteralPath "$Root/test/metadata.json") { continue }
    & $mfpPython -u -m mfp_nova.cli features --config configs/helixset.yaml `
      --track major15 --encoder-view $View --splits test --device cuda:0 --batch-size 8
    if ($LASTEXITCODE -ne 0) { throw "Test特征提取失败：$View" }
}

& $mfpPython -u -m mfp_nova.cli evaluate `
  --checkpoint "$MajorDir/best.pt" --data-root data/therapeutic_2025 `
  --feature-root features/esm2_650m --peptide-feature-root features/esm2_peptide_150m `
  --split test --device cuda:0 --batch-size 16 `
  --output "$MajorDir/test.json" --predictions "$MajorDir/test_predictions.npz"
if ($LASTEXITCODE -ne 0) { throw 'major15 Test失败。' }
(Get-Content -LiteralPath "$MajorDir/test.json" -Raw | ConvertFrom-Json).primary_metrics
```

fine39查询数增加，先测完整R-Drop显存；该诊断只读Train/Val，不读取Test。中断的特征提取需按日志使用`--resume-from`续提，不删除缓存。随后独立从头训练fine39并评价其最佳传统候选：

```powershell
$CheckDir = "runs/gpucheck_helixset_fine39_rdrop_b16_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
& $mfpPython -u -m mfp_nova.cli gpu-check --models helixset --tasks fine39 `
  --feature-root features/esm2_650m --peptide-feature-root features/esm2_peptide_150m `
  --device cuda:0 --batch-sizes 16 --precisions fp32 --samples 128 --steps 16 --output $CheckDir
if ($LASTEXITCODE -ne 0) { throw 'fine39显存检查失败。' }

$Revision = (git rev-parse --short HEAD).Trim()
$FineRun = "helixset_fine39_seed42_rtx5070_${Revision}_dualpep_rdrop_b16_a4_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
$FineDir = Join-Path 'runs' $FineRun
$FineLog = Join-Path 'runs' "${FineRun}.log"
Write-Host "fine39实验：$FineRun"
& $mfpPython -u -m mfp_nova.cli train --model helixset --config configs/helixset.yaml `
  --task fine39 --seed 42 --device cuda:0 --precision fp32 `
  --batch-size 16 --accumulation-steps 4 --num-workers 0 `
  --ema-decay 0.999 --ema-start-epoch 6 --metric-suite traditional5 `
  --threshold-strategy global --epochs 100 --name $FineRun `
  2>&1 | Tee-Object -FilePath $FineLog
if ($LASTEXITCODE -ne 0) { throw 'fine39训练异常退出，暂不执行Test。' }

& $mfpPython -u -m mfp_nova.cli evaluate `
  --checkpoint "$FineDir/best.pt" --data-root data/therapeutic_2025 `
  --feature-root features/esm2_650m --peptide-feature-root features/esm2_peptide_150m `
  --split test --device cuda:0 --batch-size 16 `
  --output "$FineDir/test.json" --predictions "$FineDir/test_predictions.npz"
if ($LASTEXITCODE -ne 0) { throw 'fine39 Test失败。' }
(Get-Content -LiteralPath "$FineDir/test.json" -Raw | ConvertFrom-Json).primary_metrics
```

同一窗口打包两任务的小型结果及major15恢复前记录，不含checkpoint或特征。重新打开窗口后须先恢复`$MajorRun`、`$FineRun`和Python路径。

```powershell
$PreviousMajorRun = 'helixset_major15_seed42_rtx5070_7505336_dualpep_rdrop_b16_a4_20261026_101951'
$ZipPath = Join-Path ([Environment]::GetFolderPath('Desktop')) `
    "helixset_major15_fine39_report_$(Get-Date -Format 'yyyyMMdd_HHmmss').zip"
@"
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import sys
archive, major, fine, previous = sys.argv[1:]
for name in (major, fine):
    if not (Path('runs') / name / 'test.json').is_file():
        raise SystemExit(f'Test结果尚未生成：{name}')
with ZipFile(archive, 'x', ZIP_DEFLATED) as out:
    for name in (major, fine, previous):
        directory = Path('runs') / name
        if not directory.is_dir():
            continue
        for path in sorted(directory.iterdir()):
            if path.is_file() and path.suffix in ('.json', '.jsonl', '.npz', '.log'):
                out.write(path, f'{name}/{path.name}')
        log = Path('runs') / f'{name}.log'
        if log.is_file():
            out.write(log, f'{name}/console.log')
print(Path(archive).resolve())
"@ | & $mfpPython - $ZipPath $MajorRun $FineRun $PreviousMajorRun
```

### V100汇报备用命令：fine39训练与Test

主汇报建议沿用RTX5070；若在V100训练fine39，明确记录环境，不同时训练两份后按Test择优。先准备两套Train/Val/Test缓存：若V100联网仍失败，完整传入RTX生成的`features/esm2_650m/test`及`features/esm2_peptide_150m/test`。下面不下载权重，采用相同16×4协议，CPU线程和GPU可调整，后台运行支持断开SSH。

```bash
cd ~/wuzhencheng/projects/MFP-Nova
conda activate MFP-Nova
mfpPython="$(command -v python)"
export PYTHONPATH="$PWD/src"
gpu=0
threads=16
export CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS="$threads" MKL_NUM_THREADS="$threads"
revision="$(git rev-parse --short HEAD)"
fine_run="helixset_fine39_seed42_v100_${revision}_dualpep_rdrop_b16_a4_$(date +%Y%m%d_%H%M%S)"
mkdir -p runs
nohup bash -c '
set -e
mfpPython="$1"
fine_run="$2"
"$mfpPython" -u -m mfp_nova.cli gpu-check --models helixset --tasks fine39 \
  --feature-root features/esm2_650m --peptide-feature-root features/esm2_peptide_150m \
  --device cuda:0 --batch-sizes 16 --precisions fp32 --samples 128 --steps 16 \
  --output "runs/${fine_run}_gpucheck"
"$mfpPython" -u -m mfp_nova.cli train --model helixset --config configs/helixset.yaml \
  --task fine39 --seed 42 --device cuda:0 --precision fp32 \
  --batch-size 16 --accumulation-steps 4 --num-workers 4 \
  --ema-decay 0.999 --ema-start-epoch 6 --metric-suite traditional5 \
  --threshold-strategy global --epochs 100 --name "$fine_run"
"$mfpPython" -u -m mfp_nova.cli evaluate \
  --checkpoint "runs/$fine_run/best.pt" --data-root data/therapeutic_2025 \
  --feature-root features/esm2_650m --peptide-feature-root features/esm2_peptide_150m \
  --split test --device cuda:0 --batch-size 16 \
  --output "runs/$fine_run/test.json" --predictions "runs/$fine_run/test_predictions.npz"
' -- "$mfpPython" "$fine_run" > "runs/${fine_run}.log" 2>&1 < /dev/null &
echo "PID=$! Run=$fine_run"
tail -n 40 -f "runs/${fine_run}.log"
```

在V100评价同一个RTX major15候选，须先传入该运行目录的`best.pt`（结果ZIP没有权重）及两套Test缓存。若要一并打包Val与环境，请同时传入该运行的小型结果；不能用另一个V100训练的checkpoint冒充同权重评价。

```bash
major_run=helixset_major15_seed42_rtx5070_7505336_dualpep_rdrop_b16_a4_20261026_101951_resume_20261026_131448
"$mfpPython" -u -m mfp_nova.cli evaluate \
  --checkpoint "runs/$major_run/best.pt" --data-root data/therapeutic_2025 \
  --feature-root features/esm2_650m --peptide-feature-root features/esm2_peptide_150m \
  --split test --device cuda:0 --batch-size 16 \
  --output "runs/$major_run/test.json" --predictions "runs/$major_run/test_predictions.npz"
```

V100 fine39完成后的打包命令，重新登录需先恢复`fine_run`为启动时打印的名称：

```bash
python - "$fine_run" <<'PY'
from pathlib import Path
from datetime import datetime
from zipfile import ZipFile, ZIP_DEFLATED
import sys
run = Path('runs') / sys.argv[1]
if not (run / 'test.json').is_file():
    raise SystemExit('fine39 Test尚未完成。')
archive = run.parent / f'{run.name}_train_test_{datetime.now():%Y%m%d_%H%M%S}.zip'
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
