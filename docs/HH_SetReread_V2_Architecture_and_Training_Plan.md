# HH-SetReread V2.0 架构与训练计划

V2.0 是前期 V1.8 设计的正式名称。模型从肽输入直接输出 Fine39 的 39 个标签分数。原 TheraSet 和 HH V1.7 用于对照。主任务沿用原始同源簇划分和长度上限 80，Train 39,455 条，Validation 8,529 条，Test 8,180 条。ACC 指样本平均 Jaccard，长期目标为 70%。

## 1. 设计依据

- V1.7 的验证 ACC 为 61.224%，相对 TheraSet 61.046% 的增量区间跨零。最佳模型执行 137 次单删、92 次单换和 0 次单加，丢失 41 个样本的第二正确功能轴。
- 在历史消融的同一批 Fine39 Validation 上，最强 TheraSet、no-PLM 和 no-pair 的等权分数平均在固定 0.50 下达到 62.223%。这支持在表征训练中学习互补证据，保留组合结果作为开发对照。
- 原 Fine39 Train 中，抗菌家族 14,796 条有记录，其中 3,111 条只记录家族泛标签，8,823 条同时有两个具体标签。抗癌家族 4,285 条中有 3,578 条只记录泛标签。家族监督和同家族亚类监督需要不同的信号。
- 与 Major15 对齐的 39,455 条 Train 中，Major15 额外给出 71 个阳性家族位。它们可以监督家族级识别。

## 2. 前向架构

~~~mermaid
flowchart LR
    X["AA34、立体构型、位置化学、单路肽 ESM 缓存"] --> H["共享局部证据图"]
    H --> Z["全肽摘要"]
    H --> G["15 个家族的局部读出"]
    H --> C["具体标签的局部读出"]
    G --> C
    G --> M["记录粒度"]
    Z --> M
    Z --> Y["39 个直接标签分数"]
    G --> Y
    C --> Y
    M --> Y
~~~

### 2.1 化学与序列编码

采用 `MotifCNN` 的 AA34 表示、两级 3/5/7/9 卷积和残基 mask，每种卷积宽度 64 通道。复用代码并随机初始化本模型的可训练参数。保留多尺度残基图，拼接后投影到 96 维。加上距两端的位置特征和一份冻结肽 ESM150M 缓存的投影，形成 \(H\in\mathbb R^{B\times L\times96}\)。全肽摘要 \(z\) 是残基图的掩码均值和最大值拼接。

D 构型与可靠对齐的残基修饰进入对应残基。首轮将明确的 N 端乙酰化与 C 端酰胺化定位到各自端点，在 CNN 前注入。可用信息包括标准 HELM 端基和明确的同义文字记录。HELM 标有常见端基、独立字段另记为其他内容时，该端局部信号置空，其他文字记录仍进入全局化学表示。单独存在的 HELM 端基可以使用，汇总表中“缺少对应文字字段”的标志不能当作明确矛盾。其余文字修饰和连接保留全局化学输入。首轮不从未对齐的连接文本推断共价边，也不需要结构缓存。

`--ablate peptide_plm` 关闭 PLM 投影，便于隔离该表示的收益。该消融复用相同数据加载口径和 150M 缓存。

### 2.2 家族与具体标签证据

每个家族用一个查询对 \(H\) 做掩码注意力，得到 \(h_g\)。采用 Fine39 到 15 家族的原始映射，家族包含定义用于监督。多功能评价中的泛抗微生物抑制规则仍由项目指标实现。

对每个具体标签 \(c\)，以所属家族的证据 \(h_{g(c)}\) 调整查询，再读取同一残基图得到 \(v_c\)。第一层家族读出学习共享证据。第二层细标签读出学习区分同一家族亚类的证据。完整模式的分数为

\[
\ell_c=\ell_c^{\mathrm{direct}}(z)+
w_c^\top f([z,h_{g(c)},v_c,v_c-h_{g(c)},d_{g(c)}]).
\]

\(d_g\) 是记录粒度分数，描述家族已记录时是否存在具体亚类记录。对同时含泛标签和具体标签的家族，向 Fine39 读出提供该特征。泛标签使用家族向量，即 \(v_c=h_{g(c)}\)。不同细标签分别使用 sigmoid，允许同时阳性。直接分数和局部分数共享同一个编码器，均在原 Train 上学习。各辅助分数作为证据输入，最终输出使用直接 39 类 logits。

## 3. 训练监督

\[
L=L_{\mathrm{Fine39}}+0.30L_{\mathrm{family}}+0.10L_{\mathrm{detail}}.
\]

- **Fine39** 沿用项目的弱标注 ASL。已记录正例作为阳性，未记录位基准权重 0.30，同家族已记录泛标签时对未记录具体标签降权到 0.075。
- **家族** 采用 Fine39 阳性投影与合法对齐的原 Major15 Train 阳性的并集。每位缺失记录仍使用弱权重。Major15 辅助覆盖遵循原同源簇划分。
- **记录粒度** 只在 Fine39 中有该家族记录且标签字典同时包含泛标签与具体标签时启用 BCE。仅泛标签为 0，有任一具体标签为 1。单凭 Major15 提供的家族阳性不赋予细化状态。所有目标描述数据库记录。

首轮不依赖 TheraSet 权重、折外分数、近邻索引或 MIC 补充包。确证的同测定对象样本可用于后续的证据对比实验。此次主实现对应现有可用监督。

## 4. 首轮对照与评价

| 溯因对照 | CLI | Fine39 输出 | 辅助监督 |
|---|---|---|---|
| 直接模型 | `--hh-v20-mode direct` | 全肽摘要直接输出 | 无 |
| 家族辅助 | `--hh-v20-mode auxiliary` | 同一个直接头 | 家族、记录粒度 |
| 完整模型 | `--hh-v20-mode full` | 直接分数加家族条件化的局部读出 | 家族、记录粒度 |

三组均使用原 Fine39 数据、同一编码设置、相同初始 seed 和训练协议。主表固定 0.50 阈值，以 Validation Jaccard 选模。除 ACC、Macro AP、Micro F1 和独立多功能检测外，使用逐样本预测对照 TheraSet 的原高置信错类、第二和第三个真实标签在前五的覆盖、原完全匹配样本是否受损。项目现有的 `scripts/compare_validation_predictions.py` 支持按行对齐并报告这类诊断。按同一口径复跑必要 seed 后做独立 Test。

每个模式保存 `metadata.json`、`history.jsonl`、`results.json`、`best.pt`、`last.pt` 和 `best_validation.npz`。记录粒度和家族损失可在训练历史中核对。

## 5. 首轮训练命令

两个平台都使用 seed 157、FP32、batch 64、最多 60 epoch、patience 10。运行前同步含 V2.0 的源码。仅用原数据和 `features/esm2_peptide_150m/{train,val}`，已存在的长度 80 缓存可复用。按实际空闲物理卡号改 `GPU` 或 `CUDA_VISIBLE_DEVICES`。

### V100 · Ubuntu Bash

下面在一张物理 GPU 后台顺序运行三组。退出 SSH 后队列继续。

~~~bash
cd ~/wuzhencheng/projects/MFP-Nova
conda activate MFP-Nova
GPU=0
CPU_THREADS=6
RUN="hh_v20_fine39_s157_$(date +%Y%m%d_%H%M%S)"
mkdir -p "runs/$RUN"

nohup env CUDA_VISIBLE_DEVICES="$GPU" OMP_NUM_THREADS="$CPU_THREADS" MKL_NUM_THREADS="$CPU_THREADS" \
  PYTHONPATH="$PWD/src" bash -c '
  set -e
  run="$1"; shift
  for mode in "$@"; do
    python -u -m mfp_nova.cli train --model hh_setreread_v20 --task fine39 \
      --hh-v20-mode "$mode" --seed 157 --device cuda:0 --precision fp32 --batch-size 64 \
      --epochs 60 --metric-suite jaccard --name "$run/$mode" > "runs/$run/$mode.log" 2>&1
  done
' _ "$RUN" direct auxiliary full > "runs/$RUN/queue.log" 2>&1 < /dev/null &
printf 'PID=%s RUN=runs/%s\n' "$!" "$RUN"
~~~

只跑完整模型时，把队列末尾的 `direct auxiliary full` 换成 `full`。有三张空闲卡时，可为三个模式各启动一个进程，使用不同物理卡和同一批次的三个子目录。

### RTX 5070 · Windows PowerShell

~~~powershell
Set-Location 'D:\Data\Code\MFP-Nova'
conda activate MFP-Nova
$env:CUDA_VISIBLE_DEVICES = '0'
$env:OMP_NUM_THREADS = '6'
$env:MKL_NUM_THREADS = '6'
$env:PYTHONPATH = "$PWD/src"
$run = "hh_v20_fine39_s157_$(Get-Date -Format yyyyMMdd_HHmmss)"
New-Item -ItemType Directory -Path "runs/$run" | Out-Null
foreach ($mode in 'direct', 'auxiliary', 'full') {
    python -u -m mfp_nova.cli train --model hh_setreread_v20 --task fine39 `
      --hh-v20-mode $mode --seed 157 --device cuda:0 --precision fp32 --batch-size 64 `
      --epochs 60 --metric-suite jaccard --name "$run/$mode" 2>&1 |
      Tee-Object -FilePath "runs/$run/$mode.log"
    if ($LASTEXITCODE -ne 0) { break }
}
~~~

如要先测显存，可用 `gpu-check --models hh_setreread_v20 --tasks fine39 --profile-only --device cuda:0 --precisions fp32 --batch-sizes 32 64 96 --steps 8 --output <新实验子目录>`。测试会使用真实随机批次和最长肽批次执行反向。

## 6. 主干与直接头对照

首轮 `direct` 的 Fine39 验证 ACC 为 47.96%，已有 TheraSet 在同一验证集上达到 61.05%。以下对照沿用首轮 Fine39 原始划分、ESM150M、最大长度 80、化学输入和训练配置，只使用 Fine39 损失并从头训练。

| 实验 | 参数 | 比较用途 |
|---|---|---|
| 首轮 `direct` | `--hh-v20-mode direct --hh-v20-cnn-channels 64` | 复用首轮结果 |
| `wide_direct` | `--hh-v20-mode direct --hh-v20-cnn-channels 128` | 单独测量增加每尺度通道数的影响 |
| `theraset_direct` | `--hh-v20-mode theraset_direct --hh-v20-cnn-channels 128` | 恢复每尺度的 masked max pooling 和 TheraSet 的 `512 → 1024 → 512 → 39` 直接头 |

`theraset_direct` 保留 V2.0 的末端化学、位置和 ESM150M 输入。CNN 的四组最大池化向量按通道拼接。位置和 ESM 的残基向量单独做均值与最大池化，投影到 512 维后与 CNN 向量相加。因此它与 `wide_direct` 的差值衡量池化、融合和直接头的组合效果。

### V100 · 两张卡并行

~~~bash
cd ~/wuzhencheng/projects/MFP-Nova
conda activate MFP-Nova
git pull --ff-only origin main
GPUS=(0 1)  # 改为实际空闲的物理卡号
CPU_THREADS=6
RUN="hh_v20_direct_fine39_s157_$(date +%Y%m%d_%H%M%S)"
mkdir -p "runs/$RUN"

launch() {
  local gpu="$1" name="$2" mode="$3"
  nohup env CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS="$CPU_THREADS" \
    MKL_NUM_THREADS="$CPU_THREADS" PYTHONPATH="$PWD/src" \
    python -u -m mfp_nova.cli train --model hh_setreread_v20 --task fine39 \
    --hh-v20-mode "$mode" --hh-v20-cnn-channels 128 \
    --seed 157 --device cuda:0 --precision fp32 --batch-size 64 \
    --epochs 60 --metric-suite jaccard --name "$RUN/$name" \
    > "runs/$RUN/$name.log" 2>&1 < /dev/null &
  printf 'GPU=%s PID=%s LOG=runs/%s/%s.log\n' "$gpu" "$!" "$RUN" "$name"
}
launch "${GPUS[0]}" wide_direct direct
launch "${GPUS[1]}" theraset_direct theraset_direct
~~~

### RTX 5070 · 单卡顺序

~~~powershell
Set-Location 'D:\Data\Code\MFP-Nova'
conda activate MFP-Nova
git pull --ff-only origin main
$env:CUDA_VISIBLE_DEVICES = '0'
$env:OMP_NUM_THREADS = '6'
$env:MKL_NUM_THREADS = '6'
$env:PYTHONPATH = "$PWD/src"
$run = "hh_v20_direct_fine39_s157_$(Get-Date -Format yyyyMMdd_HHmmss)"
New-Item -ItemType Directory -Path "runs/$run" | Out-Null
foreach ($name in 'wide_direct', 'theraset_direct') {
    $mode = if ($name -eq 'wide_direct') { 'direct' } else { 'theraset_direct' }
    python -u -m mfp_nova.cli train --model hh_setreread_v20 --task fine39 `
      --hh-v20-mode $mode --hh-v20-cnn-channels 128 `
      --seed 157 --device cuda:0 --precision fp32 --batch-size 64 `
      --epochs 60 --metric-suite jaccard --name "$run/$name" 2>&1 |
      Tee-Object -FilePath "runs/$run/$name.log"
    if ($LASTEXITCODE -ne 0) { break }
}
~~~

### 训练结束后导出

在启动训练的 V100 终端执行。归档包含整批配置、日志、指标及两组逐样本验证预测。

~~~bash
tar -czf "runs/$RUN/validation.tar.gz" -C "runs/$RUN" \
  wide_direct.log theraset_direct.log \
  wide_direct/{metadata.json,history.jsonl,results.json,best_validation.npz} \
  theraset_direct/{metadata.json,history.jsonl,results.json,best_validation.npz}
~~~

## 7. 化学 CNN 与 ESM/位置融合对照

`wide_direct` 和 `theraset_direct` 在首轮主干对照中分别达到 52.24% 和 51.05%。历史 TheraSet `cnn_chemical` 在同一验证集上为 57.78%，训练协议不同。本轮两组复用原 TheraSet 的全局修饰编码、128 通道卷积、逐尺度最大池化和直接头，共享 seed 157 下的 CNN 与分类头初始权重，按 V2.0 协议独立训练。`chemical_cnn` 仅从化学 CNN 读出。`chemical_cnn_fusion` 额外将 ESM150M 与由两端距离及其衰减得到的位置向量相加，按残基做均值与最大池化，投影后加到 CNN 池化向量。以两组在相同验证集的差值衡量融合贡献，同时查看首位、前五多标签覆盖和原正确集合保留数。

### V100 · 两张卡并行

复用上一节 V100 的启动命令，保留 `launch()` 函数和前面的设置，用以下命令替换原来的两行 `launch` 调用。

~~~bash
RUN="hh_v20_chemical_fine39_s157_$(date +%Y%m%d_%H%M%S)"
mkdir -p "runs/$RUN"
launch "${GPUS[0]}" chemical_cnn chemical_cnn
launch "${GPUS[1]}" chemical_cnn_fusion chemical_cnn_fusion
~~~

### RTX 5070 · 单卡顺序

复用上一节 Windows 的目录、环境和线程设置，再执行下面的完整训练循环。

~~~powershell
$run = "hh_v20_chemical_fine39_s157_$(Get-Date -Format yyyyMMdd_HHmmss)"
New-Item -ItemType Directory -Path "runs/$run" | Out-Null
foreach ($mode in 'chemical_cnn', 'chemical_cnn_fusion') {
    python -u -m mfp_nova.cli train --model hh_setreread_v20 --task fine39 `
      --hh-v20-mode $mode --hh-v20-cnn-channels 128 `
      --seed 157 --device cuda:0 --precision fp32 --batch-size 64 `
      --epochs 60 --metric-suite jaccard --name "$run/$mode" 2>&1 |
      Tee-Object -FilePath "runs/$run/$mode.log"
    if ($LASTEXITCODE -ne 0) { break }
}
~~~

### V100 · 验证对比与打包

~~~bash
PYTHONPATH="$PWD/src" python scripts/compare_validation_predictions.py \
    "runs/$RUN/chemical_cnn" "runs/$RUN/chemical_cnn_fusion" | tee "runs/$RUN/paired.json"
tar -czf "runs/$RUN/${RUN}_validation.tar.gz" -C "runs/$RUN" \
    paired.json chemical_cnn.log chemical_cnn_fusion.log \
    chemical_cnn/{metadata.json,history.jsonl,results.json,best_validation.npz} \
    chemical_cnn_fusion/{metadata.json,history.jsonl,results.json,best_validation.npz}
~~~
