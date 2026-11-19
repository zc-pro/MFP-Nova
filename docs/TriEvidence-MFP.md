# TriEvidence-MFP

TriEvidence-MFP 在同一份 Fine39 batch 上并行读取三路标签证据，然后融合为一个预测。输入共用原始残基、掩码和双 ESM 特征缓存。三路各自保留独立参数。

| 分支 | 输入与读出 | 首轮训练 |
|---|---|---|
| TheraSet | 原有多模态取证和 Fine39 输出 | 载入原权重，固定在推理模式 |
| AA34 CNN | 只读标准残基与 mask，两级多尺度卷积、掩码最大池化和 MLP | 原 Fine39 Train 独立学习，沿用弱权重未标注项损失 |
| HelixSet latent-direct | 保留残基对与潜变量层，直接读取已监督的 `latent_classifier` | 载入原权重并移除标签解码器，固定在推理模式 |

对三路 logits 分别减去各自 cutoff 对应的 logit，再等权平均。融合分数大于等于零的标签纳入预测。如果融合结果为空，在 TheraSet 排名 1–10 的标签中补入融合分最高的一个。训练只预计算一次原 Validation 的两路固定分数。按兜底前的 Validation 样本 Jaccard 选择 CNN epoch 和全局阈值。评估同时报告兜底前后与两路对照，以及 TheraSet Top-5 漏检救回数和第 6–10 名正确标签找回数。

首轮目标是测量单个 CNN 能否保留原五成员 CNN 的互补性。用同一批 8,529 条 Validation 与固定融合历史对照比较。其中完整三模型融合的 ACC 为 62.41%，HelixSet 更换为 latent-direct 后为 62.31%，两者均保留五成员 CNN。完整三模型加空预测兜底的独立 Test ACC 为 65.04%。

## 运行

训练机需同时有下面两份原始 `best.pt`，以及训练选模用的双冻结 ESM Validation 缓存。TheraSet 对应 V5 的 `baseline/best.pt`，最佳 53 轮。HelixSet 对应 RTX 5070 完整模型，最佳 24 轮。若两份权重分居两台机器，先通过已有传输方式将缺少的一份放到训练机，随后更新以下对应路径。

### RTX 5070 PowerShell

```powershell
conda activate MFP-Nova
$py = Join-Path $env:CONDA_PREFIX 'python.exe'
$env:PYTHONPATH = "$PWD/src"
$env:OMP_NUM_THREADS = '4'
$env:MKL_NUM_THREADS = '4'
$thera = 'runs/v5_fine39_s42_20261116_235947/baseline/best.pt'
$helix = 'runs/helixset_fine39_seed42_rtx5070_eb5ab65_dualpep_rdrop_b16_a4_20261027_002630/best.pt'
$run = "runs/trievidence_fine39_s42_$(Get-Date -Format yyyyMMdd_HHmmss)"
New-Item -ItemType Directory -Path $run -ErrorAction Stop | Out-Null
& $py -u scripts/run_trievidence.py train --theraset $thera --helixset $helix `
  --seed 42 --epochs 25 --device cuda:0 --batch-size 16 --cnn-batch-size 128 `
  --output "$run/model" 2>&1 | Tee-Object -FilePath "$run/train.log"
```

### V100 Bash

```bash
conda activate MFP-Nova
py="$CONDA_PREFIX/bin/python"
gpu=4
cpu_threads=4
thera="runs/v5_fine39_s42_20261116_235947/baseline/best.pt"
helix="runs/helixset_fine39_seed42_rtx5070_eb5ab65_dualpep_rdrop_b16_a4_20261027_002630/best.pt"
run="runs/trievidence_fine39_s42_$(date +%Y%m%d_%H%M%S)"
mkdir -p runs
mkdir "$run" && {
  nohup env CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS="$cpu_threads" MKL_NUM_THREADS="$cpu_threads" \
    PYTHONPATH="$PWD/src" "$py" -u scripts/run_trievidence.py train \
      --theraset "$thera" --helixset "$helix" --seed 42 --epochs 25 \
      --device cuda:0 --batch-size 16 --cnn-batch-size 128 --output "$run/model" \
      >"$run/train.log" 2>&1 </dev/null &
  printf 'PID=%s  日志=%s\n' "$!" "$run/train.log"
}
```

输出为 `model/best.pt`、`model/results.json` 和 `model/val_predictions.npz`。逐样本文件带原始 row ID、簇 ID、三路 logits、融合分和兜底前后预测。

原权重还在不同机器时，可以用完整原始 Validation 的历史分数先训练 CNN。在任一项目环境运行 `python -u scripts/run_trievidence.py train-cached --theraset-val THERASET_EXPORT --helix-exits HELIXSET_EXITS_NPZ --device cuda:0`。TheraSet 文件可用原 `best_validation.npz` 或包含它的 V5 归档。HelixSet 文件使用完整潜变量深度的 `exits.npz`。这种方式只读取原始残基和历史 Validation 输出。训练产物 `reader.pt` 是 CNN 权重及已选阈值。两份原始检查点齐备后，在上述对应平台的 `train` 命令加上 `--reader 路径/reader.pt`，即可跳过 CNN 重训、实跑 Validation 并打包完整 `best.pt`。如果训练时改过通道数，打包时使用同一 `--channels`。

选定配置后可在带对应 Test 缓存的环境运行固定权重和规则的独立评估。两平台激活项目环境后，使用 `python -u scripts/run_trievidence.py evaluate 路径/model/best.pt --split test --device cuda:0`。明确指定 `--split val` 可复核训练时的 Validation 结果。评估只加载打包后的权重文件与目标划分的输入缓存。

## 三卡 V100 复验

将提供的 `trievidence_val_inputs.tar.gz` 放到项目的 `runs/incoming/`。归档包含同一批原始 Validation 的 TheraSet 输出和完整深度的 HelixSet latent-direct 输出。每张卡使用一个 seed，在完整原始 Train 上跑 25 轮。同一批运行目录中记录 CNN、固定等权融合和候选补标结果。改 `gpus` 为当前可用的三张卡。Git worktree 使用独立源码目录，三个进程共用主项目的数据。

```bash
(
  set -eo pipefail
  project="$HOME/wuzhencheng/projects/MFP-Nova"
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate MFP-Nova
  set -u
  py="$CONDA_PREFIX/bin/python"
  gpus=(2 3 4)
  seeds=(42 101 202)
  stamp=$(date +%Y%m%d_%H%M%S)
  run="$project/runs/trievidence_v100_seeds_$stamp"
  code="$(dirname "$project")/MFP-Nova-trievidence-$stamp"

  test -r "$project/runs/incoming/trievidence_val_inputs.tar.gz"
  git -C "$project" fetch origin refs/heads/codex/trievidence-mfp
  git -C "$project" worktree add --detach "$code" FETCH_HEAD
  mkdir "$run"
  mkdir "$run/inputs"
  tar -xzf "$project/runs/incoming/trievidence_val_inputs.tar.gz" -C "$run/inputs"
  cd "$code"
  for i in "${!seeds[@]}"; do
    seed="${seeds[$i]}"
    gpu="${gpus[$i]}"
    arm="$run/seed_$seed"
    mkdir "$arm"
    nohup env CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH="$code/src" \
      bash -c '
        set -euo pipefail
        py=$1; project=$2; run=$3; arm=$4; seed=$5
        "$py" -u scripts/run_trievidence.py train-cached \
          --theraset-val "$run/inputs/theraset_val.npz" --helix-exits "$run/inputs/helix_latent_val.npz" \
          --data-root "$project/data/therapeutic_2025" \
          --seed "$seed" --epochs 25 --device cuda:0 --cnn-batch-size 128 --output "$arm/reader"
        "$py" -u scripts/screen_trievidence_fusion.py "$arm/reader/val_predictions.npz" --output "$arm/fusion"
      ' bash "$py" "$project" "$run" "$arm" "$seed" >"$arm/train.log" 2>&1 </dev/null &
    printf 'seed=%s GPU=%s PID=%s 日志=%s\n' "$seed" "$gpu" "$!" "$arm/train.log"
  done
  printf '实验目录=%s\n' "$run"
)
```

查看日志可在 V100 上运行下列命令。

```bash
project="$HOME/wuzhencheng/projects/MFP-Nova"
run=$(ls -td "$project"/runs/trievidence_v100_seeds_*/ | head -n 1)
tail -n 8 -F "${run%/}"/seed_{42,101,202}/train.log
```

优先比较各 seed 的 `reader/results.json` 中两路与三路的 ACC、零命中和第 6–10 名新命中数。用 `fusion/results.json` 的 `nonempty_add_fixed` 检验固定规则的跨 seed 表现。规则只为非空预测最多补一个标签，要求它位于 TheraSet 第 6–10 名和 AA34 前 5 名，已过 AA34 自身 cutoff，且等权融合分离最终 cutoff 的差距不超过 1 logit。逐簇重选规则的对照同时保留。

## 局部与背景对照读出

AA34 的 `--reader-head contrast` 用现有 CNN 残基特征预测每个标签的局部分数。分别算出高响应位置的平均分与整条肽的平均分，取两者之差作为全局 logits 的残差。每条肽选取的高响应位置数为向上取整的一半长度，最多三个。弱权重未标注项损失、数据划分和等权融合沿用原设置。

首轮对照用同一个 seed、相同的 25 轮和同一套 Validation 缓存。三臂分别为 `global` 原全局池化、`local` 加标签级局部读出、`contrast` 加标签级局部读出并减肽内背景。后两臂参数相同。对照聚焦兜底前 ACC、独立多功能样本 Jaccard、第 6–10 位新增正确标签和原命中丢失数。现有 RTX 5070 与 V100 命令均支持 `--reader-head`，打包时沿用训练所用的选项。此消融在 Validation 上进行。

## 架构精简计划

精简以已验证的 seed 42 TriEvidence-MFP 为性能基线。原始 Validation 的兜底前和兜底后 ACC 分别为 62.3032% 和 63.0487%，原始 Test 为 63.2046% 和 64.0934%。结构改变的候选在同一批 Validation 上同时比较兜底前和兜底后 ACC，要求两项均保持现有水平，并查看配对的按簇置信区间、零命中和独立多功能样本 Jaccard。置信区间跨零时按证据不足解释。候选和阈值在 Validation 上确定，再固定设置做一次 Test 复验。实际可部署的模型还要在同设备上体现参数、延迟或峰值显存收益。

现版从同一批张量读两套 ESM 缓存。TheraSet 和 HelixSet 随后独立执行双 ESM 归一化与投影、残基化学编码、理化属性编码以及几何和残基对处理。HelixSet 的 latent-direct 省去标签解码器，仍使用序列块、结构块、残基对块和潜变量层。用原配置和当前 Fine39 词表构造模型，下游参数为 TheraSet 10,404,206、HelixSet latent-direct 56,283,879、AA34 CNN 2,844,903。两套冻结的 ESM 权重未计入。

固定同一个本机 AA34 检查点的阈值，在 8,529 条原始 Validation 上移除一条分支。三路的样本 Jaccard 为 62.2807%。去掉 AA34 后为 61.7520%，去掉 HelixSet 后为 61.4268%。结果表明 HelixSet 提供的补充证据值得保留。AA34 是成本较低、输入约束不同的一条路，适合作为第一阶段的固定对照。

### 1. 等价复用与成本实测

在原 seed 42 检查点上先测基线推理。使用同一设备、相同精度与 batch，记录预热后的中位延迟、P95 延迟和峰值显存。两条分支可以共用等价的双 ESM 无参数归一化和拼接，各自的学习型投影与读出保持原权重。用同一批输入比较逐标签 logits 及预测集合，确认浮点差异没有改变决策。同样记录只计下游和包含数据读入的耗时，区分模型计算与缓存 I/O 的收益。

依据已有 logits 的裁剪上下界，在本机参考 Validation 上，8,529 条里只有 3 条可在完全不计算 HelixSet 时保证全部标签阈值判断相同且输出非空。两路加入 HelixSet 后实际改变了 1,633 条样本的至少一个标签决定，因此全分支的置信度跳过不作为首轮方案。

### 2. 先裁剪 HelixSet latent-direct 的现有潜变量层

在同一原始检查点上，使用 `scripts/screen_helixset_depth.py` 导出潜变量深度 6、5、4 的 `direct` 分数。对齐 V100 整模型的原始 Validation 逐样本输出，保持 TheraSet、AA34、三路阈值及融合和兜底规则一致，逐路替换 HelixSet 分数。六层用于验证导出能重现基线。五层可减少 399.7 万下游参数，整模型约 6,554 万。四层可减少 799.3 万，整模型约 6,154 万。每次只改潜变量层数，保留序列、几何和残基对处理。

如果较浅候选仍有互补性但出现小幅下降，优先恢复表现更好的那一组。先仅在原 Train 用实际标签监督 latent 分类头和最后一层保留的潜变量块，做少量轮次短训，再用相同协议复核。若恢复后仍低于基线，继续采用更深版本。等价预处理与裁剪分开计时，实测用 latent-direct 前向。

### 3. 共享骨干作为更大幅度精简的候选

五层和四层恢复均未达到两项 ACC 基线。当前采用六层作为性能对照，进入共享骨干实验。在本机缓存 Validation 对照中，加入六层 HelixSet 后，相比 TheraSet 加 AA34 新增命中 385 个已标注标签，也丢失 250 个。224 条原零命中样本获救，137 条原有命中的样本变为零命中。新读出应逐样本比较这些有益和有害的改变，以及两项总 ACC。

目标结构是 **一个多模态骨干、两个独立多模态读出和一个轻量 AA34 分支**。沿用 TheraSet 骨干和原读出。在同一次 `encode_evidence` 调用中取得残基状态、序列残基对、可用几何残基对与质量掩码，供两个多模态读出复用。把残基状态投影到 128 维做一次残基注意力，将现有残基对投影为注意力偏置，并用质量掩码控制几何偏置。随后通过独立标签查询读出 39 维 logits。关系读出的参数预算为 100 万以内，整模型下游参数目标约 1,425 万以内。

原始 Train 只训练新读出，复用已有弱权重未标注项损失。三组对照为原两路、加共享状态读出、加带残基对偏置的共享读出。同时看相对两路找回的正确标签和原三路保住的标签数。利用这组对照判断共享骨干剩余的表示是否足以补回 HelixSet 的信息，再决定是否继续投入。

首轮继续保留 AA34 独立分支。TheraSet 的 CNN 在卷积前注入化学信息，AA34 只读标准残基。若要进一步共享 CNN，可让序列读出从化学注入前的共同卷积分出，再在 TheraSet 路径补入化学信息。这会改变现有权重的输入，需要另做重训和验证。

## 深度精简实验与验收命令

本流程从已验证的打包目录启动。该目录的 `model/` 内同时需要有 `best.pt` 和同一次运行导出的 `val_predictions.npz`。一次记录中，`epoch=0` 表示直接裁剪的权重。启用恢复训练后，保留组内兜底前后 ACC 相对基线的较小差值最大的轮次。三路阈值固定为原 checkpoint 的值。只在 Validation 选定候选，随后各跑一次基线和候选的 Test。

### V100 Ubuntu Bash

先在同一个 Bash 终端执行以下准备和三卡筛选。改 `gpus` 为三个空闲卡号。

```bash
project="$HOME/wuzhencheng/projects/MFP-Nova"
base="$project/runs/trievidence_packed_s42_20261119_155453/model"
gpus=(2 3 4)
cpu_threads=4
stamp=$(date +%Y%m%d_%H%M%S)
work="$project/runs/trievidence_depth_$stamp"
code="$(dirname "$project")/MFP-Nova-depth-$stamp"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate MFP-Nova
py="$CONDA_PREFIX/bin/python"

(
  set -e
  git -C "$project" fetch origin refs/heads/codex/trievidence-mfp
  git -C "$project" worktree add --detach "$code" FETCH_HEAD
  mkdir "$work"
)
cd "$code"
export PYTHONPATH="$code/src"
common=(--data-root "$project/data/therapeutic_2025"
        --features "$project/features/esm2_650m"
        --peptide-features "$project/features/esm2_peptide_150m"
        --device cuda:0 --batch-size 16)

depth_job() {
  local arm="$1" gpu="$2" depth="$3" epochs="$4"
  nohup env CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS="$cpu_threads" MKL_NUM_THREADS="$cpu_threads" \
    "$py" -u scripts/refine_trievidence_depth.py "$base/best.pt" "${common[@]}" \
    --depth "$depth" --epochs "$epochs" --output "$work/$arm" \
    >"$work/$arm.log" 2>&1 </dev/null &
  printf '%s GPU=%s PID=%s 日志=%s\n' "$arm" "$gpu" "$!" "$work/$arm.log"
}

depth_job depth6 "${gpus[0]}" 6 0
depth_job depth5 "${gpus[1]}" 5 0
depth_job depth4 "${gpus[2]}" 4 0
tail -n 8 -F "$work"/depth{6,5,4}.log
```

三组完成后按 `Ctrl+C` 退出日志，并运行：

```bash
for arm in depth6 depth5 depth4; do
  "$py" scripts/compare_trievidence.py "$base" "$work/$arm" | tee "$work/compare_$arm.json"
done

# 若较浅版本需要恢复，按筛选结果改 depth 为 5 或 4。全量原 Train 跑三轮。
depth=5
depth_job "recover$depth" "${gpus[0]}" "$depth" 3
tail -n 10 -F "$work/recover$depth.log"
```

六层对照应重现原 Validation。较浅版本若已达到兜底前后两项基线，可以直接选其 `best.pt`。恢复训练结束可用同一比较命令核对，例如 `"$py" scripts/compare_trievidence.py "$base" "$work/recover5"`。选好单个候选后，用**同一张卡顺序**复跑实际整模型前向并计时：

```bash
chosen="$work/depth5"   # 在 Validation 上选定，按实际结果改为 depth4 或 recover5 等
(
  set -eo pipefail
  export CUDA_VISIBLE_DEVICES="${gpus[0]}" OMP_NUM_THREADS="$cpu_threads" MKL_NUM_THREADS="$cpu_threads"
  "$py" -u scripts/run_trievidence.py evaluate "$base/best.pt" "${common[@]}" \
    --split val --output "$work/base_val" | tee "$work/base_val.log"
  "$py" -u scripts/run_trievidence.py evaluate "$chosen/best.pt" "${common[@]}" \
    --split val --output "$work/chosen_val" | tee "$work/chosen_val.log"
  "$py" scripts/compare_trievidence.py "$work/base_val" "$work/chosen_val" | tee "$work/accept_val.json"
)
```

完成 Validation 后，沿用上一步的 `chosen`，运行固定配置的 Test。使用已备好的双 ESM Test 缓存。

```bash
(
  set -eo pipefail
  export CUDA_VISIBLE_DEVICES="${gpus[0]}" OMP_NUM_THREADS="$cpu_threads" MKL_NUM_THREADS="$cpu_threads"
  "$py" -u scripts/run_trievidence.py evaluate "$base/best.pt" "${common[@]}" \
    --split test --output "$work/base_test" | tee "$work/base_test.log"
  "$py" -u scripts/run_trievidence.py evaluate "$chosen/best.pt" "${common[@]}" \
    --split test --output "$work/chosen_test" | tee "$work/chosen_test.log"
  "$py" scripts/compare_trievidence.py "$work/base_test" "$work/chosen_test" | tee "$work/accept_test.json"
)
```

### RTX 5070 Windows PowerShell

从 RTX 5070 的项目目录运行。先将原打包目录的 `best.pt` 和 `val_predictions.npz` 同放在下面的 `$base`。项目准备相同数据和特征缓存。

```powershell
$project = "$PWD"
$base = Join-Path $project 'runs/trievidence_packed_s42_20261119_155453/model'
$stamp = Get-Date -Format yyyyMMdd_HHmmss
$work = Join-Path $project "runs/trievidence_depth_$stamp"
$code = Join-Path (Split-Path $project -Parent) "MFP-Nova-depth-$stamp"
conda activate MFP-Nova
$py = Join-Path $env:CONDA_PREFIX 'python.exe'
git -C $project fetch origin refs/heads/codex/trievidence-mfp
if ($LASTEXITCODE -ne 0) { throw 'git fetch failed' }
git -C $project worktree add --detach $code FETCH_HEAD
if ($LASTEXITCODE -ne 0) { throw 'git worktree failed' }
New-Item -ItemType Directory -Path $work -ErrorAction Stop | Out-Null
Set-Location $code
$env:PYTHONPATH = "$code/src"
$env:OMP_NUM_THREADS = '4'
$env:MKL_NUM_THREADS = '4'
$common = @('--data-root', "$project/data/therapeutic_2025", '--features', "$project/features/esm2_650m",
            '--peptide-features', "$project/features/esm2_peptide_150m", '--device', 'cuda:0', '--batch-size', '16')

foreach ($depth in 6,5,4) {
    & $py -u scripts/refine_trievidence_depth.py "$base/best.pt" @common --depth $depth --epochs 0 `
        --output "$work/depth$depth" 2>&1 | Tee-Object "$work/depth$depth.log"
    if ($LASTEXITCODE -ne 0) { throw "Depth $depth failed" }
    & $py scripts/compare_trievidence.py $base "$work/depth$depth" | Tee-Object "$work/compare_depth$depth.json"
}

# 需要恢复时执行这三行，按筛选结果选 4 或 5。
$depth = 5
& $py -u scripts/refine_trievidence_depth.py "$base/best.pt" @common --depth $depth --epochs 3 `
    --output "$work/recover$depth" 2>&1 | Tee-Object "$work/recover$depth.log"
& $py scripts/compare_trievidence.py $base "$work/recover$depth"
```

用选定的同一个 `$chosen` 先复跑 Validation。核对通过后，将 `$split` 改成 `test` 再执行本段：

```powershell
$chosen = "$work/depth5"
$split = 'val'
& $py -u scripts/run_trievidence.py evaluate "$base/best.pt" @common --split $split --output "$work/base_$split"
if ($LASTEXITCODE -ne 0) { throw 'Baseline evaluation failed' }
& $py -u scripts/run_trievidence.py evaluate "$chosen/best.pt" @common --split $split --output "$work/chosen_$split"
if ($LASTEXITCODE -ne 0) { throw 'Candidate evaluation failed' }
& $py scripts/compare_trievidence.py "$work/base_$split" "$work/chosen_$split" | Tee-Object "$work/accept_$split.json"
```

比较文件记录两项 ACC 差值、按簇配对区间、变化样本数、零命中和多功能肽 Jaccard。整模型复跑还记录参数数、预热后的批次 P50/P95 和显存峰值。批次时间包含数据读取和传输。在相同硬件、batch 和负载下解读。若原六层已经是唯一达到既有性能的候选，本轮结论是继续使用原版。

## 共享读出训练与验收

从原六层打包目录读取 `best.pt` 与 `val_predictions.npz`。两组分别用 `states`（共享残基状态）和 `pairs`（再加残基对偏置），固定 seed 42、8 轮、batch 16、学习率 0.0003。只训练新读出。TheraSet 和 AA34 的权重与阈值固定，新读出的阈值从 Validation 的 0.10 到 0.90 网格选择。比较原始三路的两项 ACC，并保留两路 `theraset_aa34` 对照。Test 仅运行在 Validation 上选定的候选。

### 共享读出 V100 Bash

```bash
project="$HOME/wuzhencheng/projects/MFP-Nova"
base="$project/runs/trievidence_packed_s42_20261119_155453/model"
stamp=$(date +%Y%m%d_%H%M%S)
work="$project/runs/trievidence_shared_$stamp"
code="$(dirname "$project")/MFP-Nova-shared-$stamp"
gpus=(2 3)
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate MFP-Nova
py="$CONDA_PREFIX/bin/python"

(
  set -e
  git -C "$project" fetch origin refs/heads/codex/trievidence-mfp
  git -C "$project" worktree add --detach "$code" FETCH_HEAD
  mkdir "$work"
)
cd "$code"
export PYTHONPATH="$code/src" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
common=(--data-root "$project/data/therapeutic_2025"
        --features "$project/features/esm2_650m"
        --peptide-features "$project/features/esm2_peptide_150m"
        --device cuda:0 --batch-size 16)

modes=(states pairs)
for i in "${!modes[@]}"; do
  mode="${modes[$i]}"
  nohup env CUDA_VISIBLE_DEVICES="${gpus[$i]}" "$py" -u scripts/train_trievidence_shared.py \
    "$base/best.pt" "${common[@]}" --mode "$mode" --epochs 8 --output "$work/$mode" \
    >"$work/$mode.log" 2>&1 </dev/null &
  printf '%s GPU=%s PID=%s 日志=%s\n' "$mode" "${gpus[$i]}" "$!" "$work/$mode.log"
done
tail -n 8 -F "$work"/{states,pairs}.log
```

两组完成后，先比较缓存分数。达到 Validation 基线的候选进入整模型复跑。

```bash
for mode in states pairs; do
  "$py" scripts/compare_trievidence.py "$base" "$work/$mode" | tee "$work/compare_$mode.json"
done
```

将 `chosen` 指向入选的 `states` 或 `pairs`。先运行 Validation 串行实测，复核通过后保持 `chosen` 不变，将 `split` 改为 `test` 重跑命令。

```bash
chosen="$work/pairs"
split=val
(
  set -eo pipefail
  export CUDA_VISIBLE_DEVICES="${gpus[0]}"
  "$py" -u scripts/run_trievidence.py evaluate "$base/best.pt" "${common[@]}" \
    --split "$split" --output "$work/base_$split" | tee "$work/base_$split.log"
  "$py" -u scripts/run_trievidence.py evaluate "$chosen/best.pt" "${common[@]}" \
    --split "$split" --output "$work/chosen_$split" | tee "$work/chosen_$split.log"
  "$py" scripts/compare_trievidence.py "$work/base_$split" "$work/chosen_$split" | tee "$work/accept_$split.json"
)
```

### 共享读出 RTX 5070 PowerShell

沿用上方深度精简 PowerShell 的准备部分，建立最新工作树并得到 `$project`、`$base`、`$py`、`$work` 和 `@common`，随后执行下面两组。

```powershell
foreach ($mode in 'states','pairs') {
    & $py -u scripts/train_trievidence_shared.py "$base/best.pt" @common --mode $mode --epochs 8 `
        --output "$work/$mode" 2>&1 | Tee-Object "$work/$mode.log"
    if ($LASTEXITCODE -ne 0) { throw "Shared readout $mode failed" }
    & $py scripts/compare_trievidence.py $base "$work/$mode" | Tee-Object "$work/compare_$mode.json"
}
```

沿用上方 PowerShell 的串行整模型验收命令，将 `$chosen` 设为 `"$work/states"` 或 `"$work/pairs"`。部署对照仍为六层 Test 兜底前 63.2046%、兜底后 64.0934%。
