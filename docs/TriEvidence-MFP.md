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
