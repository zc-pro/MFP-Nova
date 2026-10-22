#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
python_bin="${PYTHON_BIN:-python}"
cpu_threads="${CPU_THREADS:-4}"
export OMP_NUM_THREADS="$cpu_threads" MKL_NUM_THREADS="$cpu_threads" OPENBLAS_NUM_THREADS="$cpu_threads"
tasks=("$@")
if [ "${#tasks[@]}" -eq 0 ]; then tasks=(major15 fine39); fi
tag=$(date +%Y%m%d_%H%M%S)
runs=()
for task in "${tasks[@]}"; do
    run="trirelay_${task}_esm650m_ema_relationview_seed42_v100_${tag}"
    "$python_bin" -u -m mfp_nova.cli train \
        --model trirelay --task "$task" --esm 650m --seed 42 \
        --device cuda:0 --precision fp32 --batch-size 64 \
        --ema-decay 0.999 --ema-start-epoch 6 --shared-private \
        --branch-supervision shared_private --evidence-fusion normalized_competitive \
        --relation-view-weight 0.1 --relation-dropout 0.2 \
        --consistency-weight 0 --function-aux-weight 0 \
        --reconstruction-gradient sum --name "$run"
    runs+=("runs/$run")
done
out="runs/trirelay_relationview_seed42_v100_${tag}_val"
"$python_bin" -u -m mfp_nova.branch_evidence --runs "${runs[@]}" \
    --device cuda:0 --batch-size 64 --cpu-threads "$cpu_threads" --output "$out"
printf 'Transfer: %s.zip\n' "$out"
