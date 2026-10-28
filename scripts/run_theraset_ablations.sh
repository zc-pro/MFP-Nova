#!/usr/bin/env bash
# Three independent V100 task queues, one GPU per task; run this script with nohup.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
python_bin="${PYTHON_BIN:-python}"
run_root="${RUN_ROOT:-runs/theraset_ablation_seed42_v100}"
cpu_threads="${CPU_THREADS:-4}"
read -r -a gpu_ids <<< "${GPUS:-0 1 2}"
tasks=(major15 fine39 therapep49)
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="$cpu_threads" MKL_NUM_THREADS="$cpu_threads" OPENBLAS_NUM_THREADS="$cpu_threads"
mkdir -p "$run_root"
if [ -f "$run_root/PAUSE" ]; then
    printf 'Queue paused. Remove %s/PAUSE before restarting.\n' "$run_root"
    exit 0
fi

# Major15 and fine39 share caches. Prepare once to avoid concurrent writes.
CUDA_VISIBLE_DEVICES="${gpu_ids[0]}" "$python_bin" -u -m mfp_nova.theraset_ablation prepare \
    --tasks "${tasks[@]}" --device cuda:0 --cpu-threads "$cpu_threads"

pids=()
for index in 0 1 2; do
    task="${tasks[$index]}"
    CUDA_VISIBLE_DEVICES="${gpu_ids[$index]}" "$python_bin" -u -m mfp_nova.theraset_ablation run \
        --tasks "$task" --device cuda:0 --cpu-threads "$cpu_threads" \
        --output "$run_root" >> "$run_root/${task}.log" 2>&1 &
    pids+=("$!")
    printf '%s GPU=%s PID=%s Log=%s/%s.log\n' "$task" "${gpu_ids[$index]}" "$!" "$run_root" "$task"
done

exit_code=0
for pid in "${pids[@]}"; do
    wait "$pid" || exit_code=$?
done
exit "$exit_code"
