#!/usr/bin/env bash
# Run with nohup; GPUS='2 3' assigns one suite to each GPU, GPUS='2' is sequential.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
action="${1:-run}"
python_bin="${PYTHON_BIN:-python}"
run_root="${RUN_ROOT:-runs/theraset_followup_seed42_v100}"
reuse_root="${REUSE_ROOT:-runs/theraset_ablation_seed42_v100}"
cpu_threads="${CPU_THREADS:-4}"
read -r -a gpu_ids <<< "${GPUS:-2 3}"
read -r -a tasks <<< "${TASKS:-fine39}"
read -r -a suites <<< "${SUITES:-traditional5 micro_f1}"
read -r -a variants <<< "${VARIANTS:-no_label_context_single_prototype}"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8
export TZ="${TZ:-Asia/Shanghai}"
export OMP_NUM_THREADS="$cpu_threads" MKL_NUM_THREADS="$cpu_threads" OPENBLAS_NUM_THREADS="$cpu_threads"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
common=(--output "$run_root" --reuse-root "$reuse_root" --tasks "${tasks[@]}"
        --variants "${variants[@]}" --device cuda:0 --cpu-threads "$cpu_threads" --platform v100)

case "$action" in
    plan|pause|pack)
        "$python_bin" -u -m mfp_nova.theraset_ablation "$action" "${common[@]}" --suites "${suites[@]}"
        exit
        ;;
    run|diagnose) ;;
    *) printf 'Use plan, run, pause, pack, or diagnose.\n' >&2; exit 2 ;;
esac
mkdir -p "$run_root"
extra=()
if [ "$action" = run ]; then extra+=(--diagnostics); fi
if [ "${#gpu_ids[@]}" -eq 1 ]; then
    CUDA_VISIBLE_DEVICES="${gpu_ids[0]}" "$python_bin" -u -m mfp_nova.theraset_ablation "$action" \
        "${common[@]}" --suites "${suites[@]}" "${extra[@]}"
    exit
fi

pids=()
for index in "${!suites[@]}"; do
    suite="${suites[$index]}"
    gpu="${gpu_ids[$index]}"
    log="$run_root/${suite}_${action}.log"
    CUDA_VISIBLE_DEVICES="$gpu" "$python_bin" -u -m mfp_nova.theraset_ablation "$action" \
        "${common[@]}" --suites "$suite" "${extra[@]}" >> "$log" 2>&1 &
    pids+=("$!")
    printf '%s GPU=%s PID=%s Log=%s\n' "$suite" "$gpu" "$!" "$log"
done
status=0
for pid in "${pids[@]}"; do
    wait "$pid" || status=$?
done
exit "$status"
