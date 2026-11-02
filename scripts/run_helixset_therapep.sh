#!/usr/bin/env bash
# One task per GPU; use three independent nohup train commands.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

action="${1:-}"
task="${2:-}"
python_bin="${PYTHON_BIN:-python}"
run_root="${RUN_ROOT:-runs/helixset_therapep3_seed42_v100}"
case "$run_root" in /*) ;; *) run_root="$PWD/$run_root" ;; esac
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
export OMP_NUM_THREADS="${CPU_THREADS:-6}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS" OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
read -r -a suites <<< "${SUITES:-traditional5 micro_f1}"
feature_root="features/therapep_ai/esm2_650m"
peptide_root="features/therapep_ai/esm2_peptide_150m"

prepare_caches() {
    local view cache split
    for view in primary peptide; do
        if [ "$view" = primary ]; then cache="$feature_root"; else cache="$peptide_root"; fi
        for split in "$@"; do
            if [ -f "$cache/$split/metadata.json" ]; then
                printf 'Reuse cache: %s/%s\n' "$cache" "$split"
                continue
            fi
            "$python_bin" -u -m mfp_nova.cli features --config configs/helixset.yaml \
                --track therapep49 --encoder-view "$view" --splits "$split" \
                --device cuda:0 --batch-size 8
        done
    done
}

case "$action" in
    prepare)
        "$python_bin" -u -m mfp_nova.cli download-therapep
        "$python_bin" -u -m mfp_nova.cli prepare-therapep-variants
        prepare_caches train val
        exit 0 ;;
    prepare-test)
        prepare_caches test
        exit 0 ;;
    pack)
        archive="$run_root/helixset_therapep3_seed42_v100_results_$(date +%Y%m%d_%H%M%S).zip"
        "$python_bin" scripts/pack_helixset_therapep.py "$run_root" "$archive" \
            therapep49,therapep49_positive,therapep48 "$(IFS=,; echo "${suites[*]}")"
        exit 0 ;;
    train|evaluate) ;;
    *) printf 'Usage: bash %s {prepare|prepare-test|train TASK|evaluate TASK|pack}\n' "$0" >&2; exit 2 ;;
esac

case "$task" in
    therapep49) data_root="data/therapep_ai" ;;
    therapep49_positive|therapep48) data_root="data/$task" ;;
    *) printf 'Expected task: therapep49, therapep49_positive or therapep48\n' >&2; exit 2 ;;
esac
mkdir -p "$run_root/$task"
pause_file="$run_root/$task/PAUSE"
printf 'Task=%s GPU=%s Output=%s\n' "$task" "$CUDA_VISIBLE_DEVICES" "$run_root"

for suite in "${suites[@]}"; do
    case "$suite" in traditional5|micro_f1) ;; *) echo "Unknown suite: $suite" >&2; exit 2 ;; esac
    directory="$run_root/$task/$suite"
    if [ "$action" = train ]; then
        if [ -f "$pause_file" ]; then
            printf 'Paused. Remove %s and repeat the same train command.\n' "$pause_file"
            exit 0
        fi
        if [ -f "$directory/results.json" ]; then
            printf 'Completed: %s/%s\n' "$task" "$suite"
            continue
        fi
        args=(train --model helixset --config configs/helixset.yaml --task "$task"
            --data-root "$data_root" --feature-root "$feature_root" --peptide-feature-root "$peptide_root"
            --seed 42 --device cuda:0 --precision fp32 --batch-size 16 --accumulation-steps 4
            --num-workers "${NUM_WORKERS:-2}" --rdrop-weight 0.1 --ema-decay 0.999 --ema-start-epoch 6
            --metric-suite "$suite" --threshold-strategy global --epochs 100 --name "$directory"
            --stop-file "$pause_file")
        if [ -f "$directory/last.pt" ]; then
            args+=(--resume "$directory/last.pt" --resume-in-place)
        elif [ -d "$directory" ]; then
            args+=(--resume-in-place)
        fi
        printf 'Training: %s/%s -> %s\n' "$task" "$suite" "$directory"
        "$python_bin" -u -m mfp_nova.cli "${args[@]}" 2>&1 | tee -a "$run_root/${task}_${suite}.log"
        if [ ! -f "$directory/results.json" ]; then
            if [ -f "$pause_file" ]; then
                echo 'Paused after the last complete epoch.'
                exit 0
            fi
            echo "Training ended without results.json: $directory" >&2
            exit 1
        fi
    else
        if [ ! -f "$directory/results.json" ]; then
            echo "Training is not complete: $directory" >&2
            exit 1
        fi
        for split in val test; do
            if [ -f "$directory/$split.json" ] && [ -f "$directory/${split}_predictions.npz" ]; then
                printf 'Already evaluated: %s/%s/%s\n' "$task" "$suite" "$split"
                continue
            fi
            "$python_bin" -u -m mfp_nova.cli evaluate --checkpoint "$directory/best.pt" \
                --data-root "$data_root" --feature-root "$feature_root" --peptide-feature-root "$peptide_root" \
                --split "$split" --device cuda:0 --batch-size 16 \
                --output "$directory/$split.json" --predictions "$directory/${split}_predictions.npz"
            printf 'Evaluated: %s/%s/%s\n' "$task" "$suite" "$split"
        done
    fi
done
