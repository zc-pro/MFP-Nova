#!/usr/bin/env bash
# Run one or more original paper experiments sequentially on the selected V100.
set -eo pipefail
kit=$(cd "$(dirname "$0")" && pwd)
root=${PAPER_ROOT:-"$HOME/paper-repro"}
source "$(conda info --base)/etc/profile.d/conda.sh"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-$OMP_NUM_THREADS}
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1

if (( $# == 0 )); then
    echo "Choose: TPpred-LE AMCL iMFP-LG SA-MTP METFAN" >&2
    exit 2
fi

mkdir -p "$root/results"
for paper in "$@"; do
    cd "$root/$paper"
    case "$paper" in
        TPpred-LE|SA-MTP) conda activate repro-mtp ;;
        AMCL|METFAN) conda activate repro-modern ;;
        iMFP-LG) conda activate repro-imfp ;;
        *) echo "Unknown paper: $paper" >&2; exit 2 ;;
    esac
    run=$(mktemp -d "$root/results/$paper.XXXXXX")
    echo "Starting $paper; output $run"
    (
        export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
        git rev-parse HEAD
        python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))'
        case "$paper" in
            TPpred-LE)
                python main.py -act e -task3 released -pth3 trained_models/TPpred_tw_60_1.pth -result_folder "$run"
                python main.py -act t e -seed 46 -task stageA -pth "$run/stageA.pth" -result_folder "$run"
                python main.py -act rt e -seed 46 -s square -task2 stageB -pth "$run/stageA.pth" -pth2 "$run/stageB.pth" -result_folder "$run"
                ;;
            AMCL)
                mkdir -p result
                python pep_main.py -task published_test
                python "$kit/train_amcl.py" -task reconstructed_train -model_path "$run/amcl.pth"
                ;;
            iMFP-LG)
                cd Task
                python - "$run" <<'PY'
import random, sys
from pathlib import Path
import numpy as np
import torch
from TaskForMultiPeptide_MFTP import TrainConfig, train
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
cfg = TrainConfig(0)
cfg.model_save_path = str(Path(sys.argv[1]) / "model0.bin")
train(cfg)
PY
                ;;
            SA-MTP)
                mkdir -p logs trained_models features
                python main.py -act t e -seed 50 -task original_mtp15 -pth "$run/sa_mtp.pth" -result_folder "$run"
                ;;
            METFAN)
                git apply --ignore-space-change --reverse --check "$kit/metfan_traditional5.patch" 2>/dev/null ||
                    git apply --ignore-space-change --reverse --check "$kit/metfan_therapep.patch" 2>/dev/null ||
                    git apply --ignore-space-change --reverse --check "$kit/metfan_val_thresholds.patch"
                # The public text files are at repo root; load_dataset expects split subdirectories.
                for split in train val test; do
                    mkdir -p "$run/data/$split"
                    ln -s "$PWD/$split.txt" "$run/data/$split/$split.txt"
                done
                if [[ -z "${METFAN_FEATURES:-}" ]]; then
                    export METFAN_FEATURES="$run/features"
                    python "$kit/metfan_features.py" esm2_mean "${METFAN_ESM2:-facebook/esm2_t33_650M_UR50D}"
                    python "$kit/metfan_features.py" prot_t5_mean "${METFAN_PROTT5:-Rostlab/prot_t5_xl_half_uniref50-enc}"
                fi
                python -c 'from main import main; from config import get_args; main(get_args())' \
                    --data_dir "$run/data" --result_dir "$run" --features_dir "$METFAN_FEATURES" \
                    --mode train --seed 42 --features textcnn esm2_mean prot_t5_mean --optimize_features
                ;;
        esac
    ) >"$run/run.log" 2>&1
    echo "Completed $paper; output $run"
done
