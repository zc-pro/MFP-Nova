#!/usr/bin/env bash
set -euo pipefail

# Run from the project root. Use a distinct work directory for any missing cache.
run=$1
checkpoint=$2
py=${PYTHON:-python}
primary_gpu=${PRIMARY_GPU:-0}
peptide_gpu=${PEPTIDE_GPU:-1}
data=data/therapeutic_2025
primary=features/esm2_650m
peptide=features/esm2_peptide_150m

mkdir -p "$run"
if [[ ! -f "$checkpoint" ]]; then
  printf 'Checkpoint not found at %s\n' "$checkpoint" >&2
  exit 1
fi
export PYTHONPATH="$PWD/src"

prepare() {
  local view=$1 gpu=$2 cache=$3
  CUDA_VISIBLE_DEVICES="$gpu" "$py" -u -m mfp_nova.cli features \
    --config configs/theraset.yaml --track fine39 --encoder-view "$view" \
    --data-root "$data" --feature-root "$cache" --splits test --device cuda:0
}

jobs_to_wait=()
if [[ ! -f "$primary/test/metadata.json" ]]; then
  primary="$run/cache_primary"
  prepare primary "$primary_gpu" "$primary" >"$run/primary.log" 2>&1 &
  jobs_to_wait+=("$!")
fi
if [[ ! -f "$peptide/test/metadata.json" ]]; then
  peptide="$run/cache_peptide"
  prepare peptide "$peptide_gpu" "$peptide" >"$run/peptide.log" 2>&1 &
  jobs_to_wait+=("$!")
fi
status=0
for job_pid in "${jobs_to_wait[@]}"; do
  wait "$job_pid" || status=$?
done
if (( status != 0 )); then
  echo 'Test cache extraction failed. See the encoder logs in the run directory.'
  exit "$status"
fi

CUDA_VISIBLE_DEVICES="$primary_gpu" "$py" -u -m mfp_nova.cli evaluate \
  --checkpoint "$checkpoint" --split test --device cuda:0 --batch-size 64 \
  --data-root "$data" --feature-root "$primary" --peptide-feature-root "$peptide" \
  --output "$run/theraset_test_metrics.json" --predictions "$run/theraset_test.npz"
printf 'Predictions saved to %s\n' "$run/theraset_test.npz"
