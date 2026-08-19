#!/usr/bin/env bash

set -u

script_dir="$(cd "$(dirname "$0")" && pwd)"
repo_dir="$(cd "$script_dir/../../.." && pwd)/UAI-MP-AIF-JAX"
python="$repo_dir/.venv/bin/python"
logs_dir="$script_dir/examples/logs"
results_dir="$script_dir/examples/results"
summary="$script_dir/examples/summary.tsv"
methods=(
  loopy-vbp
  loopy
  region-extended
  dyn-channel
  nuijten
  vbp-channel
  precise-info-seeking
  active-inference
)

mkdir -p "$logs_dir" "$results_dir"
printf 'environment\tmethod\texit_code\treal_seconds\tresult_json\tlog\n' > "$summary"

run_case() {
  local environment="$1"
  local method="$2"
  shift 2

  local result="$results_dir/${environment}__${method}.json"
  local log="$logs_dir/${environment}__${method}.log"

  {
    printf 'environment: %s\n' "$environment"
    printf 'method: %s\n' "$method"
    printf 'command:'
    printf ' %q' "$python" "$@" --planning-method "$method" --output "$result"
    printf '\n\n'
  } > "$log"

  /usr/bin/time -p "$python" "$@" \
    --planning-method "$method" \
    --output "$result" >> "$log" 2>&1
  local status=$?
  local seconds
  seconds="$(awk '$1 == "real" { value = $2 } END { print value }' "$log")"

  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$environment" "$method" "$status" "$seconds" \
    "examples/results/${environment}__${method}.json" \
    "examples/logs/${environment}__${method}.log" >> "$summary"

  printf '%-14s %-24s exit=%s real=%ss\n' \
    "$environment" "$method" "$status" "$seconds"
}

cd "$repo_dir"

for method in "${methods[@]}"; do
  run_case frozen_lake "$method" run_frozen_lake.py \
    --grid-size 3 --n-configs 3 --hole-fraction 0.2 --min-hamming 0 \
    --slip-prob 0.0 --episodes 1 --max-steps 3 \
    --planning-horizon 2 --planning-iterations 2 --damping 0.5 \
    --receding-horizon --seed 0
done
for method in "${methods[@]}"; do
  run_case wumpus_world "$method" run_wumpus_world.py \
    --grid-size 3 --n-configs 3 --n-pits 1 \
    --slip-prob 0.0 --episodes 1 --max-steps 3 \
    --planning-horizon 2 --planning-iterations 2 --damping 0.5 \
    --receding-horizon --seed 0
done

for method in "${methods[@]}"; do
  run_case rocksample "$method" run_rocksample.py \
    --grid-size 3 --n-rocks 1 --slip-prob 0.0 \
    --episodes 1 --max-steps 3 --planning-horizon 2 \
    --planning-iterations 2 --damping 0.5 --terminal-goal-only \
    --receding-horizon --seed 0
done

for method in "${methods[@]}"; do
  run_case minigrid "$method" run_minigrid.py \
    --grid-size 3 --fov-size 3 --episodes 1 --max-steps 3 \
    --planning-horizon 2 --inference-iterations 2 \
    --planning-iterations 2 --damping 0.5 --receding-horizon --seed 0
done
