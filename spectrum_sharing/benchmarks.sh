#!/bin/bash
CONFIG_NAME="simulation5"

# Trap signals and clean up
cleanup() {
  echo "Caught signal... terminating background jobs"
  pkill -P $$
  exit 1
}

trap cleanup SIGINT SIGTERM

run_batch() {
  local arg=$1
  for i in {0..3}; do
    python3 -m spectrum_sharing.benchmark $i $arg &
  done
  wait
}

# Usage for different random seeds
for j in {0..45}; do
    echo "Running batch for random seed $j"
    run_batch $j
    echo "Merging CSVs for seed $j"
    # Aggregate safely and avoid duplicate headers
    temp_files=(spectrum_sharing/Tests/temp_seed${j}_*.csv)

    if [ ${#temp_files[@]} -gt 0 ]; then
        output="spectrum_sharing/Tests/aggregated_results_${CONFIG_NAME}.csv"

        if [ ! -f "$output" ]; then
            # Write header from the first file
            head -n 1 "${temp_files[0]}" > "$output"
        fi

        # Append contents without header
        tail -n +2 -q "${temp_files[@]}" >> "$output"

        # Clean up
        rm "${temp_files[@]}"
    else
        echo "⚠️  No temp files found for seed $j"
    fi
  done
