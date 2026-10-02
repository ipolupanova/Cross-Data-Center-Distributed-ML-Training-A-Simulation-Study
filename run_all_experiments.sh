#!/bin/bash
# Runs all four network sweeps; each sweep runs all four schedules back-to-back per value.

set -e

cd "$(dirname "${BASH_SOURCE[0]}")"

run() {
    echo "============================================================"
    echo "Running: $1"
    echo "============================================================"
    bash "$1"
}

run run_latency
run run_jitter
run run_packet_loss
run run_bandwidth

echo "============================================================"
echo "All experiments done."
echo "============================================================"
