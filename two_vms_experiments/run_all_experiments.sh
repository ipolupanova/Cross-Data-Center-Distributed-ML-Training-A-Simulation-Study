#!/bin/bash

# Run all NCCL experiments sequentially (all use tc on eth0 so can't parallelize)
set -e

cd "$(dirname "${BASH_SOURCE[0]}")"

run() {
    echo "============================================================"
    echo "Running: $1"
    echo "============================================================"
    bash "$1"
}

# AllReduce
run CCT_latency
run CCT_jitter
run CCT_packet_loss
#run CCT_bandwidth_clouds

run CCT_latency_64M
run CCT_jitter_64M
run CCT_packet_loss_64M
#run CCT_bandwidth_64M

run CCT_latency_128M
run CCT_jitter_128M
run CCT_packet_loss_128M
#run CCT_bandwidth_128M

# SendRecv
run CCT_sendrecv_latency 
run CCT_sendrecv_jitter
run CCT_sendrecv_packet_loss
#run CCT_sendrecv_bandwidth

run CCT_sendrecv_latency_64M
run CCT_sendrecv_jitter_64M
run CCT_sendrecv_packet_loss_64M
#run CCT_sendrecv_bandwidth_64M

run CCT_sendrecv_latency_128M
run CCT_sendrecv_jitter_128M
run CCT_sendrecv_packet_loss_128M
#run CCT_sendrecv_bandwidth_128M

echo "============================================================"
echo "All experiments done. Generating plots..."
echo "============================================================"
python3 script_plot_all.py
