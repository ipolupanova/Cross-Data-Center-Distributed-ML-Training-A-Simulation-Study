#!/bin/bash
# Runs one sweep of a network factor on the two VMs from vm_config.sh.
# For every value of the factor, all schedules run back-to-back under the same netem setting,
# so drift of the (fluctuating) link between the VMs affects all schedules alike. The schedule
# order rotates from value to value, so no schedule always runs first.
#
#   usage: bash run_experiment.sh <latency|jitter|packet_loss|bandwidth> [schedule ...]
#          schedules default to: gpipe 1f1b interleaved zbh1
#
# Rank 0 runs on this VM (the master), rank 1 on the worker via ssh. netem is applied to the
# master's egress only (rank0 -> rank1). Results go to results/run_<schedule>_<factor>.csv
# (one file per schedule), the full rank-0 output of every run to results/logs/.

cd "$(dirname "${BASH_SOURCE[0]}")"
source ./vm_config.sh

FACTOR=$1
shift
SCHEDULES=("$@")
[ ${#SCHEDULES[@]} -eq 0 ] && SCHEDULES=(gpipe 1f1b interleaved zbh1)

LOG_DIR=./results/logs
ITERS=13          # 3 warmup + 10 measured steps (N_STEPS in the one_iteration scripts)
BASE_DELAY=10     # ms, jitter is added on top of this delay

# sweep values and CSV header per factor: the values every old schedule was tested at
# (drops the extreme tails latency 400-500 ms, jitter 7-8 ms, bandwidth 10-25 Mbit)
case "$FACTOR" in
    latency)     VALUES=(0 10 25 50 100 200 300);          HEADER=latency_ms ;;
    jitter)      VALUES=(0 1 2 3 4 5 6);                   HEADER=jitter_ms ;;
    packet_loss) VALUES=(0 0.25 0.5 0.75 1);               HEADER=loss_pct ;;
    bandwidth)   VALUES=(50 100 200 400 600 800 1000);     HEADER=bandwidth_mbps ;;
    *) echo "usage: bash run_experiment.sh <latency|jitter|packet_loss|bandwidth> [schedule ...]"; exit 1 ;;
esac

# 12 MiB transfers per iteration (both directions), used to size the timeout
transfers() {
    case "$1" in
        gpipe|1f1b|zbh1) echo 8 ;;
        interleaved)     echo 24 ;;
    esac
}

for s in "${SCHEDULES[@]}"; do
    [ -n "$(transfers "$s")" ] || { echo "unknown schedule '$s'"; exit 1; }
    [ -f "one_iteration_${s}.py" ] || { echo "missing one_iteration_${s}.py"; exit 1; }
done
for v in MASTER_ADDR WORKER_ADDR MASTER_IF WORKER_IF TORCHRUN_MASTER TORCHRUN_WORKER; do
    if [ -z "${!v}" ]; then
        echo "$v could not be resolved, set it in vm_config.sh (run check_setup.sh for hints)"; exit 1
    fi
done


# netem arguments for one sweep value (empty = no qdisc)
netem_args() {
    case "$FACTOR" in
        latency)     [ "$1" != 0 ] && echo "delay ${1}ms" ;;
        jitter)      if [ "$1" != 0 ]; then echo "delay ${BASE_DELAY}ms ${1}ms"; else echo "delay ${BASE_DELAY}ms"; fi ;;
        packet_loss) [ "$1" != 0 ] && echo "loss ${1}%" ;;
        bandwidth)   echo "rate ${1}mbit" ;;
    esac
}

# seconds after which a run counts as hung: run_timeout <schedule> <value>
run_timeout() {
    local per_transfer=$SEC_PER_TRANSFER
    if [ "$FACTOR" = bandwidth ]; then
        per_transfer=$(( 101 / $2 + 5 ))   # 12 MiB = 101 Mbit at $2 Mbit/s, + slack
    fi
    echo $(( MIN_TIMEOUT + ITERS * $(transfers "$1") * per_transfer ))
}

worker() {
    ssh $SSH_OPTS "$SSH_USER@$WORKER_ADDR" "$@"
}

kill_runs() {
    pkill -9 -f 'one_iteration_[a-z0-9]*\.py' 2>/dev/null
    worker "pkill -9 -f 'one_iteration_[a-z0-9]*\.py'" 2>/dev/null
    true
}

clear_netem() {
    $TC qdisc del dev "$MASTER_IF" root 2>/dev/null
    true
}

# one run of one schedule under the netem setting that is currently applied: run_one <schedule> <value>
run_one() {
    local s=$1 val=$2
    local script=one_iteration_${s}.py
    local name=run_${s}_${FACTOR}
    local log=$LOG_DIR/${name}_${val}.log
    local worker_log=/tmp/${name}_rank1.log
    local timeout_s
    timeout_s=$(run_timeout "$s" "$val")

    echo "------------------------------------------------------------"
    echo "$s | $FACTOR = $val | timeout ${timeout_s}s | rank 0 log: $log"

    kill_runs
    sleep 2
    fuser -k ${MASTER_PORT}/tcp 2>/dev/null
    sleep 1

    # rank 0 on this VM; pipefail so that wait returns the exit code of timeout, not of tee
    ( set -o pipefail
      NCCL_SOCKET_IFNAME=$MASTER_IF NCCL_DEBUG=WARN \
      timeout --kill-after=10 $timeout_s $TORCHRUN_MASTER \
          --nproc_per_node=1 --nnodes=2 --node_rank=0 \
          --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
          "$script" 2>&1 | tee "$log" ) &
    local rank0_pid=$!

    sleep 2

    # rank 1 on the worker VM
    worker "cd $WORKER_DIR && \
        nohup env NCCL_SOCKET_IFNAME=$WORKER_IF NCCL_DEBUG=WARN \
        timeout --kill-after=10 $timeout_s $TORCHRUN_WORKER \
        --nproc_per_node=1 --nnodes=2 --node_rank=1 \
        --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
        $script > $worker_log 2>&1 &"

    wait $rank0_pid
    local rank0_exit=$?
    if [ $rank0_exit -eq 124 ] || [ $rank0_exit -eq 137 ]; then
        echo "WARNING: rank 0 TIMED OUT after ${timeout_s}s (likely hang/deadlock)"
    elif [ $rank0_exit -ne 0 ]; then
        echo "WARNING: rank 0 exited with code $rank0_exit"
    fi

    # clean up the worker in case it is still stuck
    kill_runs

    local time bubble
    time=$(grep "avg iter time" "$log" | awk '{print $(NF-1)}')
    bubble=$(grep "bubble_ratio" "$log" | awk '{print $NF}')

    if [ -z "$time" ]; then
        echo "no result, last lines of the rank 1 log:"
        worker "tail -n 20 $worker_log" 2>/dev/null
    fi

    echo "${val},${time:-TIMEOUT},${bubble:-NA}" >> "./results/${name}.csv"
    echo "$s | $FACTOR $val -> iter time ${time:-TIMEOUT} ms  bubble_ratio=${bubble:-NA}"
}


# copy the schedules to the worker
SCRIPTS=()
for s in "${SCHEDULES[@]}"; do SCRIPTS+=("one_iteration_${s}.py"); done
worker "mkdir -p $WORKER_DIR" && scp $SSH_OPTS "${SCRIPTS[@]}" "$SSH_USER@$WORKER_ADDR:$WORKER_DIR/" \
    || { echo "could not copy the schedules to $WORKER_ADDR:$WORKER_DIR"; exit 1; }

trap 'clear_netem; kill_runs' EXIT
trap 'exit 130' INT TERM
clear_netem

mkdir -p ./results "$LOG_DIR"
for s in "${SCHEDULES[@]}"; do
    echo "${HEADER},iter_time_ms,bubble_ratio" > "./results/run_${s}_${FACTOR}.csv"
done

N=${#SCHEDULES[@]}
for i in "${!VALUES[@]}"; do
    val=${VALUES[$i]}
    echo "============================================================"
    echo "$FACTOR = $val"

    clear_netem
    NETEM=$(netem_args "$val")
    if [ -n "$NETEM" ]; then
        echo "applying netem $NETEM on $MASTER_IF"
        $TC qdisc add dev "$MASTER_IF" root netem $NETEM || { echo "tc failed"; exit 1; }
    fi

    # all schedules under this setting, starting with a different one for each value
    for j in $(seq 0 $(( N - 1 ))); do
        run_one "${SCHEDULES[$(( (i + j) % N ))]}" "$val"
    done

    clear_netem
    sleep 2
done

echo "Results saved to results/run_<schedule>_${FACTOR}.csv"
