# Settings for the two VMs. Every run script reads this file.
# Run `bash check_setup.sh` on the master VM: it prints the resolved values, verifies them,
# and lists candidate interfaces / torchrun paths if something could not be found.
# (no spaces in any of the paths)

# MASTER = the VM you start the run scripts on (rank 0), WORKER = the other VM (rank 1).
WORKER_ADDR=31.22.104.93
SSH_USER=root
MASTER_PORT=12356

# "auto" detects the value; replace it with a fixed value if detection picks the wrong one.
#   MASTER_ADDR: the master's source IP on the route to the worker (must be reachable from the worker)
#   MASTER_IF / WORKER_IF: the interface that route uses; netem is applied to MASTER_IF egress only
#   TORCHRUN_*: the torchrun of the Python environment with torch + CUDA (auto = torchrun on PATH)
MASTER_ADDR=auto
MASTER_IF=auto
WORKER_IF=auto
TORCHRUN_MASTER=auto
TORCHRUN_WORKER=auto

# directory on the worker the one_iteration_*.py scripts are copied to (created if missing)
WORKER_DIR=/root/layer2_experiments

# tc needs root: use TC="sudo tc" if you are not root on the master
TC=tc

SSH_OPTS="-o BatchMode=yes -o ConnectTimeout=10"

# hang detection: a run is killed after
#   MIN_TIMEOUT + 13 iterations × transfers per iteration × SEC_PER_TRANSFER  seconds
# (bandwidth sweeps use the transfer time at the set rate instead of SEC_PER_TRANSFER)
MIN_TIMEOUT=300
SEC_PER_TRANSFER=30


# ---- resolve "auto" values (nothing to edit below) ----
route_field() {   # route_field <src|dev>, reads `ip route get` output on stdin
    awk -v k="$1" '{for (i = 1; i < NF; i++) if ($i == k) { print $(i + 1); exit }}'
}
[ "$MASTER_ADDR" = auto ] && MASTER_ADDR=$(ip route get "$WORKER_ADDR" 2>/dev/null | route_field src)
[ "$MASTER_IF" = auto ] && MASTER_IF=$(ip route get "$WORKER_ADDR" 2>/dev/null | route_field dev)
[ "$WORKER_IF" = auto ] && WORKER_IF=$(ssh $SSH_OPTS "$SSH_USER@$WORKER_ADDR" "ip route get $MASTER_ADDR" 2>/dev/null | route_field dev)
# torchrun on PATH, else the first /root/<env>/bin/torchrun (e.g. /root/venv)
FIND_TORCHRUN='{ command -v torchrun || ls -d /root/*/bin/torchrun /root/.*/bin/torchrun; } 2>/dev/null | head -n 1'
[ "$TORCHRUN_MASTER" = auto ] && TORCHRUN_MASTER=$(bash -lc "$FIND_TORCHRUN")
[ "$TORCHRUN_WORKER" = auto ] && TORCHRUN_WORKER=$(ssh $SSH_OPTS "$SSH_USER@$WORKER_ADDR" "bash -lc \"$FIND_TORCHRUN\"" 2>/dev/null)
