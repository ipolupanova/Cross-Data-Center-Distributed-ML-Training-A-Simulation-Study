#!/bin/bash
# Checks vm_config.sh before running experiments. Run it on the master VM.
# Prints the resolved settings; if something is missing it lists candidates to put into vm_config.sh.

cd "$(dirname "${BASH_SOURCE[0]}")"
source ./vm_config.sh

OK=true
pass() { echo "  ok    $*"; }
fail() { echo "  FAIL  $*"; OK=false; }

# prints interfaces and torchrun candidates of the machine it runs on
FIND_CANDIDATES='
echo "  interfaces:"; ip -o -4 addr show | awk "{print \"    \" \$2 \"  \" \$4}"
echo "  torchrun candidates:"
{ command -v torchrun; ls -d /root/*/bin/torchrun /root/.*/bin/torchrun /home/*/*/bin/torchrun /opt/*/bin/torchrun /opt/conda/bin/torchrun 2>/dev/null; } | sort -u | sed "s/^/    /"
'

# checks that a torchrun exists and its python has torch with CUDA
CHECK_TORCHRUN='
if [ ! -x "$1" ]; then echo "missing"; exit; fi
"$(dirname "$1")/python" -c "import torch; print(\"torch\", torch.__version__, \"cuda\" if torch.cuda.is_available() else \"NO-CUDA\")" 2>&1 | tail -n 1
'

echo "== resolved settings"
for v in WORKER_ADDR MASTER_ADDR MASTER_IF WORKER_IF TORCHRUN_MASTER TORCHRUN_WORKER WORKER_DIR; do
    printf "  %-16s %s\n" "$v" "${!v:-<not found>}"
done

echo "== master (this VM)"
bash -c "$FIND_CANDIDATES"
[ -n "$MASTER_ADDR" ] && pass "MASTER_ADDR=$MASTER_ADDR" || fail "MASTER_ADDR not found (no route to $WORKER_ADDR?)"
ip link show "$MASTER_IF" >/dev/null 2>&1 && pass "interface $MASTER_IF" || fail "interface '$MASTER_IF' not found"
if $TC qdisc show dev "$MASTER_IF" >/dev/null 2>&1; then pass "tc works"; else fail "'$TC' does not work (not installed or not root?)"; fi
command -v fuser >/dev/null && pass "fuser found" || fail "fuser not found (apt install psmisc)"
R=$(bash -c "$CHECK_TORCHRUN" _ "$TORCHRUN_MASTER")
[[ "$R" == *" cuda" ]] && pass "TORCHRUN_MASTER: $R" || fail "TORCHRUN_MASTER='$TORCHRUN_MASTER': $R (pick one of the candidates above)"

echo "== worker ($WORKER_ADDR)"
if ! ssh $SSH_OPTS "$SSH_USER@$WORKER_ADDR" true 2>/dev/null; then
    fail "cannot ssh to $SSH_USER@$WORKER_ADDR without a password (ssh-keygen, then ssh-copy-id $SSH_USER@$WORKER_ADDR)"
else
    pass "ssh $SSH_USER@$WORKER_ADDR"
    ssh $SSH_OPTS "$SSH_USER@$WORKER_ADDR" "$FIND_CANDIDATES"
    ssh $SSH_OPTS "$SSH_USER@$WORKER_ADDR" "ip link show $WORKER_IF" >/dev/null 2>&1 \
        && pass "interface $WORKER_IF" || fail "interface '$WORKER_IF' not found on worker"
    ssh $SSH_OPTS "$SSH_USER@$WORKER_ADDR" "mkdir -p $WORKER_DIR && test -w $WORKER_DIR" \
        && pass "WORKER_DIR $WORKER_DIR writable" || fail "cannot create/write $WORKER_DIR on worker"
    R=$(ssh $SSH_OPTS "$SSH_USER@$WORKER_ADDR" "bash -s -- $TORCHRUN_WORKER" <<< "$CHECK_TORCHRUN")
    [[ "$R" == *" cuda" ]] && pass "TORCHRUN_WORKER: $R" || fail "TORCHRUN_WORKER='$TORCHRUN_WORKER': $R (pick one of the candidates above)"
    if [ -n "$MASTER_ADDR" ]; then
        ssh $SSH_OPTS "$SSH_USER@$WORKER_ADDR" "ping -c 1 -W 2 $MASTER_ADDR" >/dev/null 2>&1 \
            && pass "worker reaches $MASTER_ADDR" || fail "worker cannot ping $MASTER_ADDR (set MASTER_ADDR to the master's public IP?)"
    fi
fi

echo
$OK && echo "All checks passed." || { echo "Fix the FAIL lines in vm_config.sh and run again."; exit 1; }
