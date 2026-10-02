"""
Synthetic ZB-H1 (Zero Bubble) Pipeline Parallel Kernel
======================================================
nanotron 109M LLaMA, pp=2, tp=1, dp=1

Config:
  p = 2   pipeline stages
  m = 4   microbatches


Layer split:
  GPU 0: embedding + layers 0-5
  GPU 1: layers 6-11 + LM head


Activation tensor: (16, 512, 768) bf16 = 12 MiB per transfer
(m=4 microbatches of 16 = 64 samples per iteration, as in the Nsight profile)


Compute times from Nsight (micro_batch=64), scaled linearly to micro_batch=16,
backward split into B (input gradient, on the critical path) and W (weight gradient, local only):
  FORWARD_US  = 18,814 / 4 = 4,703.5 μs per microbatch
  B_US        = 16,012 / 4 = 4,003 μs per microbatch     (55% of backward)
  W_US        = 13,101 / 4 = 3,275.25 μs per microbatch  (45% of backward)


Schedule (ZB-H1, p=2, m=4): same F/B order and warm-up as 1F1B, but W is
split off and scheduled to fill bubbles. Like 1F1B, at most p=2 microbatches
are in flight per rank (F done, W not done), so peak memory matches 1F1B.

  Rank 0                                      Rank 1

  ----- warm-up -----
  F0
  send A0 ----------------------------------> recv A0
  F1                                          F0
                                              B0
  ----- steady state -----
  send A1 / recv G0  <---- paired P2P ----->  send G0 / recv A1
  B0                                          F1
  W0                                          B1
  F2                                          W0      # fills the wait for A2
  send A2 / recv G1  <---- paired P2P ----->  send G1 / recv A2
  B1                                          F2
  W1                                          B2
  F3                                          W1
  send A3 / recv G2  <---- paired P2P ----->  send G2 / recv A3
  B2                                          F3
  W2      # fills the wait for G3             B3
  ----- cooldown -----
  recv G3 <---------------------------------- send G3
  B3                                          W2
  W3                                          W3

The steady-state exchange must be a paired batch_isend_irecv: both ranks
send first, so blocking send/recv would deadlock on 12 MiB messages.

Bubble on rank 0: waits B for G0 and F-W for G3  ->  (p-1)(F+B-W)
vs 1F1B: (p-1)(F+B+W)

"""

import os
import time
import torch
import torch.distributed as dist

MICRO_BATCH = 16    # 4 × 16 = 64 samples per iteration
SEQ_LEN     = 512
HIDDEN      = 768
DTYPE       = torch.bfloat16
N_STEPS     = 10

PP_STAGES      = 2    # p
N_MICROBATCHES = 4    # m

FORWARD_US = 18_814 * MICRO_BATCH / 64  # Nsight value at micro_batch=64, scaled to MICRO_BATCH
B_US       = 16_012 * MICRO_BATCH / 64   # input gradient
W_US       = 13_101 * MICRO_BATCH / 64   # weight gradient

# ZB-H1 bubble (p-1)(F+B-W), as a fraction of the ideal compute m(F+B+W)
BUBBLE_FRACTION = (PP_STAGES - 1) * (FORWARD_US + B_US - W_US) / (N_MICROBATCHES * (FORWARD_US + B_US + W_US))

ACTIVATION_SHAPE = (MICRO_BATCH, SEQ_LEN, HIDDEN)
ACTIVATION_MB    = MICRO_BATCH * SEQ_LEN * HIDDEN * 2 / 1024**2  # 12 MiB


# GPU clock cycles per μs, measured at startup by calibrate_sleep()
CYCLES_PER_US = None


def time_sleep_ms(cycles: int) -> float:
    """Runs torch.cuda._sleep(cycles) and returns its GPU duration in ms."""
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    start.record()
    torch.cuda._sleep(cycles)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end)


def calibrate_sleep(n_trials: int = 5) -> float:
    """_sleep spins for a number of GPU clock cycles, not a duration,
    so measure how many cycles make up one μs on this GPU."""
    cycles = 100_000_000
    time_sleep_ms(cycles)  # warm up (GPU clocks ramp up under load)
    rates = sorted(cycles / (time_sleep_ms(cycles) * 1000) for _ in range(n_trials))
    return rates[len(rates) // 2]  # median


#kernel that simulates the computation
def fake_compute(duration_us: float):
    """Burns GPU time without real math. _sleep takes GPU clock cycles."""
    torch.cuda._sleep(int(duration_us * CYCLES_PER_US))


def p2p(send_buf=None, send_dst=None, recv_buf=None, recv_src=None):
    """Issues a send and/or recv together (paired P2P) and waits for both."""
    ops = []
    if send_buf is not None:
        ops.append(dist.P2POp(dist.isend, send_buf, send_dst))
    if recv_buf is not None:
        ops.append(dist.P2POp(dist.irecv, recv_buf, recv_src))
    if not ops:
        return
    for r in dist.batch_isend_irecv(ops):
        r.wait()


#iteration according to zb-h1 schedule with 2 ranks, 4 microbatches
# bufs[i] holds activation A(i) on the way forward and gradient G(i) on the way back
def iteration(rank: int, bufs: list):
    m = N_MICROBATCHES

    if rank == 0:
        # warm-up: F0
        fake_compute(FORWARD_US)
        p2p(send_buf=bufs[0], send_dst=1)

        # steady state: F(i), send A(i) + recv G(i-1), B(i-1), W(i-1)
        for i in range(1, m):
            fake_compute(FORWARD_US)
            p2p(send_buf=bufs[i], send_dst=1, recv_buf=bufs[i - 1], recv_src=1)
            fake_compute(B_US)
            fake_compute(W_US)

        # cooldown: recv G(m-1), B(m-1), W(m-1)
        p2p(recv_buf=bufs[m - 1], recv_src=1)
        fake_compute(B_US)
        fake_compute(W_US)

    else:  # rank == 1
        # warm-up: F0, B0
        p2p(recv_buf=bufs[0], recv_src=0)
        fake_compute(FORWARD_US)
        fake_compute(B_US)

        # steady state: send G(i-1) + recv A(i), F(i), B(i), then W(i-1) while rank 0 is still busy
        for i in range(1, m):
            p2p(send_buf=bufs[i - 1], send_dst=0, recv_buf=bufs[i], recv_src=0)
            fake_compute(FORWARD_US)
            fake_compute(B_US)
            if i < m - 1:
                fake_compute(W_US)

        # cooldown: send G(m-1) first so rank 0 is not delayed, then the last two W
        p2p(send_buf=bufs[m - 1], send_dst=0)
        fake_compute(W_US)
        fake_compute(W_US)

    torch.cuda.synchronize()


def main():
    global CYCLES_PER_US

    #setup
    print("started main", flush=True)
    dist.init_process_group(backend="nccl")
    rank       = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    assert world_size == 2, f"Need exactly 2 ranks, got {world_size}"

    # each rank calibrates its own GPU, then checks the sleep durations
    CYCLES_PER_US = calibrate_sleep()
    f_ms = time_sleep_ms(int(FORWARD_US * CYCLES_PER_US))
    b_ms = time_sleep_ms(int(B_US * CYCLES_PER_US))
    w_ms = time_sleep_ms(int(W_US * CYCLES_PER_US))
    print(f"rank {rank}: {CYCLES_PER_US:.0f} cycles/μs | "
          f"F sleep {f_ms:.3f} ms (target {FORWARD_US/1000:.3f}) | "
          f"B sleep {b_ms:.3f} ms (target {B_US/1000:.3f}) | "
          f"W sleep {w_ms:.3f} ms (target {W_US/1000:.3f})", flush=True)

    if rank == 0:
        print("=" * 60)
        print("Synthetic ZB-H1 Pipeline Kernel  (p=2, m=4)")
        print("=" * 60)
        print(f"  micro_batch_size  : {MICRO_BATCH}")
        print(f"  sequence_length   : {SEQ_LEN}")
        print(f"  hidden_size       : {HIDDEN}")
        print(f"  dtype             : {DTYPE}")
        print(f"  n_steps           : {N_STEPS}")
        print(f"  pp_stages (p)     : {PP_STAGES}")
        print(f"  microbatches (m)  : {N_MICROBATCHES}")
        print(f"  activation size   : {ACTIVATION_MB:.1f} MB per transfer")
        print(f"  forward_us        : {FORWARD_US:,} μs  ({FORWARD_US/1000:.1f} ms)")
        print(f"  b_us              : {B_US:,} μs  ({B_US/1000:.1f} ms)  input grad")
        print(f"  w_us              : {W_US:,} μs  ({W_US/1000:.1f} ms)  weight grad")
        print(f"  bubble_fraction   : {BUBBLE_FRACTION:.1%}  (p-1)(F+B-W) / m(F+B+W)")
        print(f"  transfers/step    : {2 * N_MICROBATCHES}  ({N_MICROBATCHES} mb × fwd + bwd) × {ACTIVATION_MB:.0f} MB")
        print("=" * 60, flush=True)

    bufs = [
        torch.zeros(ACTIVATION_SHAPE, dtype=DTYPE, device="cuda")
        for _ in range(N_MICROBATCHES)
    ]

    # warmup (not measured)
    print(f"rank {rank}: warming up", flush=True)
    for _ in range(3):
        iteration(rank, bufs)
    dist.barrier()
    torch.cuda.synchronize()
    print(f"rank {rank}: warmup done", flush=True)

    #measured run
    step_times = []

    for step in range(N_STEPS):
        dist.barrier()
        t0 = time.perf_counter()

        iteration(rank, bufs)

        t1 = time.perf_counter()
        step_times.append((t1 - t0) * 1000)  # ms

        if rank == 0 and (step + 1) % 5 == 0:
            print(f"  step {step+1:3d}/{N_STEPS} | iter time: {step_times[-1]:.1f} ms", flush=True)

    dist.barrier()


    #prints
    if rank == 0:
        avg     = sum(step_times) / len(step_times)
        min_t   = min(step_times)
        max_t   = max(step_times)
        total_s = sum(step_times) / 1000

        # ideal: m microbatches × (F + B + W), no idle, no comm
        ideal_ms  = N_MICROBATCHES * (FORWARD_US + B_US + W_US) / 1000
        bubble_ms = (PP_STAGES - 1) * (FORWARD_US + B_US - W_US) / 1000   # rank 0 waits B for G0, F-W for G3
        comm_ms   = avg - ideal_ms - bubble_ms
        bubble_ratio = (avg - ideal_ms) / avg

        print()
        print("=" * 60)
        print("Results — ZB-H1 (p=2, m=4)")
        print("=" * 60)
        print(f"  avg iter time     : {avg:.1f} ms")
        print(f"  min iter time     : {min_t:.1f} ms")
        print(f"  max iter time     : {max_t:.1f} ms")
        print(f"  total wall time   : {total_s:.1f} s  ({total_s/60:.1f} min)")
        print()
        print("Breakdown (rank 0):")
        print(f"  ideal compute     : {ideal_ms:.1f} ms  (m × F + B + W, no idle)")
        print(f"  bubble            : {bubble_ms:.1f} ms  = {bubble_ms/avg*100:.1f}% of wall time")
        print(f"  bubble formula    : (p-1)(F+B-W) = {bubble_ms:.1f} ms = {BUBBLE_FRACTION:.1%} of ideal")
        print(f"  bubble_ratio      : {bubble_ratio:.4f}")
        print(f"  communication     : {comm_ms:.1f} ms  ({2 * N_MICROBATCHES} × {ACTIVATION_MB:.0f} MB NCCL)")
        print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
