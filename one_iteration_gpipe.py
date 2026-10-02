"""
Synthetic GPipe Pipeline Parallel Kernel
========================================
nanotron 109M LLaMA, pp=2, tp=1, dp=1

Config:
  p = 2   pipeline stages
  m = 4   microbatches


Layer split:
  GPU 0: embedding + layers 0-5
  GPU 1: layers 6-11 + LM head


Activation tensor: (16, 512, 768) bf16 = 12 MiB per transfer
(m=4 microbatches of 16 = 64 samples per iteration, as in the Nsight profile)


Compute times from Nsight (micro_batch=64), scaled linearly to micro_batch=16:
  FORWARD_US  = 18,814 / 4 = 4,703.5 μs per microbatch
  BACKWARD_US = 29,113 / 4 = 7,278.25 μs per microbatch


Schedule (p=2, m=4): all forwards first, then all backwards in reverse order

  Rank 0                                  Rank 1

  ----- forward phase -----
  F0
  send A0 ------------------------------> recv A0
  F1                                      F0
  send A1 ------------------------------> recv A1
  F2                                      F1
  send A2 ------------------------------> recv A2
  F3                                      F2
  send A3 ------------------------------> recv A3
                                          F3
  ----- backward phase (3, 2, 1, 0) -----
                                          B3
  recv G3 <------------------------------ send G3
  B3                                      B2
  recv G2 <------------------------------ send G2
  B2                                      B1
  recv G1 <------------------------------ send G1
  B1                                      B0
  recv G0 <------------------------------ send G0
  B0

Never both ranks send at the same time, so plain blocking send/recv is enough.

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

# bubble fraction (p-1)/m
BUBBLE_FRACTION = (PP_STAGES - 1) / N_MICROBATCHES

FORWARD_US  = 18_814 * MICRO_BATCH / 64  # Nsight value at micro_batch=64, scaled to MICRO_BATCH
BACKWARD_US = 29_113 * MICRO_BATCH / 64

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


#iteration according to gpipe schedule with 2 ranks, 4 microbatches
# bufs[i] holds activation A(i) on the way forward and gradient G(i) on the way back
def iteration(rank: int, bufs: list):
    m = N_MICROBATCHES

    if rank == 0:
        # forward phase: F(i), send A(i)
        for i in range(m):
            fake_compute(FORWARD_US)
            dist.send(bufs[i], dst=1)

        # backward phase in reverse order: recv G(i), B(i)
        for i in reversed(range(m)):
            dist.recv(bufs[i], src=1)
            fake_compute(BACKWARD_US)

    else:  # rank == 1
        # forward phase: recv A(i), F(i)
        for i in range(m):
            dist.recv(bufs[i], src=0)
            fake_compute(FORWARD_US)

        # backward phase in reverse order: B(i), send G(i)
        for i in reversed(range(m)):
            fake_compute(BACKWARD_US)
            dist.send(bufs[i], dst=0)

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
    fwd_ms = time_sleep_ms(int(FORWARD_US * CYCLES_PER_US))
    bwd_ms = time_sleep_ms(int(BACKWARD_US * CYCLES_PER_US))
    print(f"rank {rank}: {CYCLES_PER_US:.0f} cycles/μs | "
          f"fwd sleep {fwd_ms:.3f} ms (target {FORWARD_US/1000:.3f}) | "
          f"bwd sleep {bwd_ms:.3f} ms (target {BACKWARD_US/1000:.3f})", flush=True)

    if rank == 0:
        print("=" * 60)
        print("Synthetic GPipe Pipeline Kernel  (p=2, m=4)")
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
        print(f"  backward_us       : {BACKWARD_US:,} μs  ({BACKWARD_US/1000:.1f} ms)")
        print(f"  bubble_fraction   : {BUBBLE_FRACTION:.1%}  (p-1)/m = ({PP_STAGES}-1)/{N_MICROBATCHES}")
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

        # ideal: m microbatches × (fwd + bwd), no idle, no comm
        ideal_ms  = N_MICROBATCHES * (FORWARD_US + BACKWARD_US) / 1000
        bubble_ms = (PP_STAGES - 1) * (FORWARD_US + BACKWARD_US) / 1000   # rank 0 idle during F3 + B3 on rank 1
        comm_ms   = avg - ideal_ms - bubble_ms
        bubble_ratio = (avg - ideal_ms) / avg

        print()
        print("=" * 60)
        print("Results — GPipe (p=2, m=4)")
        print("=" * 60)
        print(f"  avg iter time     : {avg:.1f} ms")
        print(f"  min iter time     : {min_t:.1f} ms")
        print(f"  max iter time     : {max_t:.1f} ms")
        print(f"  total wall time   : {total_s:.1f} s  ({total_s/60:.1f} min)")
        print()
        print("Breakdown (rank 0):")
        print(f"  ideal compute     : {ideal_ms:.1f} ms  (m × fwd + bwd, no idle)")
        print(f"  bubble            : {bubble_ms:.1f} ms  = {bubble_ms/avg*100:.1f}% of wall time")
        print(f"  bubble formula    : (p-1)/m = ({PP_STAGES}-1)/{N_MICROBATCHES} = {BUBBLE_FRACTION:.1%}")
        print(f"  bubble_ratio      : {bubble_ratio:.4f}")
        print(f"  communication     : {comm_ms:.1f} ms  ({2 * N_MICROBATCHES} × {ACTIVATION_MB:.0f} MB NCCL)")
        print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
