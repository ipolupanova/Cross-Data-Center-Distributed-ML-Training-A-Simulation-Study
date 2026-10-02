"""
Synthetic interleaved 1F1B Pipeline Parallel Kernel
===================================================
nanotron 109M LLaMA, pp=2, tp=1, dp=1

Config:
  p = 2   pipeline stages (ranks)
  m = 4   microbatches
  v = 2   model chunks per rank


Layer split (4 virtual stages of 3 layers):
  R0/C0: embedding + layers 0-2      R0/C1: layers 6-8
  R1/C0: layers 3-5                  R1/C1: layers 9-11 + LM head

Every microbatch follows:
  forward : R0/C0 -> R1/C0 -> R0/C1 -> R1/C1
  backward: R1/C1 -> R0/C1 -> R1/C0 -> R0/C0
  = 3 activation + 3 gradient transfers per microbatch


Activation tensor: (16, 512, 768) bf16 = 12 MiB per transfer
(m=4 microbatches of 16 = 64 samples per iteration, as in the Nsight profile)


Compute times from Nsight (micro_batch=64), scaled linearly to micro_batch=16
and halved because each chunk holds half of a rank's layers:
  FORWARD_CHUNK_US  = 18,814 / 4 / 2 = 2,351.75 μs
  BACKWARD_CHUNK_US = 29,113 / 4 / 2 = 3,639.125 μs


Schedule (Megatron-LM interleaved 1F1B):
  microbatches run in groups of p per chunk:
    forward  order: 0c0 1c0 0c1 1c1 | 2c0 3c0 2c1 3c1
    backward order: 0c1 1c1 0c0 1c0 | 2c1 3c1 2c0 3c0
  warm-up forwards = (p-1-rank)*2 + (v-1)*p  ->  rank 0: 4, rank 1: 2
  then one forward + one backward per step, then drain the backwards.

  All transfers happen in 11 exchange rows X1..X11. Each row is one paired
  batch_isend_irecv on both ranks and the rows run in the same order on
  both ranks, so every send meets its recv and nothing deadlocks.

  Rank 0                                   Rank 1
  F0c0                                     .
  X1  send A0c0 --------------------------> recv
  F1c0                                     F0c0
  X2  send A1c0 / recv A0c0 <-------------> send A0c0 / recv A1c0
  F0c1                                     F1c0
  X3  send A0c1 / recv A1c0 <-------------> send A1c0 / recv A0c1
  F1c1                                     F0c1  B0c1
  X4  send A1c1 / recv G0c1 <-------------> send G0c1 / recv A1c1
  F2c0  B0c1                               F1c1  B1c1
  X5  send A2c0 G0c1 / recv G1c1 <--------> send G1c1 / recv A2c0 G0c1
  F3c0  B1c1                               F2c0  B0c0
  X6  send A3c0 G1c1 / recv A2c0 G0c0 <---> send A2c0 G0c0 / recv A3c0 G1c1
  F2c1  B0c0                               F3c0  B1c0
  X7  send A2c1 / recv A3c0 G1c0 <--------> send A3c0 G1c0 / recv A2c1
  F3c1  B1c0                               F2c1  B2c1
  X8  send A3c1 / recv G2c1 <-------------> send G2c1 / recv A3c1
  B2c1                                     F3c1  B3c1
  X9  send G2c1 / recv G3c1 <-------------> send G3c1 / recv G2c1
  B3c1                                     B2c0
  X10 send G3c1 / recv G2c0 <-------------> send G2c0 / recv G3c1
  B2c0                                     B3c0
  X11 recv G3c0 <-------------------------- send G3c0
  B3c0

  (A = activation, G = gradient, named after the chunk that produced it)

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
N_CHUNKS       = 2    # v

# bubble fraction (p-1)/(m*v)
BUBBLE_FRACTION = (PP_STAGES - 1) / (N_MICROBATCHES * N_CHUNKS)

FORWARD_US  = 18_814 * MICRO_BATCH / 64    # per rank per microbatch (all layers of the rank), Nsight value at micro_batch=64 scaled to MICRO_BATCH
BACKWARD_US = 29_113 * MICRO_BATCH / 64
FORWARD_CHUNK_US  = FORWARD_US / N_CHUNKS
BACKWARD_CHUNK_US = BACKWARD_US / N_CHUNKS

ACTIVATION_SHAPE = (MICRO_BATCH, SEQ_LEN, HIDDEN)
ACTIVATION_MB    = MICRO_BATCH * SEQ_LEN * HIDDEN * 2 / 1024**2  # 12 MiB

# 3 activations + 3 gradients per microbatch
N_TRANSFERS = N_MICROBATCHES * 2 * (PP_STAGES * N_CHUNKS - 1)


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


def F():
    fake_compute(FORWARD_CHUNK_US)


def B():
    fake_compute(BACKWARD_CHUNK_US)


def exchange(peer: int, send=(), recv=()):
    """One exchange row: all sends and recvs issued together (paired P2P), waits for all."""
    ops  = [dist.P2POp(dist.isend, t, peer) for t in send]
    ops += [dist.P2POp(dist.irecv, t, peer) for t in recv]
    for r in dist.batch_isend_irecv(ops):
        r.wait()


#iteration according to interleaved 1f1b schedule with 2 ranks, 4 microbatches, 2 chunks per rank
# no row sends or receives two tensors of the same kind, so one buffer per kind and direction is enough
def iteration(rank: int, bufs: dict):
    a_send, g_send = bufs["act_send"], bufs["grad_send"]
    a_recv, g_recv = bufs["act_recv"], bufs["grad_recv"]

    if rank == 0:
        # warm-up: 4 forwards
        F()                                                        # F0c0
        exchange(1, send=[a_send])                                 # X1  A0c0 ->
        F()                                                        # F1c0
        exchange(1, send=[a_send], recv=[a_recv])                  # X2  A1c0 ->   <- A0c0
        F()                                                        # F0c1
        exchange(1, send=[a_send], recv=[a_recv])                  # X3  A0c1 ->   <- A1c0
        F()                                                        # F1c1
        exchange(1, send=[a_send], recv=[g_recv])                  # X4  A1c1 ->   <- G0c1

        # steady state: 1 forward + 1 backward per step
        F(); B()                                                   # F2c0 B0c1
        exchange(1, send=[a_send, g_send], recv=[g_recv])          # X5  A2c0 G0c1 ->   <- G1c1
        F(); B()                                                   # F3c0 B1c1
        exchange(1, send=[a_send, g_send], recv=[a_recv, g_recv])  # X6  A3c0 G1c1 ->   <- A2c0 G0c0
        F(); B()                                                   # F2c1 B0c0
        exchange(1, send=[a_send], recv=[a_recv, g_recv])          # X7  A2c1 ->   <- A3c0 G1c0
        F(); B()                                                   # F3c1 B1c0
        exchange(1, send=[a_send], recv=[g_recv])                  # X8  A3c1 ->   <- G2c1

        # cooldown: remaining 4 backwards
        B()                                                        # B2c1
        exchange(1, send=[g_send], recv=[g_recv])                  # X9  G2c1 ->   <- G3c1
        B()                                                        # B3c1
        exchange(1, send=[g_send], recv=[g_recv])                  # X10 G3c1 ->   <- G2c0
        B()                                                        # B2c0
        exchange(1, recv=[g_recv])                                 # X11           <- G3c0
        B()                                                        # B3c0

    else:  # rank == 1
        # warm-up: 2 forwards
        exchange(0, recv=[a_recv])                                 # X1            <- A0c0
        F()                                                        # F0c0
        exchange(0, send=[a_send], recv=[a_recv])                  # X2  A0c0 ->   <- A1c0
        F()                                                        # F1c0
        exchange(0, send=[a_send], recv=[a_recv])                  # X3  A1c0 ->   <- A0c1

        # steady state: 1 forward + 1 backward per step
        F(); B()                                                   # F0c1 B0c1
        exchange(0, send=[g_send], recv=[a_recv])                  # X4  G0c1 ->   <- A1c1
        F(); B()                                                   # F1c1 B1c1
        exchange(0, send=[g_send], recv=[a_recv, g_recv])          # X5  G1c1 ->   <- A2c0 G0c1
        F(); B()                                                   # F2c0 B0c0
        exchange(0, send=[a_send, g_send], recv=[a_recv, g_recv])  # X6  A2c0 G0c0 ->   <- A3c0 G1c1
        F(); B()                                                   # F3c0 B1c0
        exchange(0, send=[a_send, g_send], recv=[a_recv])          # X7  A3c0 G1c0 ->   <- A2c1
        F(); B()                                                   # F2c1 B2c1
        exchange(0, send=[g_send], recv=[a_recv])                  # X8  G2c1 ->   <- A3c1
        F(); B()                                                   # F3c1 B3c1
        exchange(0, send=[g_send], recv=[g_recv])                  # X9  G3c1 ->   <- G2c1

        # cooldown: remaining 2 backwards
        B()                                                        # B2c0
        exchange(0, send=[g_send], recv=[g_recv])                  # X10 G2c0 ->   <- G3c1
        B()                                                        # B3c0
        exchange(0, send=[g_send])                                 # X11 G3c0 ->

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
    fwd_ms = time_sleep_ms(int(FORWARD_CHUNK_US * CYCLES_PER_US))
    bwd_ms = time_sleep_ms(int(BACKWARD_CHUNK_US * CYCLES_PER_US))
    print(f"rank {rank}: {CYCLES_PER_US:.0f} cycles/μs | "
          f"fwd chunk sleep {fwd_ms:.3f} ms (target {FORWARD_CHUNK_US/1000:.3f}) | "
          f"bwd chunk sleep {bwd_ms:.3f} ms (target {BACKWARD_CHUNK_US/1000:.3f})", flush=True)

    if rank == 0:
        print("=" * 60)
        print("Synthetic interleaved 1F1B Pipeline Kernel  (p=2, m=4, v=2)")
        print("=" * 60)
        print(f"  micro_batch_size  : {MICRO_BATCH}")
        print(f"  sequence_length   : {SEQ_LEN}")
        print(f"  hidden_size       : {HIDDEN}")
        print(f"  dtype             : {DTYPE}")
        print(f"  n_steps           : {N_STEPS}")
        print(f"  pp_stages (p)     : {PP_STAGES}")
        print(f"  microbatches (m)  : {N_MICROBATCHES}")
        print(f"  model chunks (v)  : {N_CHUNKS}")
        print(f"  activation size   : {ACTIVATION_MB:.1f} MB per transfer")
        print(f"  forward_chunk_us  : {FORWARD_CHUNK_US:,.1f} μs  ({FORWARD_CHUNK_US/1000:.1f} ms)")
        print(f"  backward_chunk_us : {BACKWARD_CHUNK_US:,.1f} μs  ({BACKWARD_CHUNK_US/1000:.1f} ms)")
        print(f"  bubble_fraction   : {BUBBLE_FRACTION:.1%}  (p-1)/(m*v) = ({PP_STAGES}-1)/({N_MICROBATCHES}*{N_CHUNKS})")
        print(f"  transfers/step    : {N_TRANSFERS}  ({N_MICROBATCHES} mb × 3 fwd + 3 bwd) × {ACTIVATION_MB:.0f} MB")
        print("=" * 60, flush=True)

    bufs = {
        name: torch.zeros(ACTIVATION_SHAPE, dtype=DTYPE, device="cuda")
        for name in ("act_send", "grad_send", "act_recv", "grad_recv")
    }

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

        # ideal: m microbatches × v chunks × (fwd + bwd), no idle, no comm
        ideal_ms  = N_MICROBATCHES * N_CHUNKS * (FORWARD_CHUNK_US + BACKWARD_CHUNK_US) / 1000
        bubble_ms = BUBBLE_FRACTION * ideal_ms   # rank 0 idle waiting for G0c1 (X4) and G3c1 (X9)
        comm_ms   = avg - ideal_ms - bubble_ms
        bubble_ratio = (avg - ideal_ms) / avg

        print()
        print("=" * 60)
        print("Results — interleaved 1F1B (p=2, m=4, v=2)")
        print("=" * 60)
        print(f"  avg iter time     : {avg:.1f} ms")
        print(f"  min iter time     : {min_t:.1f} ms")
        print(f"  max iter time     : {max_t:.1f} ms")
        print(f"  total wall time   : {total_s:.1f} s  ({total_s/60:.1f} min)")
        print()
        print("Breakdown (rank 0):")
        print(f"  ideal compute     : {ideal_ms:.1f} ms  (m × v × fwd + bwd, no idle)")
        print(f"  bubble            : {bubble_ms:.1f} ms  = {bubble_ms/avg*100:.1f}% of wall time")
        print(f"  bubble formula    : (p-1)/(m*v) = ({PP_STAGES}-1)/({N_MICROBATCHES}*{N_CHUNKS}) = {BUBBLE_FRACTION:.1%}")
        print(f"  bubble_ratio      : {bubble_ratio:.4f}")
        print(f"  communication     : {comm_ms:.1f} ms  ({N_TRANSFERS} × {ACTIVATION_MB:.0f} MB NCCL)")
        print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
