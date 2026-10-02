"""Figures for the two-VM NCCL CCT report (reads ../results/*.csv, writes fig_*.pdf here)."""
import csv
import os

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results")

# palette: categorical slots 1-3 (validated all-pairs, light), chrome inks
SIZE_COLOR = {32: "#2a78d6", 64: "#eb6834", 128: "#1baf7a"}
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
OP_STYLE = {"AllReduce": dict(ls="-", marker="o"), "SendRecv": dict(ls="--", marker="s")}
SIZES = (32, 64, 128)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7.5, "axes.labelsize": 7.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.8,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 1.4, "lines.markersize": 4,
    "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})

FILES = {  # (op, factor, size) -> csv
    ("AllReduce", "latency", 32): "results_cct_latency.csv",
    ("AllReduce", "jitter", 32): "CCT_jitter.csv",
    ("AllReduce", "packet_loss", 32): "CCT_packet_loss.csv",
    ("AllReduce", "bandwidth", 32): "CCT_bandwidth_clouds.csv",
}
for s in (64, 128):
    for f in ("latency", "jitter", "packet_loss", "bandwidth"):
        FILES[("AllReduce", f, s)] = f"CCT_{f}_{s}M.csv"
for s in SIZES:
    for f in ("latency", "jitter", "packet_loss", "bandwidth"):
        FILES[("SendRecv", f, s)] = f"CCT_sendrecv_{f}" + ("" if s == 32 else f"_{s}M") + ".csv"


def load(op, factor, size):
    """-> (x list, CCT in ms list); rows without a time are skipped."""
    xs, ys = [], []
    with open(os.path.join(RESULTS, FILES[(op, factor, size)])) as fh:
        for row in csv.reader(fh):
            try:
                x, y = float(row[0]), float(row[1]) / 1000
            except (ValueError, IndexError):
                continue
            xs.append(x); ys.append(y)
    return xs, ys


# baseline = unimpaired CCT of the same op and size (0 % point of the packet-loss sweep)
BASE = {(op, s): load(op, "packet_loss", s)[1][0] for op in OP_STYLE for s in SIZES}


def plot_series(ax, op, factor, size, normalize=True, **kw):
    xs, ys = load(op, factor, size)
    if normalize:
        ys = [y / BASE[(op, size)] for y in ys]
    ax.plot(xs, ys, color=SIZE_COLOR[size], markerfacecolor="white", markeredgewidth=1.1,
            **OP_STYLE[op], **kw)
    return xs, ys


def legend(ax, loc, ops=("AllReduce", "SendRecv")):
    h = [Line2D([], [], color=SIZE_COLOR[s], lw=1.4, label=f"{s} MiB") for s in SIZES]
    h += [Line2D([], [], color=INK2, markerfacecolor="white", lw=1.1, label=op, **OP_STYLE[op]) for op in ops]
    ax.legend(handles=h, loc=loc, frameon=False, ncol=1, handlelength=2.2, labelcolor=INK2)


def save(fig, name):
    fig.savefig(os.path.join(HERE, name))
    fig.savefig(os.path.join(HERE, "preview_" + name.replace(".pdf", ".png")), dpi=170)
    plt.close(fig)
    print("wrote", name)


# ---- Figure 1: latency ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(3.3, 2.35))
for op in OP_STYLE:
    for s in SIZES:
        plot_series(ax, op, "latency", s)
ax.set_yscale("log")
ax.set_yticks([1, 2, 5, 10, 20]); ax.set_yticklabels(["1×", "2×", "5×", "10×", "20×"])
ax.set_xticks([0, 25, 50, 100, 200])
ax.set_xlabel("Added one-way delay (ms)")
ax.set_ylabel("CCT / unimpaired CCT")
ax.annotate("AllReduce: 15–20×\nat 200 ms", xy=(197, 15), xytext=(128, 1.12), fontsize=6.6, color=INK2,
            arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
legend(ax, "upper left")
save(fig, "fig_latency.pdf")

# ---- Figure 2: jitter and packet loss -----------------------------------------------
fig, (a, b) = plt.subplots(1, 2, figsize=(6.9, 2.2), gridspec_kw=dict(wspace=0.22))
for op in OP_STYLE:
    for s in SIZES:
        plot_series(a, op, "jitter", s)
        plot_series(b, op, "packet_loss", s)
a.set_yscale("log")
a.set_yticks([1, 1.5, 2, 5, 10, 20]); a.set_yticklabels(["1×", "1.5×", "2×", "5×", "10×", "20×"])
a.minorticks_off()
a.set_xlabel("Jitter on a 10 ms base delay (ms)")
a.set_ylabel("CCT / unimpaired CCT")
a.set_title("(a) Jitter", loc="left", fontsize=7.5, color=INK)
a.annotate("AllReduce ≈20× at 6 ms\n(SendRecv not measured)", xy=(5.95, 18), xytext=(1.6, 7), fontsize=6.6,
           color=INK2, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
b.set_xlabel("Random packet loss (%)")
b.set_title("(b) Packet loss", loc="left", fontsize=7.5, color=INK)
b.set_yticks([1.0, 1.1, 1.2, 1.3, 1.4, 1.5]); b.set_yticklabels(["1.0×", "1.1×", "1.2×", "1.3×", "1.4×", "1.5×"])
legend(b, "upper left")
save(fig, "fig_jitter_loss.pdf")

# ---- Figure 3: bandwidth cap ------------------------------------------------------
fig, ax = plt.subplots(figsize=(3.3, 2.35))
caps = [100, 200, 400, 600, 800, 1000]
for s in SIZES:
    mbit = s * 8 * 1.048576  # MiB -> Mbit
    ax.plot(caps, [mbit / c for c in caps], color=SIZE_COLOR[s], lw=0.8, ls=":", alpha=0.9)
for op in OP_STYLE:
    for s in SIZES:
        xs, ys = load(op, "bandwidth", s)
        ax.plot(xs, [y / 1000 for y in ys], color=SIZE_COLOR[s], markerfacecolor="white",
                markeredgewidth=1.1, **OP_STYLE[op])
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xticks([100, 200, 400, 1000]); ax.set_xticklabels(["100", "200", "400", "1000"]); ax.minorticks_off()
ax.set_yticks([0.3, 1, 3, 10]); ax.set_yticklabels(["0.3", "1", "3", "10"])
ax.set_xlabel("Bandwidth cap (Mbit/s)")
ax.set_ylabel("CCT (s)")
h = [Line2D([], [], color=SIZE_COLOR[s], lw=1.4, label=f"{s} MiB") for s in SIZES]
h += [Line2D([], [], color=INK2, markerfacecolor="white", lw=1.1, label=op, **OP_STYLE[op]) for op in OP_STYLE]
h += [Line2D([], [], color=INK2, lw=0.8, ls=":", label="ideal S / cap")]
ax.legend(handles=h, loc="lower left", frameon=False, handlelength=2.2, labelcolor=INK2, ncol=2,
          columnspacing=1.0, borderaxespad=0.2)
ax.set_ylim(0.12, 14)
save(fig, "fig_bandwidth.pdf")

# numbers quoted in the text
for op in OP_STYLE:
    for s in SIZES:
        lat = load(op, "latency", s)[1]; loss = load(op, "packet_loss", s)[1]
        print(f"{op:9s} {s:3d} MiB base {BASE[(op, s)]:7.1f} ms | latency x: "
              + " ".join(f"{y / BASE[(op, s)]:.2f}" for y in lat)
              + f" | loss 1%: {loss[-1] / BASE[(op, s)]:.2f}")
