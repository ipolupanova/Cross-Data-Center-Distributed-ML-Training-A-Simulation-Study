import matplotlib.pyplot as plt
import csv

def read_csv(path):
    xs, ys = [], []
    with open(path) as f:
        next(f)
        for line in f:
            line = line.strip()
            if not line or ',' not in line:
                continue
            parts = line.split(',')
            try:
                xs.append(float(parts[0]))
                ys.append(float(parts[1]))
            except ValueError:
                pass
    return xs, ys

latencies, lat_times = read_csv('/root/two_vms_experiments/results/results_cct_latency.csv')
bandwidths, bw_times = read_csv('/root/two_vms_experiments/results/CCT_bandwidth_clouds.csv')
losses, loss_times = read_csv('/root/two_vms_experiments/results/CCT_packet_loss.csv')

lat_baseline = lat_times[0]
bw_baseline = bw_times[-1]
loss_baseline = loss_times[0]

lat_x_norm = [l / max(latencies) for l in latencies]
bw_x_norm = [(max(bandwidths) - b) / (max(bandwidths) - min(bandwidths)) for b in bandwidths]
loss_x_norm = [l / max(losses) for l in losses]

lat_slowdown = [t / lat_baseline for t in lat_times]
bw_slowdown = [t / bw_baseline for t in bw_times]
loss_slowdown = [t / loss_baseline for t in loss_times]

plt.figure(figsize=(9, 5))
plt.plot(lat_x_norm, lat_slowdown, marker='o', linewidth=2, markersize=7, label='Latency (0–200ms)')
plt.plot(bw_x_norm, bw_slowdown, marker='s', linewidth=2, markersize=7, label='Bandwidth (10→1 Gbps)')
plt.plot(loss_x_norm, loss_slowdown, marker='^', linewidth=2, markersize=7, label='Packet Loss (0–1%)')
plt.axhline(1.0, color='gray', linestyle='--', linewidth=1)
plt.xlabel('Normalized Degradation Severity (0 = baseline, 1 = worst tested)', fontsize=12)
plt.ylabel('Slowdown Factor (CCT / baseline CCT)', fontsize=12)
plt.title('AllReduce CCT Slowdown by Network Factor\n(message size: 32MB, 2 nodes)', fontsize=13)
plt.legend(fontsize=11)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
path = '/root/two_vms_experiments/plots/normalized_comparison.png'
plt.savefig(path, dpi=150)
print(f"Plot saved to {path}")
