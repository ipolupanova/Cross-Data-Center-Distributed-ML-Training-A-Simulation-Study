import csv
import matplotlib.pyplot as plt

bandwidths = []
times = []

with open('/root/two_vms_experiments/results/CCT_bandwidth_clouds.csv') as f:
    next(f)  # skip header
    for line in f:
        line = line.strip()
        if not line:
            continue
        if ',' in line:
            parts = line.split(',')
            try:
                bdw = int(parts[0])
                t = float(parts[1])
                bandwidths.append(bdw)
                times.append(t)
            except ValueError:
                pass

times_ms = [t / 1000 for t in times]

plt.figure(figsize=(8, 5))
plt.plot(bandwidths, times_ms, marker='o', linewidth=2, markersize=8, color='steelblue')
plt.xlabel('Available Bandwidth (Mbps)', fontsize=13)
plt.ylabel('Communication Completion Time (ms)', fontsize=13)
plt.title('AllReduce Communication Time vs Available Bandwidth (Cloud)\n(message size: 32MB, 2 nodes)', fontsize=13)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
path = '/root/two_vms_experiments/plots/bandwidth_clouds_vs_CCT.png'
plt.savefig(path, dpi=150)
print(f"Plot saved to {path}")
