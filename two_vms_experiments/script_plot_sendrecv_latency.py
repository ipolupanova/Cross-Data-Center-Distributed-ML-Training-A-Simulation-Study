import matplotlib.pyplot as plt

latencies = []
times = []

with open('/root/two_vms_experiments/results/CCT_sendrecv_latency.csv') as f:
    next(f)
    for line in f:
        line = line.strip()
        if not line or ',' not in line:
            continue
        parts = line.split(',')
        try:
            latencies.append(float(parts[0]))
            times.append(float(parts[1]))
        except ValueError:
            pass

times_ms = [t / 1000 for t in times]

plt.figure(figsize=(8, 5))
plt.plot(latencies, times_ms, marker='o', linewidth=2, markersize=8, color='steelblue')
plt.xlabel('Injected Latency (ms)', fontsize=13)
plt.ylabel('Communication Completion Time (ms)', fontsize=13)
plt.title('SendRecv Communication Time vs Network Latency\n(message size: 32MB, 2 nodes)', fontsize=13)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
path = '/root/two_vms_experiments/plots/sendrecv_latency_vs_CCT.png'
plt.savefig(path, dpi=150)
print(f"Plot saved to {path}")
