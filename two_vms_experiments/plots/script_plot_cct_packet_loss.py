import matplotlib.pyplot as plt

losses = []
times = []

with open('/root/two_vms_experiments/results/CCT_packet_loss.csv') as f:
    next(f)
    for line in f:
        line = line.strip()
        if not line:
            continue
        if ',' in line:
            parts = line.split(',')
            try:
                losses.append(float(parts[0]))
                times.append(float(parts[1]))
            except ValueError:
                pass

times_ms = [t / 1000 for t in times]

plt.figure(figsize=(8, 5))
plt.plot(losses, times_ms, marker='o', linewidth=2, markersize=8, color='steelblue')
plt.xlabel('Packet Loss (%)', fontsize=13)
plt.ylabel('Communication Completion Time (ms)', fontsize=13)
plt.title('AllReduce Communication Time vs Packet Loss\n(message size: 32MB, 2 nodes)', fontsize=13)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
path = '/root/two_vms_experiments/plots/packet_loss_vs_CCT.png'
plt.savefig(path, dpi=150)
print(f"Plot saved to {path}")
