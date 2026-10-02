import matplotlib.pyplot as plt
import os

BASE = '/root/two_vms_experiments/results'

def load_csv(filename):
    path = os.path.join(BASE, filename)
    xs, ys = [], []
    try:
        with open(path) as f:
            next(f)
            for line in f:
                line = line.strip()
                if not line or ',' not in line:
                    continue
                parts = line.split(',')
                try:
                    xs.append(float(parts[0]))
                    ys.append(float(parts[1]) / 1000)  # us -> ms
                except ValueError:
                    pass
    except FileNotFoundError:
        return None, None
    if not xs:
        return None, None
    return xs, ys

# (operation, label, color, csvs_by_size)
OPERATIONS = [
    ('AllReduce', {
        '32MB':  {'latency': 'results_cct_latency.csv', 'jitter': 'CCT_jitter.csv',        'packet_loss': 'CCT_packet_loss.csv',        'bandwidth': 'CCT_bandwidth_clouds.csv'},
        '64MB':  {'latency': 'CCT_latency_64M.csv',     'jitter': 'CCT_jitter_64M.csv',    'packet_loss': 'CCT_packet_loss_64M.csv',    'bandwidth': 'CCT_bandwidth_64M.csv'},
        '128MB': {'latency': 'CCT_latency_128M.csv',    'jitter': 'CCT_jitter_128M.csv',   'packet_loss': 'CCT_packet_loss_128M.csv',   'bandwidth': 'CCT_bandwidth_128M.csv'},
    }),
    ('SendRecv', {
        '32MB':  {'latency': 'CCT_sendrecv_latency.csv',      'jitter': 'CCT_sendrecv_jitter.csv',      'packet_loss': 'CCT_sendrecv_packet_loss.csv',      'bandwidth': 'CCT_sendrecv_bandwidth.csv'},
        '64MB':  {'latency': 'CCT_sendrecv_latency_64M.csv',  'jitter': 'CCT_sendrecv_jitter_64M.csv',  'packet_loss': 'CCT_sendrecv_packet_loss_64M.csv',  'bandwidth': 'CCT_sendrecv_bandwidth_64M.csv'},
        '128MB': {'latency': 'CCT_sendrecv_latency_128M.csv', 'jitter': 'CCT_sendrecv_jitter_128M.csv', 'packet_loss': 'CCT_sendrecv_packet_loss_128M.csv', 'bandwidth': 'CCT_sendrecv_bandwidth_128M.csv'},
    }),
]

# (color, marker, label, invert_x)
# bandwidth is inverted: higher bandwidth = less impairment
FACTOR_STYLES = {
    'latency':     ('steelblue',  'o', 'Network Latency', False),
    'jitter':      ('darkorange', 's', 'Jitter',          False),
    'packet_loss': ('seagreen',   '^', 'Packet Loss',     False),
    'bandwidth':   ('crimson',    'D', 'Bandwidth',       True),
}

for op_name, csvs_by_size in OPERATIONS:
    for size, csvs in csvs_by_size.items():
        fig, ax = plt.subplots(figsize=(8, 5))
        fig.suptitle(f'{op_name} — CCT vs Normalized Network Impairment ({size}, 2 nodes)',
                     fontsize=13, fontweight='bold')

        any_plotted = False
        for factor, (color, marker, label, invert) in FACTOR_STYLES.items():
            csv_name = csvs.get(factor)
            if not csv_name:
                continue
            xs, ys = load_csv(csv_name)
            if not xs:
                continue
            x_min, x_max = min(xs), max(xs)
            if x_max == x_min:
                continue
            if invert:
                xs_norm = [(x_max - x) / (x_max - x_min) for x in xs]
                # sort by normalized x so line goes left to right
                pairs = sorted(zip(xs_norm, ys))
                xs_norm, ys = zip(*pairs)
            else:
                xs_norm = [(x - x_min) / (x_max - x_min) for x in xs]
            ax.plot(xs_norm, ys, marker=marker, linewidth=2, markersize=7,
                    color=color, label=label)
            any_plotted = True

        ax.set_xlabel('Normalized Impairment (0 = none, 1 = max tested)', fontsize=12)
        ax.set_ylabel('CCT (ms)', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.6)
        if any_plotted:
            ax.legend(fontsize=11)

        plt.tight_layout()
        out = f'/root/two_vms_experiments/plots/summary_{op_name.lower()}_{size}.png'
        plt.savefig(out, dpi=150)
        plt.close()
        print(f'Saved {out}')
