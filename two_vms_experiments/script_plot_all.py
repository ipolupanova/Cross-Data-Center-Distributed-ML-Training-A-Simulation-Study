import matplotlib.pyplot as plt
import os

BASE = '/root/two_vms_experiments'

def load_csv(path):
    """Load CSV, handling both clean 'x,y' lines and legacy two-line format."""
    xs, ys = [], []
    try:
        with open(path) as f:
            next(f)
            pending_x = None
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if ',' in line:
                    parts = line.split(',')
                    try:
                        x = float(parts[0])
                        y = float(parts[1])
                        xs.append(x)
                        ys.append(y)
                        pending_x = None
                    except ValueError:
                        # second column is not numeric (e.g. old 'x,minBytes' format)
                        try:
                            pending_x = float(parts[0])
                        except ValueError:
                            pending_x = None
                else:
                    if pending_x is not None:
                        try:
                            ys.append(float(line))
                            xs.append(pending_x)
                            pending_x = None
                        except ValueError:
                            pass
    except FileNotFoundError:
        return None, None
    return xs, ys

def make_plot(xs, ys, xlabel, title, outpath):
    ys_ms = [y / 1000 for y in ys]
    plt.figure(figsize=(8, 5))
    plt.plot(xs, ys_ms, marker='o', linewidth=2, markersize=8, color='steelblue')
    plt.xlabel(xlabel, fontsize=13)
    plt.ylabel('Communication Completion Time (ms)', fontsize=13)
    plt.title(title, fontsize=13)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()
    print(f'Saved {outpath}')

# (csv_stem, xlabel, operation, message_size, plot_stem)
CONFIGS = [
    # AllReduce 32M
    ('results_cct_latency',   'Injected Latency (ms)', 'AllReduce', '32MB', 'latency_vs_CCT'),
    ('CCT_jitter',            'Injected Jitter (ms)',  'AllReduce', '32MB', 'jitter_vs_CCT'),
    ('CCT_packet_loss',       'Packet Loss (%)',        'AllReduce', '32MB', 'packet_loss_vs_CCT'),
    ('CCT_bandwidth_clouds',  'Bandwidth (Mbps)',       'AllReduce', '32MB', 'bandwidth_vs_CCT'),
    # AllReduce 64M
    ('CCT_latency_64M',       'Injected Latency (ms)', 'AllReduce', '64MB', 'latency_64M_vs_CCT'),
    ('CCT_jitter_64M',        'Injected Jitter (ms)',  'AllReduce', '64MB', 'jitter_64M_vs_CCT'),
    ('CCT_packet_loss_64M',   'Packet Loss (%)',        'AllReduce', '64MB', 'packet_loss_64M_vs_CCT'),
    ('CCT_bandwidth_64M',     'Bandwidth (Mbps)',       'AllReduce', '64MB', 'bandwidth_64M_vs_CCT'),
    # AllReduce 128M
    ('CCT_latency_128M',      'Injected Latency (ms)', 'AllReduce', '128MB', 'latency_128M_vs_CCT'),
    ('CCT_jitter_128M',       'Injected Jitter (ms)',  'AllReduce', '128MB', 'jitter_128M_vs_CCT'),
    ('CCT_packet_loss_128M',  'Packet Loss (%)',        'AllReduce', '128MB', 'packet_loss_128M_vs_CCT'),
    ('CCT_bandwidth_128M',    'Bandwidth (Mbps)',       'AllReduce', '128MB', 'bandwidth_128M_vs_CCT'),
    # SendRecv 32M
    ('CCT_sendrecv_latency',        'Injected Latency (ms)', 'SendRecv', '32MB', 'sendrecv_latency_vs_CCT'),
    ('CCT_sendrecv_jitter',         'Injected Jitter (ms)',  'SendRecv', '32MB', 'sendrecv_jitter_vs_CCT'),
    ('CCT_sendrecv_packet_loss',    'Packet Loss (%)',        'SendRecv', '32MB', 'sendrecv_packet_loss_vs_CCT'),
    ('CCT_sendrecv_bandwidth',      'Bandwidth (Mbps)',       'SendRecv', '32MB', 'sendrecv_bandwidth_vs_CCT'),
    # SendRecv 64M
    ('CCT_sendrecv_latency_64M',    'Injected Latency (ms)', 'SendRecv', '64MB', 'sendrecv_latency_64M_vs_CCT'),
    ('CCT_sendrecv_jitter_64M',     'Injected Jitter (ms)',  'SendRecv', '64MB', 'sendrecv_jitter_64M_vs_CCT'),
    ('CCT_sendrecv_packet_loss_64M','Packet Loss (%)',        'SendRecv', '64MB', 'sendrecv_packet_loss_64M_vs_CCT'),
    ('CCT_sendrecv_bandwidth_64M',  'Bandwidth (Mbps)',       'SendRecv', '64MB', 'sendrecv_bandwidth_64M_vs_CCT'),
    # SendRecv 128M
    ('CCT_sendrecv_latency_128M',   'Injected Latency (ms)', 'SendRecv', '128MB', 'sendrecv_latency_128M_vs_CCT'),
    ('CCT_sendrecv_jitter_128M',    'Injected Jitter (ms)',  'SendRecv', '128MB', 'sendrecv_jitter_128M_vs_CCT'),
    ('CCT_sendrecv_packet_loss_128M','Packet Loss (%)',       'SendRecv', '128MB', 'sendrecv_packet_loss_128M_vs_CCT'),
    ('CCT_sendrecv_bandwidth_128M', 'Bandwidth (Mbps)',       'SendRecv', '128MB', 'sendrecv_bandwidth_128M_vs_CCT'),
]

skipped = 0
for csv_stem, xlabel, op, msgsize, plot_stem in CONFIGS:
    csv_path = os.path.join(BASE, 'results', csv_stem + '.csv')
    xs, ys = load_csv(csv_path)
    if not xs:
        print(f'Skipping {csv_stem} (no data)')
        skipped += 1
        continue
    factor = xlabel.split('(')[0].strip()
    title = f'{op} Communication Time vs {factor}\n(message size: {msgsize}, 2 nodes)'
    outpath = os.path.join(BASE, 'plots', plot_stem + '.png')
    make_plot(xs, ys, xlabel, title, outpath)

if skipped:
    print(f'\n{skipped} config(s) skipped — run the corresponding experiment scripts first.')
