"""Figures for the conditioning demo: discrimination + generalization."""
import json, sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

run = Path(sys.argv[1] if len(sys.argv) > 1 else 'runs/p1')
rows = [json.loads(l) for l in open(run / 'episodes.jsonl')]
by = {r['episode']: r for r in rows}
fig_dir = Path('figures'); fig_dir.mkdir(exist_ok=True)
AMBER, EYE, INK = '#B47B12', '#A93B2E', '#55575c'

# --- Fig L1: pre vs post MBON response, A vs B ---
pre_a, pre_b = by['pre_A']['mbon_total_hz'], by['pre_B']['mbon_total_hz']
post_a, post_b = by['post_A']['mbon_total_hz'], by['post_B']['mbon_total_hz']
fig, ax = plt.subplots(figsize=(6, 4))
x = [0, 1, 2.2, 3.2]
vals = [pre_a, post_a, pre_b, post_b]
cols = [AMBER, EYE, AMBER, INK]
ax.bar(x, vals, color=cols, width=0.8)
for xi, v in zip(x, vals):
    ax.text(xi, v, f'{v:.0f}', ha='center', va='bottom', fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(['A pre', 'A post\n(trained)', 'B pre', 'B post\n(control)'])
ax.set_ylabel('MBON population output (Hz)')
ax.set_title('Conditioning: trained odor loses MBON drive, control does not')
fig.tight_layout(); fig.savefig(fig_dir / 'l1_discrimination.png', dpi=150)

# --- Fig L2: generalization gradient ---
meta = json.load(open(run / 'meta.json'))
ovs = sorted({int(k.replace('probe', '')) for k in meta['probes']}, reverse=True)
# depression index for a probe: 1 - probe_response / matched-novel(0%) response
base = by['probe_0']['mbon_total_hz']
dep = [1 - by[f'probe_{o}']['mbon_total_hz'] / base for o in ovs]
dep_a = 1 - post_a / pre_a  # trained stimulus itself (100% overlap), vs its own pre
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot([100] + ovs, [dep_a] + dep, marker='o', color=AMBER)
ax.axhline(0, color=INK, lw=0.7, ls=':')
ax.set_xlabel('% of trained odor\'s input neurons shared')
ax.set_ylabel('response depression (fraction)')
ax.set_title('Generalization: learning transfers with stimulus similarity')
ax.invert_xaxis()
fig.tight_layout(); fig.savefig(fig_dir / 'l2_generalization.png', dpi=150)
print('figures written:', sorted(str(p) for p in fig_dir.glob('l*.png')))
