"""Figures: acquisition, generalization (MBON), generalization (synaptic trace)."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

AMBER, EYE, INK, OK = '#B47B12', '#A93B2E', '#55575c', '#4A7A3A'
fig_dir = Path('figures'); fig_dir.mkdir(exist_ok=True)

def load(run):
    rows = [json.loads(l) for l in open(f'runs/{run}/episodes.jsonl')]
    return {r['episode']: r for r in rows}, rows

seeds = ['mb_g20_s0', 'mb_g20_s1', 'mb_g20_s2']

# --- Fig 1: acquisition ---
fig, ax = plt.subplots(figsize=(6.5, 4))
for si, run in enumerate(seeds):
    by, rows = load(run)
    a_eps = [r for r in rows if r['stimulus'] == 'A' and 'probe' not in r['episode']]
    b_eps = [r for r in rows if r['stimulus'] == 'B' and 'probe' not in r['episode']]
    x = np.arange(len(a_eps))
    ax.plot(x, [r['mbon_spk'] for r in a_eps], marker='o', color=AMBER,
            alpha=0.8, label='odor A (paired with reward)' if si == 0 else None)
    ax.plot(x, [r['mbon_spk'] for r in b_eps], marker='s', color=INK,
            alpha=0.6, label='odor B (unpaired control)' if si == 0 else None)
ax.axvspan(1.5, 6.5, color=EYE, alpha=0.08)
ax.text(4, ax.get_ylim()[1] * 0.95, 'training (5 pairings)', ha='center', color=EYE, fontsize=9)
ax.set_xticks(range(9))
ax.set_xticklabels(['pre1', 'pre2', 't1', 't2', 't3', 't4', 't5', 'post1', 'post2'])
ax.set_ylabel('MBON population response (spikes / 0.5 s)')
ax.set_title('Acquisition: the paired odor loses its MBON response (3 seeds)')
ax.legend(frameon=False)
fig.tight_layout(); fig.savefig(fig_dir / 'm1_acquisition.png', dpi=150)

# --- Fig 2: generalization, MBON readout ---
fig, ax = plt.subplots(figsize=(6.5, 4))
xs = [100, 67, 50, 33, 0]  # % shared classes: A(6/6), probe4, probe3, probe2, probe0
for si, run in enumerate(seeds):
    by, _ = load(run)
    pre_a = np.mean([by['pre_A0']['mbon_spk'], by['pre_A1']['mbon_spk']])
    post_a = np.mean([by['post_A0']['mbon_spk'], by['post_A1']['mbon_spk']])
    vals = [1 - post_a / pre_a]
    for k in (4, 3, 2, 0):
        pr = by[f'probe_{k}']['mbon_spk']
        base = by['probe_0']['mbon_spk']
        vals.append(1 - pr / max(base, 1))
    ax.plot(xs, vals, marker='o', color=AMBER, alpha=0.8,
            label='suppression vs novel-odor baseline' if si == 0 else None)
ax.axhline(0, color=INK, lw=0.7, ls=':')
ax.set_xlabel('% of trained odor\'s ORN classes shared')
ax.set_ylabel('learned suppression (fraction)')
ax.set_title('Generalization: suppression transfers with odor similarity (3 seeds)')
ax.invert_xaxis(); ax.legend(frameon=False, loc='upper right')
fig.tight_layout(); fig.savefig(fig_dir / 'm2_generalization_mbon.png', dpi=150)

# --- Fig 3: synaptic trace, incl. parameter-free arm ---
fig, ax = plt.subplots(figsize=(6.5, 4))
for si, run in enumerate(seeds):
    by, _ = load(run)
    vals = [np.mean([by['post_A0']['trace_depression'], by['post_A1']['trace_depression']])]
    vals += [by[f'probe_{k}']['trace_depression'] for k in (4, 3, 2, 0)]
    ax.plot(xs, vals, marker='o', color=AMBER, alpha=0.8,
            label='with readout gain (G=20)' if si == 0 else None)
try:
    by, _ = load('mb_g1_s0')
    vals = [np.mean([by['post_A0']['trace_depression'], by['post_A1']['trace_depression']])]
    vals += [by[f'probe_{k}']['trace_depression'] for k in (4, 3, 2, 0)]
    ax.plot(xs, vals, marker='D', color=OK, label='no gain (G=1, parameter-free)')
except FileNotFoundError:
    pass
for si, run in enumerate(seeds):
    by, _ = load(run)
    b_tr = np.mean([by['post_B0']['trace_depression'], by['post_B1']['trace_depression']])
    ax.plot([0], [b_tr], marker='x', color=INK)
ax.set_xlabel('% of trained odor\'s ORN classes shared')
ax.set_ylabel('synaptic trace: KC→MBON weight depressed\nwithin the probe\'s KC code')
ax.set_title('The memory trace generalizes smoothly (and needs no readout gain)')
ax.invert_xaxis(); ax.legend(frameon=False)
fig.tight_layout(); fig.savefig(fig_dir / 'm3_generalization_trace.png', dpi=150)
print('figures:', sorted(str(p) for p in fig_dir.glob('m*.png')))
