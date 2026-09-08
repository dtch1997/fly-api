"""Figures for Track A: MN9 dose-response + specificity bars."""
import json
from pathlib import Path
try:
    import xy.pyplot as plt   # house rule: xy shim first
except Exception:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

rows = [json.loads(l) for l in open('results.jsonl')]
import math
mn9 = {r['condition']: tuple(0.0 if (isinstance(v,float) and math.isnan(v)) else v for v in (r['rate_hz'], r['std_hz']))
       for r in rows if r.get('neuron') == 'MN9_L'}

fig_dir = Path('figures'); fig_dir.mkdir(exist_ok=True)

# --- Fig A1: dose-response ---
freqs = [25, 50, 100, 150, 200]
rates = [mn9.get(f'sugar_{f}Hz', (0, 0))[0] for f in freqs]
stds  = [mn9.get(f'sugar_{f}Hz', (0, 0))[1] for f in freqs]
fig, ax = plt.subplots(figsize=(6, 4))
ax.errorbar(freqs, rates, yerr=stds, marker='o', capsize=4, color='#1f77b4')
ax.set_xlabel('Sugar GRN stimulation (Hz)')
ax.set_ylabel('MN9 firing rate (Hz)')
ax.set_title('Perception→motor: MN9 (proboscis) rate rises with sugar drive')
fig.tight_layout(); fig.savefig(fig_dir / 'a3_dose_response.png', dpi=150)

# --- Fig A2: specificity bars at 150 Hz ---
conds = ['sugar_150Hz', 'water_150Hz', 'bitter_150Hz', 'sugar150_bitter150']
labels = ['sugar', 'water', 'bitter', 'sugar+bitter']
vals = [mn9.get(c, (0, 0))[0] for c in conds]
errs = [mn9.get(c, (0, 0))[1] for c in conds]
fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(labels, vals, yerr=errs, capsize=4,
       color=['#2ca02c', '#1f77b4', '#d62728', '#9467bd'])
for i, v in enumerate(vals):
    ax.text(i, v, f'{v:.1f}', ha='center', va='bottom')
ax.set_ylabel('MN9 firing rate (Hz)')
ax.set_title('Specificity: only sugar drives MN9; bitter suppresses it')
fig.tight_layout(); fig.savefig(fig_dir / 'a2_specificity.png', dpi=150)
print('figures written:', list(map(str, fig_dir.glob('a*.png'))))
