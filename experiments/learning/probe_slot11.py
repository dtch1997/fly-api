"""Microscopic attribution: what drives the A-responding MBON, and were
those synapses depressed?"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd

md = sys.argv[1]
sys.path.insert(0, md); sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import default_params
from model_ext import build_subnet
from brian2 import PoissonGroup, Synapses, Network, Hz, ms, mV, volt

S = '/tmp/claude-2038/-mnt-nw-home-d-tan-jarvis-monorepo-jarvis-os/7e623888-45f1-4fd0-9563-5048d259e964/scratchpad'
here = Path(__file__).resolve().parent
ann = pd.read_csv(f'{S}/flywire-ann/annotations.tsv', sep='\t', low_memory=False)
cc = ann.cell_class.fillna(''); sc = ann.super_class.fillna(''); ct = ann.cell_type.fillna(''); ht = ann.hemibrain_type.fillna('')
df_comp = pd.read_csv(f'{md}/Completeness_783.csv', index_col=0)
flyid2i = {fid: i for i, fid in enumerate(df_comp.index)}
def ids(mask): return [flyid2i[i] for i in ann.loc[mask, 'root_id'].astype(int) if i in flyid2i]
orn_mask = sc.str.contains('sensory') & cc.str.contains('olfactory')
groups = {'orn': ids(orn_mask), 'alln': ids(cc == 'ALLN'), 'alpn': ids(cc == 'ALPN'),
          'kc': ids(cc.str.contains('Kenyon', case=False)), 'apl': ids(ct == 'APL'),
          'mbon': ids(ct.str.startswith('MBON') | ht.str.startswith('MBON')),
          'pam': ids(ct.str.startswith('PAM') | ht.str.startswith('PAM')),
          'ppl1': ids(ct.str.startswith('PPL1') | ht.str.startswith('PPL1'))}
keep = sorted(set(i for v in groups.values() for i in v))
p, neu, syn, spk_mon, old2new = build_subnet(md, default_params, keep)
g = {k: np.array(sorted(old2new[i] for i in v)) for k, v in groups.items()}
i_arr = np.array(syn.i[:]); j_arr = np.array(syn.j[:])
w = np.array(syn.w[:])
w[np.isin(i_arr, np.concatenate([g['pam'], g['ppl1']]))] = 0
w[np.isin(i_arr, g['kc']) & np.isin(j_arr, g['kc'])] = 0
w[np.isin(j_arr, g['orn'])] = 0
w[np.isin(i_arr, g['alln']) & (w > 0)] = 0
syn.w[:] = w * volt

orn = ann[orn_mask]
counts = orn.cell_type.value_counts()
types = sorted((t for t in counts.index if counts[t] >= 40), key=lambda t: -counts[t])
rng = np.random.default_rng(0)
order = list(rng.permutation(18))
A_types = [types[order[k]] for k in range(0, 12, 2)]
by_type = {t: [old2new[flyid2i[i]] for i in orn.loc[orn.cell_type == t, 'root_id'].astype(int) if i in flyid2i] for t in types[:18]}
A = [i for t in A_types for i in by_type[t]]
tgt = np.array(sorted(set(A)))
pg = PoissonGroup(len(tgt), rates=500 * Hz)
drv = Synapses(pg, neu, on_pre='v_post += w_drv', namespace={'w_drv': p['w_syn'] * p['f_poi']})
drv.connect(i=np.arange(len(tgt)), j=tgt)
neu.rfc[tgt] = 0 * ms
net = Network(neu, syn, spk_mon, pg, drv)
neu.v = p['v_0']; neu.g = 0 * mV
net.run(500 * ms)
e = np.array(spk_mon.count[:], dtype=np.int64)

mbon_sorted = g['mbon']
mv = e[mbon_sorted]
slots = np.flatnonzero(mv > 2)
print('responding slots:', slots.tolist(), 'spikes:', mv[slots].tolist())
slot = slots[0] if len(slots) else 11
mb = mbon_sorted[slot]
inm = np.flatnonzero(j_arr == mb)
srcs = i_arr[inm]; ws = np.array(syn.w[:])[inm]
spk = e[srcs]
drive = ws * spk
order2 = np.argsort(-np.abs(drive))
print(f'slot {slot} total drive={drive.sum():.4f} V·spk ; top sources:')
for k in order2[:12]:
    s = srcs[k]
    grp = next((n for n, arr in g.items() if s in set(arr.tolist())), '?')
    print(f'  pre={s} grp={grp} w={ws[k]*1000:.3f}mV spikes={spk[k]} drive={drive[k]*1000:.2f} mV·spk')
kc_set = set(g['kc'].tolist())
kc_drive = drive[[i for i, s in enumerate(srcs) if s in kc_set]].sum()
print(f'KC share of drive: {kc_drive/max(drive.sum(),1e-12):.2%}')
kc_active = set(np.flatnonzero(e >= 1).tolist()) & kc_set
kc_srcs_active = [i for i, s in enumerate(srcs) if s in kc_set and e[s] >= 1]
print(f'A-active KCs among slot inputs: {len(kc_srcs_active)}; their drive {drive[kc_srcs_active].sum()*1000:.2f} mV·spk')
