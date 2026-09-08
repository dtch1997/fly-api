"""Drive odors at the periphery (ORN classes) — is the response stable and
are KC codes sparse? 50ms-binned, two different 'odors'."""
import sys
from pathlib import Path
import json
import numpy as np, pandas as pd

md = sys.argv[1]
sys.path.insert(0, md); sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import default_params
from model_ext import build_net
from brian2 import PoissonGroup, Synapses, Network, Hz, ms, mV

S = '/tmp/claude-2038/-mnt-nw-home-d-tan-jarvis-monorepo-jarvis-os/7e623888-45f1-4fd0-9563-5048d259e964/scratchpad'
here = Path(__file__).resolve().parent
ann = pd.read_csv(f'{S}/flywire-ann/annotations.tsv', sep='\t', low_memory=False)
cc = ann.cell_class.fillna(''); sc = ann.super_class.fillna(''); ct = ann.cell_type.fillna('')
orn = ann[sc.str.contains('sensory') & cc.str.contains('olfactory')]
types = orn.cell_type.value_counts()
print('ORN types:', len(types), 'top:', dict(types.head(8)))
df_comp = pd.read_csv(f'{md}/Completeness_783.csv', index_col=0)
flyid2i = {fid: i for i, fid in enumerate(df_comp.index)}
circ = json.load(open(here / 'circuit_ids.json'))
kc_idx = np.array([flyid2i[i] for i in circ['kc'] if i in flyid2i])
mbons = [i for v in circ['mbon'].values() for i in v]
mbon_idx = np.array([flyid2i[i] for i in mbons if i in flyid2i])

# two disjoint 'odors': 4 ORN classes each (deterministic pick from sorted list)
tnames = sorted(t for t in types.index if types[t] >= 20)
odorA_types, odorB_types = tnames[0:4], tnames[4:8]
def orn_ids(ts):
    ids = orn.loc[orn.cell_type.isin(ts), 'root_id'].astype(int)
    return [flyid2i[i] for i in ids if i in flyid2i]
A, B = orn_ids(odorA_types), orn_ids(odorB_types)
print(f'odorA={odorA_types} n={len(A)}; odorB={odorB_types} n={len(B)}')

p, neu, syn, spk_mon = build_net(md, default_params)
tgt = np.array(A + B)
pg = PoissonGroup(len(tgt), rates=0 * Hz)
drv = Synapses(pg, neu, on_pre='v_post += w_drv', namespace={'w_drv': p['w_syn'] * p['f_poi']})
drv.connect(i=np.arange(len(tgt)), j=tgt)
neu.rfc[tgt] = 0 * ms
net = Network(neu, syn, spk_mon, pg, drv)
prev = np.zeros(len(df_comp), dtype=np.int64)

def phase(name, ra, rb, dur_bins):
    global prev
    r = np.zeros(len(tgt)); r[:len(A)] = ra; r[len(A):] = rb
    pg.rates = r * Hz
    kc_sets = []
    for t in range(dur_bins):
        net.run(50 * ms)
        c = np.array(spk_mon.count[:], dtype=np.int64); e = c - prev; prev = c
        kc_on = e[kc_idx] >= 1
        kc_sets.append(set(np.flatnonzero(kc_on)))
        print(f'{name} bin{t}  total={int(e.sum()):7d} active={int((e>0).sum()):6d} '
              f'KC>=1={int(kc_on.sum()):5d} MBONspk={int(e[mbon_idx].sum()):5d}', flush=True)
    return kc_sets

kcA = phase('A@100', 100, 0, 6)
neu.v = p['v_0']; neu.g = 0 * mV
phase('washout', 0, 0, 4)
kcB = phase('B@100', 0, 100, 6)
a = kcA[-1] | kcA[-2]; b = kcB[-1] | kcB[-2]
inter = len(a & b); union = len(a | b)
print(f'KC codes: |A|={len(a)} |B|={len(b)} overlap={inter} jaccard={inter/max(union,1):.2f}')
