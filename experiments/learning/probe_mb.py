"""Olfactory->MB subcircuit: stable? sparse KC codes? distinct A vs B?"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd

md = sys.argv[1]
sys.path.insert(0, md); sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import default_params
from model_ext import build_subnet
from brian2 import PoissonGroup, Synapses, Network, Hz, ms, mV

S = '/tmp/claude-2038/-mnt-nw-home-d-tan-jarvis-monorepo-jarvis-os/7e623888-45f1-4fd0-9563-5048d259e964/scratchpad'
here = Path(__file__).resolve().parent
ann = pd.read_csv(f'{S}/flywire-ann/annotations.tsv', sep='\t', low_memory=False)
cc = ann.cell_class.fillna(''); sc = ann.super_class.fillna(''); ct = ann.cell_type.fillna(''); ht = ann.hemibrain_type.fillna('')
df_comp = pd.read_csv(f'{md}/Completeness_783.csv', index_col=0)
flyid2i = {fid: i for i, fid in enumerate(df_comp.index)}
def ids(mask): return [flyid2i[i] for i in ann.loc[mask, 'root_id'].astype(int) if i in flyid2i]

orn_mask = sc.str.contains('sensory') & cc.str.contains('olfactory')
groups = {
    'orn': ids(orn_mask),
    'alln': ids(cc == 'ALLN'),
    'alpn': ids(cc == 'ALPN'),
    'kc': ids(cc.str.contains('Kenyon', case=False)),
    'apl': ids(ct == 'APL'),
    'mbon': ids(ct.str.startswith('MBON') | ht.str.startswith('MBON')),
    'dan': ids(ct.str.startswith('PAM') | ht.str.startswith('PAM') | ct.str.startswith('PPL1') | ht.str.startswith('PPL1')),
}
print({k: len(v) for k, v in groups.items()})
keep = sorted(set(i for v in groups.values() for i in v))
p, neu, syn, spk_mon, old2new = build_subnet(md, default_params, keep)
g_new = {k: np.array([old2new[i] for i in v]) for k, v in groups.items()}

orn = ann[orn_mask]
types = sorted(t for t, n in orn.cell_type.value_counts().items() if n >= 40)
by_type = {t: [old2new[flyid2i[i]] for i in orn.loc[orn.cell_type == t, 'root_id'].astype(int) if i in flyid2i] for t in types}
print(f'{len(types)} ORN types usable')
A_types, B_types = types[0:4], types[4:8]
A = [i for t in A_types for i in by_type[t]]
B = [i for t in B_types for i in by_type[t]]

tgt = np.array(sorted(set(A + B)))
pos = {i: k for k, i in enumerate(tgt)}
pg = PoissonGroup(len(tgt), rates=0 * Hz)
drv = Synapses(pg, neu, on_pre='v_post += w_drv', namespace={'w_drv': p['w_syn'] * p['f_poi']})
drv.connect(i=np.arange(len(tgt)), j=tgt)
neu.rfc[tgt] = 0 * ms
net = Network(neu, syn, spk_mon, pg, drv)
prev = np.zeros(len(keep), dtype=np.int64)

def episode(name, stim, hz, dur=500):
    global prev
    r = np.zeros(len(tgt))
    for i in stim: r[pos[i]] = hz
    pg.rates = r * Hz
    neu.v = p['v_0']; neu.g = 0 * mV
    net.run(dur * ms)
    c = np.array(spk_mon.count[:], dtype=np.int64); e = c - prev; prev = c
    kc_on = np.flatnonzero(e[g_new['kc']] >= 2)
    print(f'{name:10s} active={int((e>0).sum()):5d} KCon={len(kc_on):5d} '
          f'PN={int((e[g_new["alpn"]]>0).sum()):4d} MBONspk={int(e[g_new["mbon"]].sum()):6d} '
          f'APLspk={int(e[g_new["apl"]].sum()):4d} spikes={int(e.sum()):7d}', flush=True)
    return set(kc_on.tolist())

episode('silence', [], 0)
kcA1 = episode('A@50', A, 50)
episode('silence2', [], 0)
kcB1 = episode('B@50', B, 50)
episode('silence3', [], 0)
kcA2 = episode('A@50 rpt', A, 50)
inter = len(kcA1 & kcB1); union = max(len(kcA1 | kcB1), 1)
rel = len(kcA1 & kcA2) / max(len(kcA1 | kcA2), 1)
print(f'KC codes: |A|={len(kcA1)} |B|={len(kcB1)} A-B jaccard={inter/union:.2f}  A-A reliability={rel:.2f}')
json.dump({k: [int(x) for x in v] for k, v in g_new.items()} | {'keep_flyidx': [int(i) for i in keep]},
          open(here / 'mb_subnet_ids.json', 'w'))
