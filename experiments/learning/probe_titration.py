"""Ascend ORN drive from sugar-scale; find the ignition knee and any
subcritical regime with a usable (sparse, nonzero) KC code."""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd

md = sys.argv[1]
sys.path.insert(0, md); sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import default_params
from model_ext import build_net
from brian2 import PoissonGroup, Synapses, Network, Hz, ms, mV

S = '/tmp/claude-2038/-mnt-nw-home-d-tan-jarvis-monorepo-jarvis-os/7e623888-45f1-4fd0-9563-5048d259e964/scratchpad'
here = Path(__file__).resolve().parent
ann = pd.read_csv(f'{S}/flywire-ann/annotations.tsv', sep='\t', low_memory=False)
cc = ann.cell_class.fillna(''); sc = ann.super_class.fillna('')
orn = ann[sc.str.contains('sensory') & cc.str.contains('olfactory')]
df_comp = pd.read_csv(f'{md}/Completeness_783.csv', index_col=0)
flyid2i = {fid: i for i, fid in enumerate(df_comp.index)}
circ = json.load(open(here / 'circuit_ids.json'))
kc_idx = np.array([flyid2i[i] for i in circ['kc'] if i in flyid2i])

types = sorted(t for t, n in orn.cell_type.value_counts().items() if n >= 40)
by_type = {t: [flyid2i[i] for i in orn.loc[orn.cell_type == t, 'root_id'].astype(int) if i in flyid2i] for t in types}
all_ids = [i for t in types for i in by_type[t]]

p, neu, syn, spk_mon = build_net(md, default_params)
tgt = np.array(all_ids)
pos = {i: k for k, i in enumerate(all_ids)}
pg = PoissonGroup(len(tgt), rates=0 * Hz)
drv = Synapses(pg, neu, on_pre='v_post += w_drv', namespace={'w_drv': p['w_syn'] * p['f_poi']})
drv.connect(i=np.arange(len(tgt)), j=tgt)
neu.rfc[tgt] = 0 * ms
net = Network(neu, syn, spk_mon, pg, drv)
prev = np.zeros(len(df_comp), dtype=np.int64)

ladder = [(1, 10), (1, 25), (1, 50), (2, 25), (1, 100), (2, 50), (3, 50), (4, 50)]
for k, hz in ladder:
    ids = [i for t in types[:k] for i in by_type[t]]
    drive = len(ids) * hz
    rates = np.zeros(len(tgt))
    for i in ids: rates[pos[i]] = hz
    pg.rates = rates * Hz
    neu.v = p['v_0']; neu.g = 0 * mV
    net.run(300 * ms)
    c = np.array(spk_mon.count[:], dtype=np.int64); e = c - prev; prev = c
    kc_on = int((e[kc_idx] >= 1).sum())
    act = int((e > 0).sum())
    print(f'k={k} hz={hz:3d} n_orn={len(ids):3d} drive={drive:6d}/s  active={act:6d}  KC>=1={kc_on:5d}  spikes={int(e.sum()):7d}', flush=True)
    # silence check
    pg.rates = np.zeros(len(tgt)) * Hz
    neu.v = p['v_0']; neu.g = 0 * mV
    net.run(200 * ms)
    c = np.array(spk_mon.count[:], dtype=np.int64); e2 = c - prev; prev = c
    ok = int((e2 > 0).sum())
    print(f'          post-silence active={ok:6d} {"<IGNITED - stopping ladder>" if ok > 2000 else ""}', flush=True)
    if ok > 2000:
        break
