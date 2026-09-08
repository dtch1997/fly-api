"""Is there a global-gain window: no ignition, feeding circuit alive?"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd

md = sys.argv[1]
sys.path.insert(0, md); sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import default_params
from model_ext import build_net
from brian2 import PoissonGroup, Synapses, Network, Hz, ms, mV, volt

S = '/tmp/claude-2038/-mnt-nw-home-d-tan-jarvis-monorepo-jarvis-os/7e623888-45f1-4fd0-9563-5048d259e964/scratchpad'
here = Path(__file__).resolve().parent
ann = pd.read_csv(f'{S}/flywire-ann/annotations.tsv', sep='\t', low_memory=False)
cc = ann.cell_class.fillna(''); sc = ann.super_class.fillna('')
orn = ann[sc.str.contains('sensory') & cc.str.contains('olfactory')]
df_comp = pd.read_csv(f'{md}/Completeness_783.csv', index_col=0)
flyid2i = {fid: i for i, fid in enumerate(df_comp.index)}
circ = json.load(open(here / 'circuit_ids.json'))
sugar_ids = json.load(open(here / 'sugar_ids_783.json'))
kc_idx = np.array([flyid2i[i] for i in circ['kc'] if i in flyid2i])
mn9_idx = np.array([flyid2i[i] for i in (720575940660219265, 720575940645521262) if i in flyid2i])
da1 = [flyid2i[i] for i in orn.loc[orn.cell_type == 'ORN_DA1', 'root_id'].astype(int) if i in flyid2i]
sugar_tgt = [flyid2i[f] for f in sugar_ids]

p, neu, syn, spk_mon = build_net(md, default_params)
w_base = np.array(syn.w[:])
tgt = np.array(da1 + sugar_tgt)
pg = PoissonGroup(len(tgt), rates=0 * Hz)
drv = Synapses(pg, neu, on_pre='v_post += w_drv', namespace={'w_drv': p['w_syn'] * p['f_poi']})
drv.connect(i=np.arange(len(tgt)), j=tgt)
neu.rfc[tgt] = 0 * ms
net = Network(neu, syn, spk_mon, pg, drv)
prev = np.zeros(len(df_comp), dtype=np.int64)

def win(hz_orn, hz_sugar, dur):
    global prev
    r = np.zeros(len(tgt)); r[:len(da1)] = hz_orn; r[len(da1):] = hz_sugar
    pg.rates = r * Hz
    net.run(dur * ms)
    c = np.array(spk_mon.count[:], dtype=np.int64); e = c - prev; prev = c
    return e

for scale in [0.55, 0.7, 0.85]:
    syn.w[:] = (w_base * scale) * volt
    neu.v = p['v_0']; neu.g = 0 * mV
    base = win(0, 0, 200)
    e_odor = win(50, 0, 300)
    e_sil = win(0, 0, 300)
    neu.v = p['v_0']; neu.g = 0 * mV
    win(0, 0, 200); prev = np.array(spk_mon.count[:], dtype=np.int64)
    e_sug = win(0, 150, 500)
    print(f'scale={scale:4.2f}  baseline={int((base>0).sum()):5d}  '
          f'odor(DA1@50): active={int((e_odor>0).sum()):6d} KC>=1={int((e_odor[kc_idx]>=1).sum()):5d}  '
          f'sustained={int((e_sil>0).sum()):6d}  '
          f'sugar: MN9={float(e_sug[mn9_idx].mean()/0.5):6.1f}Hz feeding={int((e_sug>0).sum()):5d}', flush=True)
