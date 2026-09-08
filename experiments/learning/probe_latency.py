"""How fast does the storm ignite? Binned population + KC spikes under PN drive."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

md = sys.argv[1]
sys.path.insert(0, md); sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import default_params
from model_ext import build_net
from brian2 import PoissonGroup, Synapses, Network, Hz, ms

here = Path(__file__).resolve().parent
circ = json.load(open(here / 'circuit_ids.json'))
df_comp = pd.read_csv(f'{md}/Completeness_783.csv', index_col=0)
flyid2i = {fid: i for i, fid in enumerate(df_comp.index)}
kc_idx = np.array([flyid2i[i] for i in circ['kc'] if i in flyid2i])
mbons = [i for v in circ['mbon'].values() for i in v]
mbon_idx = np.array([flyid2i[i] for i in mbons if i in flyid2i])
pns_all = [i for i in circ['alpn_uni_chol'] if i in flyid2i]
rng = np.random.default_rng(0)
tgt = np.array([flyid2i[f] for f in list(rng.permutation(pns_all))[:25]])

p, neu, syn, spk_mon = build_net(md, default_params)
pg = PoissonGroup(len(tgt), rates=0 * Hz)
drv = Synapses(pg, neu, on_pre='v_post += w_drv', namespace={'w_drv': p['w_syn'] * p['f_poi']})
drv.connect(i=np.arange(len(tgt)), j=tgt)
neu.rfc[tgt] = 0 * ms
net = Network(neu, syn, spk_mon, pg, drv)

pg.rates = 50 * Hz
prev = np.zeros(len(df_comp), dtype=np.int64)
for t in range(20):  # 20 x 50ms bins = 1s
    net.run(50 * ms)
    c = np.array(spk_mon.count[:], dtype=np.int64); e = c - prev; prev = c
    print(f'bin {t*50:4d}-{(t+1)*50:4d}ms  total={int(e.sum()):7d}  active={int((e>0).sum()):6d}  '
          f'KCspk={int(e[kc_idx].sum()):6d}  KC>=1={int((e[kc_idx]>=1).sum()):5d}  '
          f'MBONspk={int(e[mbon_idx].sum()):5d}', flush=True)
