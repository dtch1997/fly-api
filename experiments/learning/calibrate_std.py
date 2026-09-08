"""Calibrate short-term depression: kill ignition, preserve sugar->MN9."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

md = sys.argv[1]
sys.path.insert(0, md); sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import default_params
from model_ext import build_net
from brian2 import PoissonGroup, Synapses, Network, Hz, ms, mV

here = Path(__file__).resolve().parent
circ = json.load(open(here / 'circuit_ids.json'))
sugar_ids = json.load(open(here / 'sugar_ids_783.json'))
df_comp = pd.read_csv(f'{md}/Completeness_783.csv', index_col=0)
flyid2i = {fid: i for i, fid in enumerate(df_comp.index)}
kc_idx = np.array([flyid2i[i] for i in circ['kc'] if i in flyid2i])
mn9_idx = np.array([flyid2i[i] for i in (720575940660219265, 720575940645521262) if i in flyid2i])
pns_all = [i for i in circ['alpn_uni_chol'] if i in flyid2i]
rng = np.random.default_rng(0)
pn10 = [flyid2i[f] for f in list(rng.permutation(pns_all))[:10]]
sugar_tgt = [flyid2i[f] for f in sugar_ids]

for U, tau_rec in [(0.1, 500), (0.2, 500), (0.3, 500), (0.4, 800)]:
    p, neu, syn, spk_mon = build_net(md, default_params, std=(U, tau_rec))
    tgt = np.array(pn10 + sugar_tgt)
    pg = PoissonGroup(len(tgt), rates=0 * Hz)
    drv = Synapses(pg, neu, on_pre='v_post += w_drv', namespace={'w_drv': p['w_syn'] * p['f_poi']})
    drv.connect(i=np.arange(len(tgt)), j=tgt)
    neu.rfc[tgt] = 0 * ms
    net = Network(neu, syn, spk_mon, pg, drv)
    prev = np.zeros(len(df_comp), dtype=np.int64)
    def win(hz_pn, hz_sugar, dur):
        global prev
        r = np.zeros(len(tgt)); r[:10] = hz_pn; r[10:] = hz_sugar
        pg.rates = r * Hz
        net.run(dur * ms)
        c = np.array(spk_mon.count[:], dtype=np.int64); e = c - prev; prev = c
        return e
    e_kick = win(25, 0, 1000)
    e_sil = win(0, 0, 1000)
    sustained = int((e_sil > 0).sum())
    kc_kick = int((e_kick[kc_idx] >= 3).sum())
    neu.v = p['v_0']; neu.g = 0 * mV; syn.x = 1.0
    win(0, 0, 300)
    prev = np.array(spk_mon.count[:], dtype=np.int64)
    e_sug = win(0, 150, 1000)
    print(f'U={U:.1f} tau_rec={tau_rec:4d}ms  kick: active={int((e_kick>0).sum()):6d} KCact={kc_kick:5d}  '
          f'sustained={sustained:6d}  sugar: MN9={float(e_sug[mn9_idx].mean()):6.1f}Hz feeding={int((e_sug>0).sum()):5d}', flush=True)
