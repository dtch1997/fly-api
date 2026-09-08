"""Calibrate spike-frequency adaptation: smallest b that kills the ignition
attractor while preserving the sugar->MN9 feeding result."""
import json, sys
from pathlib import Path
from textwrap import dedent
import numpy as np, pandas as pd

md = sys.argv[1]
sys.path.insert(0, md)
from model import create_model, default_params
from brian2 import PoissonGroup, Synapses, Network, Hz, ms, mV, volt

here = Path(__file__).resolve().parent
circ = json.load(open(here / 'circuit_ids.json'))
sugar_ids = json.load(open(here / 'sugar_ids_783.json'))
df_comp = pd.read_csv(f'{md}/Completeness_783.csv', index_col=0)
flyid2i = {fid: i for i, fid in enumerate(df_comp.index)}
kc_idx = np.array([flyid2i[i] for i in circ['kc'] if i in flyid2i])
MN9 = [i for i in (720575940660219265, 720575940645521262) if i in flyid2i]
print('MN9 in 783:', len(MN9))
mn9_idx = np.array([flyid2i[i] for i in MN9])
pns_all = [i for i in circ['alpn_uni_chol'] if i in flyid2i]
rng = np.random.default_rng(0)
pn10 = [flyid2i[f] for f in list(rng.permutation(pns_all))[:10]]
sugar_tgt = [flyid2i[f] for f in sugar_ids]

def build(b_mV, tau_ms):
    p = dict(default_params)
    if b_mV > 0:
        p['t_a'] = tau_ms * ms
        p['b_a'] = b_mV * mV
        p['eqs'] = dedent('''
            dv/dt = (v_0 - v + g - a) / t_mbr : volt (unless refractory)
            dg/dt = -g / tau               : volt (unless refractory)
            da/dt = -a / t_a               : volt
            rfc                            : second
            ''')
        p['eq_rst'] = 'v = v_rst; w = 0; g = 0 * mV; a += b_a'
    neu, syn, spk_mon = create_model(f'{md}/Completeness_783.csv', f'{md}/Connectivity_783.parquet', p)
    tgt = np.array(pn10 + sugar_tgt)
    pg = PoissonGroup(len(tgt), rates=0 * Hz)
    drv = Synapses(pg, neu, on_pre='v_post += w_drv', namespace={'w_drv': p['w_syn'] * p['f_poi']})
    drv.connect(i=np.arange(len(tgt)), j=tgt)
    neu.rfc[tgt] = 0 * ms
    net = Network(neu, syn, spk_mon, pg, drv)
    return p, neu, syn, spk_mon, pg, net, tgt

for b, tau in [(8.0, 100), (16.0, 100), (8.0, 300), (16.0, 300), (32.0, 300)]:
    p, neu, syn, spk_mon, pg, net, tgt = build(b, tau)
    prev = np.zeros(len(df_comp), dtype=np.int64)
    def win(hz_pn, hz_sugar, dur):
        global prev
        r = np.zeros(len(tgt)); r[:10] = hz_pn; r[10:] = hz_sugar
        pg.rates = r * Hz
        net.run(dur * ms)
        c = np.array(spk_mon.count[:], dtype=np.int64); e = c - prev; prev = c
        return e
    # ignition test: kick PNs, then silence
    win(25, 0, 1000)  # deterministic-ish ignition kick
    e_sil = win(0, 0, 1000)
    sustained = int((e_sil > 0).sum())
    # reset state, sugar regression
    neu.v = p['v_0']; neu.g = 0 * mV
    if b > 0: neu.a = 0 * mV
    win(0, 0, 300)  # settle
    prev = np.array(spk_mon.count[:], dtype=np.int64)
    e_sug = win(0, 150, 1000)
    mn9 = float(e_sug[mn9_idx].mean())
    kc_sug = int((e_sug[kc_idx] >= 3).sum())
    print(f'b={b:4.1f}mV tau={tau:3d}ms  sustained_active={sustained:6d}  sugar: MN9={mn9:6.1f}Hz  '
          f'feeding_active={(e_sug>0).sum():5d}  KCact={kc_sug}', flush=True)
