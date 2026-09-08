"""Is the whole-brain LIF net self-sustaining once kicked? Fresh net, staged windows."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

md = sys.argv[1]
sys.path.insert(0, md)
from model import create_model, default_params
from brian2 import PoissonGroup, Synapses, Network, Hz, ms, mV, volt

here = Path(__file__).resolve().parent
circ = json.load(open(here / 'circuit_ids.json'))
df_comp = pd.read_csv(f'{md}/Completeness_783.csv', index_col=0)
flyid2i = {fid: i for i, fid in enumerate(df_comp.index)}
kc_idx = np.array([flyid2i[i] for i in circ['kc'] if i in flyid2i])
pns = [i for i in circ['alpn_uni_chol'] if i in flyid2i]
rng = np.random.default_rng(0)
pns = list(rng.permutation(pns))[:10]

params = dict(default_params)
neu, syn, spk_mon = create_model(f'{md}/Completeness_783.csv', f'{md}/Connectivity_783.parquet', params)
tgt = np.array([flyid2i[f] for f in pns])
pg = PoissonGroup(len(pns), rates=0 * Hz)
drv = Synapses(pg, neu, on_pre='v_post += w_drv', namespace={'w_drv': params['w_syn'] * params['f_poi']})
drv.connect(i=np.arange(len(pns)), j=tgt)
neu.rfc[tgt] = 0 * ms
net = Network(neu, syn, spk_mon, pg, drv)
prev = np.zeros(len(df_comp), dtype=np.int64)

def window(name, hz, dur_ms):
    global prev
    rates = np.zeros(len(pns)); rates[:] = hz
    pg.rates = rates * Hz
    net.run(dur_ms * ms)
    c = np.array(spk_mon.count[:], dtype=np.int64); epi = c - prev; prev = c
    print(f'{name:20s} {dur_ms:5.0f}ms drive={hz:3.0f}Hz  active={int((epi>0).sum()):6d}  '
          f'KC>=1={int((epi[kc_idx]>=1).sum()):5d}  spikes={int(epi.sum()):8d}', flush=True)

window('baseline_silence', 0, 1000)
window('kick_10pn_15hz', 15, 1000)
window('post_silence_1', 0, 1000)
window('post_silence_2', 0, 1000)
window('post_silence_3', 0, 1000)
