"""Find the sparse-KC regime: scan PN count x rate, report KC/PAM/MBON activity."""
import json, sys, time
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
pam_idx = np.array([flyid2i[i] for i in circ['pam'] if i in flyid2i])
mbons = [i for v in circ['mbon'].values() for i in v]
mbon_idx = np.array([flyid2i[i] for i in mbons if i in flyid2i])
pns = [i for i in circ['alpn_uni_chol'] if i in flyid2i]
rng = np.random.default_rng(0)
pns = list(rng.permutation(pns))[:40]  # fixed pool; configs use prefixes

params = dict(default_params)
neu, syn, spk_mon = create_model(f'{md}/Completeness_783.csv', f'{md}/Connectivity_783.parquet', params)

# dopamine is neuromodulatory, not fast-excitatory: silence DAN fast outputs
dan_idx = set((pam_idx.tolist() + [flyid2i[i] for i in circ['ppl1'] if i in flyid2i]))
df_con = pd.read_parquet(f'{md}/Connectivity_783.parquet')
pre_arr = df_con['Presynaptic_Index'].values
dan_out = np.flatnonzero(np.isin(pre_arr, list(dan_idx)))
w_all = np.array(syn.w[:])
w_all[dan_out] = 0.0
syn.w[:] = w_all * volt
print(f'[fix] zeroed {len(dan_out)} DAN output synapses', flush=True)
tgt = np.array([flyid2i[f] for f in pns])
pg = PoissonGroup(len(pns), rates=0 * Hz)
drv = Synapses(pg, neu, on_pre='v_post += w_drv', namespace={'w_drv': params['w_syn'] * params['f_poi']})
drv.connect(i=np.arange(len(pns)), j=tgt)
neu.rfc[tgt] = 0 * ms
net = Network(neu, syn, spk_mon, pg, drv)
prev = np.zeros(len(df_comp), dtype=np.int64)

for n_pn, hz in [(25, 100), (25, 50), (25, 25), (15, 50), (15, 25), (10, 25), (10, 15)]:
    rates = np.zeros(len(pns)); rates[:n_pn] = hz
    pg.rates = rates * Hz
    neu.v = params['v_0']; neu.g = 0 * mV
    net.run(1000 * ms)
    counts = np.array(spk_mon.count[:], dtype=np.int64); epi = counts - prev; prev = counts
    # washout: drain queues + settle before next config
    pg.rates = 0 * Hz; net.run(200 * ms)
    prev = np.array(spk_mon.count[:], dtype=np.int64)
    kc = epi[kc_idx]
    print(f'n_pn={n_pn:3d} hz={hz:4d}  KC>=3: {(kc>=3).sum():5d} ({100*(kc>=3).mean():.1f}%)  '
          f'KC>=1: {(kc>=1).sum():5d}  kc_p90={np.percentile(kc,90):.0f}  '
          f'PAM={epi[pam_idx].mean():6.2f}Hz  MBON={epi[mbon_idx].sum():7.0f}Hz  total_active={(epi>0).sum()}', flush=True)
