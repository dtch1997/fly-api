"""Localize residual attractor; stage AL-side fixes. Fresh build per stage."""
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
          'dan': ids(ct.str.startswith('PAM') | ht.str.startswith('PAM') | ct.str.startswith('PPL1') | ht.str.startswith('PPL1'))}
keep = sorted(set(i for v in groups.values() for i in v))
orn = ann[orn_mask]
types = sorted(t for t, n in orn.cell_type.value_counts().items() if n >= 40)

def run_stage(label, alln_gain=1.0, pnpn_zero=False, apl_gain=5.0):
    p, neu, syn, spk_mon, old2new = build_subnet(md, default_params, keep)
    g = {k: np.array([old2new[i] for i in v]) for k, v in groups.items()}
    i_arr = np.array(syn.i[:]); j_arr = np.array(syn.j[:])
    w = np.array(syn.w[:])
    w[np.isin(i_arr, g['dan'])] = 0
    w[np.isin(i_arr, g['kc']) & np.isin(j_arr, g['kc'])] = 0
    w[np.isin(i_arr, g['apl'])] *= apl_gain
    if alln_gain != 1.0:
        alln_out = np.isin(i_arr, g['alln']) & (w < 0)
        w[alln_out] *= alln_gain
    pnpn = np.isin(i_arr, g['alpn']) & np.isin(j_arr, g['alpn'])
    if pnpn_zero: w[pnpn] = 0
    syn.w[:] = w * volt
    by_type = {t: [old2new[flyid2i[i]] for i in orn.loc[orn.cell_type == t, 'root_id'].astype(int) if i in flyid2i] for t in types}
    A = [i for t in types[0:4] for i in by_type[t]]
    B = [i for t in types[4:8] for i in by_type[t]]
    tgt = np.array(sorted(set(A + B))); pos = {i: k for k, i in enumerate(tgt)}
    pg = PoissonGroup(len(tgt), rates=0 * Hz)
    drv = Synapses(pg, neu, on_pre='v_post += w_drv', namespace={'w_drv': p['w_syn'] * p['f_poi']})
    drv.connect(i=np.arange(len(tgt)), j=tgt)
    neu.rfc[tgt] = 0 * ms
    net = Network(neu, syn, spk_mon, pg, drv)
    prev = np.zeros(len(keep), dtype=np.int64)
    def episode(stim, hz, dur=500):
        nonlocal prev
        r = np.zeros(len(tgt))
        for i in stim: r[pos[i]] = hz
        pg.rates = r * Hz
        neu.v = p['v_0']; neu.g = 0 * mV
        net.run(dur * ms)
        c = np.array(spk_mon.count[:], dtype=np.int64); e = c - prev; prev = c
        return e
    eA = episode(A, 50); e_sil = episode([], 0)
    eB = episode(B, 50); episode([], 0); eA2 = episode(A, 50)
    kA = set(np.flatnonzero(eA[g['kc']] >= 2).tolist())
    kB = set(np.flatnonzero(eB[g['kc']] >= 2).tolist())
    kA2 = set(np.flatnonzero(eA2[g['kc']] >= 2).tolist())
    parts = ' '.join(f'{k}:{int((e_sil[g[k]]>0).sum())}' for k in ('orn','alln','alpn','kc','mbon','dan'))
    print(f'{label:26s} pnpn_syn={int(pnpn.sum()):5d} sustained={int((e_sil>0).sum()):5d} [{parts}] '
          f'|A|={len(kA):4d} |B|={len(kB):4d} jacc={len(kA & kB)/max(len(kA | kB),1):.2f} '
          f'rel={len(kA & kA2)/max(len(kA | kA2),1):.2f} MBONspkA={int(eA[g["mbon"]].sum()):5d}', flush=True)

run_stage('fix3+apl5 (ref)')
run_stage('fix3+apl5+alln3', alln_gain=3)
run_stage('fix3+apl5+pnpn0', pnpn_zero=True)
run_stage('fix3+apl5+alln3+pnpn0', alln_gain=3, pnpn_zero=True)
