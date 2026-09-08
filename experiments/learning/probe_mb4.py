"""Final stabilization pass: + ORNs input-only; small grid over gains/rate."""
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

def run_stage(label, alln_gain, apl_gain, hz, eln0=False, pnpn0=False, alff=False):
    p, neu, syn, spk_mon, old2new = build_subnet(md, default_params, keep)
    g = {k: np.array([old2new[i] for i in v]) for k, v in groups.items()}
    i_arr = np.array(syn.i[:]); j_arr = np.array(syn.j[:])
    w = np.array(syn.w[:])
    w[np.isin(i_arr, g['dan'])] = 0
    w[np.isin(i_arr, g['kc']) & np.isin(j_arr, g['kc'])] = 0
    w[np.isin(j_arr, g['orn'])] = 0            # ORNs input-only
    w[np.isin(i_arr, g['apl'])] *= apl_gain
    w[np.isin(i_arr, g['alln']) & (w < 0)] *= alln_gain
    if eln0:
        w[np.isin(i_arr, g['alln']) & (w > 0)] = 0
    if pnpn0:
        w[np.isin(i_arr, g['alpn']) & np.isin(j_arr, g['alpn'])] = 0
    if alff:
        al = np.concatenate([g['alln'], g['alpn']])
        w[np.isin(i_arr, al) & np.isin(j_arr, al) & (w > 0)] = 0
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
    def episode(stim, r_hz, dur=500):
        nonlocal prev
        r = np.zeros(len(tgt))
        for i in stim: r[pos[i]] = r_hz
        pg.rates = r * Hz
        neu.v = p['v_0']; neu.g = 0 * mV
        net.run(dur * ms)
        c = np.array(spk_mon.count[:], dtype=np.int64); e = c - prev; prev = c
        return e
    eA = episode(A, hz); e_sil = episode([], 0)
    eB = episode(B, hz); episode([], 0); eA2 = episode(A, hz)
    kA = set(np.flatnonzero(eA[g['kc']] >= 2).tolist())
    kB = set(np.flatnonzero(eB[g['kc']] >= 2).tolist())
    kA2 = set(np.flatnonzero(eA2[g['kc']] >= 2).tolist())
    print(f'{label:22s} sustained={int((e_sil>0).sum()):4d} |A|={len(kA):4d} |B|={len(kB):4d} '
          f'jacc={len(kA & kB)/max(len(kA | kB),1):.2f} rel={len(kA & kA2)/max(len(kA | kA2),1):.2f} '
          f'MBONspkA={int(eA[g["mbon"]].sum()):5d} MBONspkB={int(eB[g["mbon"]].sum()):5d}', flush=True)

run_stage('eln0 alln1 apl1 h100', 1, 1, 100, eln0=True)
run_stage('eln0 alln1 apl1 h150', 1, 1, 150, eln0=True)
run_stage('eln0 alln1 apl1 h250', 1, 1, 250, eln0=True)
run_stage('eln0 alln1 apl2 h150', 1, 2, 150, eln0=True)
