"""Associative conditioning + generalization in the connectome's
olfactory->mushroom-body pathway (LIF, FlyWire weights).

Substrate = the stabilized subcircuit (see notes.md): ORNs+ALLNs+PNs+KCs+
APL+MBONs+DANs (8,991 neurons, 792k synapses), Shiu parameters, with four
structural edits (DAN fast-outputs=0, KC->KC=0, ORN afferents input-only,
eLN excitatory outputs=0). No gain tuning.

Plasticity: episodic dopamine-gated LTD at KC->MBON synapses (KC active in
episode AND PAM DANs firing above gate -> w *= 1-eta).

Odors = disjoint sets of 4 ORN glomerulus classes (count-balanced).
Generalization probes share 3/2/1/0 of CS+'s classes.

Usage: python learning_driver_mb.py --model-dir <shiu clone> --out runs/mb1
       [--seed 0] [--eta 0.3] [--n-pair 3] [--orn-hz 250] [--dan-hz 60]
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

ANN = '/tmp/claude-2038/-mnt-nw-home-d-tan-jarvis-monorepo-jarvis-os/7e623888-45f1-4fd0-9563-5048d259e964/scratchpad/flywire-ann/annotations.tsv'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-dir', required=True)
    ap.add_argument('--out', default='runs/mb1')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--eta', type=float, default=0.3)
    ap.add_argument('--n-pair', type=int, default=3)
    ap.add_argument('--orn-hz', type=float, default=250.0)
    ap.add_argument('--dan-hz', type=float, default=60.0)
    ap.add_argument('--epi-sec', type=float, default=0.5)
    ap.add_argument('--kc-thresh', type=int, default=2)
    ap.add_argument('--pam-gate-hz', type=float, default=1.0)
    ap.add_argument('--n-classes', type=int, default=4, help='ORN classes per odor')
    ap.add_argument('--kcmbon-gain', type=float, default=1.0,
                    help='efficacy correction on KC->MBON synapses (restores anatomical share)')
    args = ap.parse_args()

    here = Path(__file__).resolve().parent
    md = args.model_dir
    sys.path.insert(0, md); sys.path.insert(0, str(here))
    from model import default_params
    from model_ext import build_subnet
    from brian2 import PoissonGroup, Synapses, Network, Hz, ms, mV, volt

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    log = open(out / 'episodes.jsonl', 'w')

    # ---------------- circuit ----------------
    ann = pd.read_csv(ANN, sep='\t', low_memory=False)
    cc = ann.cell_class.fillna(''); sc = ann.super_class.fillna('')
    ct = ann.cell_type.fillna(''); ht = ann.hemibrain_type.fillna('')
    df_comp = pd.read_csv(f'{md}/Completeness_783.csv', index_col=0)
    flyid2i = {fid: i for i, fid in enumerate(df_comp.index)}

    def ids(mask):
        return [flyid2i[i] for i in ann.loc[mask, 'root_id'].astype(int) if i in flyid2i]

    orn_mask = sc.str.contains('sensory') & cc.str.contains('olfactory')
    groups = {
        'orn': ids(orn_mask), 'alln': ids(cc == 'ALLN'), 'alpn': ids(cc == 'ALPN'),
        'kc': ids(cc.str.contains('Kenyon', case=False)), 'apl': ids(ct == 'APL'),
        'mbon': ids(ct.str.startswith('MBON') | ht.str.startswith('MBON')),
        'pam': ids(ct.str.startswith('PAM') | ht.str.startswith('PAM')),
        'ppl1': ids(ct.str.startswith('PPL1') | ht.str.startswith('PPL1')),
    }
    keep = sorted(set(i for v in groups.values() for i in v))
    p, neu, syn, spk_mon, old2new = build_subnet(md, default_params, keep)
    g = {k: np.array(sorted(old2new[i] for i in v)) for k, v in groups.items()}
    dan_all = np.concatenate([g['pam'], g['ppl1']])

    # ---- structural edits (see notes.md) ----
    i_arr = np.array(syn.i[:]); j_arr = np.array(syn.j[:])
    w = np.array(syn.w[:])
    n0 = [0, 0, 0, 0]
    m = np.isin(i_arr, dan_all);                     w[m] = 0; n0[0] = m.sum()
    m = np.isin(i_arr, g['kc']) & np.isin(j_arr, g['kc']); w[m] = 0; n0[1] = m.sum()
    m = np.isin(j_arr, g['orn']);                    w[m] = 0; n0[2] = m.sum()
    m = np.isin(i_arr, g['alln']) & (w > 0);         w[m] = 0; n0[3] = m.sum()
    if args.kcmbon_gain != 1.0:
        m = np.isin(i_arr, g['kc']) & np.isin(j_arr, g['mbon'])
        w[m] *= args.kcmbon_gain
    syn.w[:] = w * volt
    print(f'[edit] zeroed: dan_out={n0[0]} kc_kc={n0[1]} orn_in={n0[2]} eln_out={n0[3]}', flush=True)

    # ---- odors: count-balanced disjoint 4-class sets ----
    orn = ann[orn_mask]
    counts = orn.cell_type.value_counts()
    types = sorted((t for t in counts.index if counts[t] >= 40),
                   key=lambda t: -counts[t])
    nc = args.n_classes
    n_needed = 2 * nc + nc  # A + B + fresh pool (nc for probes)
    rng = np.random.default_rng(args.seed)
    order = list(rng.permutation(min(len(types), n_needed)))
    A_types = [types[order[k]] for k in range(0, 2 * nc, 2)]  # alternate for balance
    B_types = [types[order[k]] for k in range(1, 2 * nc, 2)]
    fresh_types = [types[order[k]] for k in range(2 * nc, min(len(order), 2 * nc + nc))]
    by_type = {t: [old2new[flyid2i[i]] for i in orn.loc[orn.cell_type == t, 'root_id'].astype(int) if i in flyid2i] for t in types[:n_needed]}

    def odor(ts):
        return [i for t in ts for i in by_type[t]]

    probes = {}
    probe_ks = sorted({round(nc * f) for f in (0.75, 0.5, 0.25, 0.0)}, reverse=True)
    for k_shared in probe_ks:
        ts = A_types[:k_shared] + fresh_types[:nc - k_shared]
        probes[f'probe{k_shared}'] = odor(ts)
    stim_sets = {'A': odor(A_types), 'B': odor(B_types), **probes}
    print(f'[odors] A={A_types} B={B_types} fresh={fresh_types}', flush=True)

    # ---- drive ----
    drivable = sorted(set(i for v in stim_sets.values() for i in v) | set(g['pam'].tolist()))
    pos = {i: k for k, i in enumerate(drivable)}
    tgt = np.array(drivable)
    pg = PoissonGroup(len(tgt), rates=0 * Hz)
    drv = Synapses(pg, neu, on_pre='v_post += w_drv',
                   namespace={'w_drv': p['w_syn'] * p['f_poi']})
    drv.connect(i=np.arange(len(tgt)), j=tgt)
    neu.rfc[tgt] = 0 * ms
    net = Network(neu, syn, spk_mon, pg, drv)

    # ---- plastic KC->MBON positions ----
    plastic_pos = np.flatnonzero(np.isin(i_arr, g['kc']) & np.isin(j_arr, g['mbon']))
    plastic_pre = i_arr[plastic_pos]
    w0 = np.array(syn.w[plastic_pos])
    print(f'[build] {len(plastic_pos)} plastic KC->MBON synapses', flush=True)

    prev = np.zeros(len(keep), dtype=np.int64)

    def run_episode(name, stim, us):
        nonlocal prev
        rates = np.zeros(len(tgt))
        if stim:
            for i in stim_sets[stim]:
                rates[pos[i]] = args.orn_hz
        if us:
            for i in g['pam']:
                rates[pos[i]] = args.dan_hz
        pg.rates = rates * Hz
        neu.v = p['v_0']; neu.g = 0 * mV
        t1 = time.time()
        net.run(args.epi_sec * 1000 * ms)
        c = np.array(spk_mon.count[:], dtype=np.int64)
        epi = c - prev
        # washout
        pg.rates = np.zeros(len(tgt)) * Hz
        net.run(150 * ms)
        prev = np.array(spk_mon.count[:], dtype=np.int64)

        kc_active_idx = g['kc'][np.flatnonzero(epi[g['kc']] >= args.kc_thresh)]
        pam_hz = float(epi[g['pam']].mean() / args.epi_sec)
        applied = False
        if us and pam_hz >= args.pam_gate_hz:
            hot = np.isin(plastic_pre, kc_active_idx)
            wp = np.array(syn.w[plastic_pos])
            wp[hot] *= (1 - args.eta)
            syn.w[plastic_pos] = wp * volt
            applied = True
        wp_now = np.array(syn.w[plastic_pos])
        ep_mask = np.isin(plastic_pre, kc_active_idx)
        trace = 1.0 - float(wp_now[ep_mask].sum() / max(w0[ep_mask].sum(), 1e-12)) if ep_mask.any() else 0.0
        row = {
            'episode': name, 'stimulus': stim, 'us': us, 'plast': applied,
            'trace_depression': round(trace, 4),
            'kc_active': int(len(kc_active_idx)),
            'kc_ids': [int(x) for x in kc_active_idx],
            'pam_hz': round(pam_hz, 2),
            'mbon_spk': int(epi[g['mbon']].sum()),
            'mbon_vec': [int(x) for x in epi[g['mbon']]],
            'w_frac': round(float(np.array(syn.w[plastic_pos]).sum() / w0.sum()), 4),
            'wall_s': round(time.time() - t1, 1),
        }
        log.write(json.dumps(row) + '\n'); log.flush()
        print(f"[epi] {name:12s} KCact={row['kc_active']:4d} PAM={row['pam_hz']:6.1f} "
              f"MBONspk={row['mbon_spk']:5d} plast={int(applied)} wfrac={row['w_frac']:.3f} "
              f"trace={row['trace_depression']:.3f}", flush=True)
        return row

    meta = {'args': vars(args), 'A_types': A_types, 'B_types': B_types,
            'fresh_types': fresh_types, 'n_plastic': int(len(plastic_pos)), 'probe_ks': probe_ks, 'n_classes': nc}
    (out / 'meta.json').write_text(json.dumps(meta))

    for r in (0, 1):
        run_episode(f'pre_A{r}', 'A', False)
        run_episode(f'pre_B{r}', 'B', False)
    for c in range(args.n_pair):
        run_episode(f'train{c}_A+US', 'A', True)
        run_episode(f'train{c}_B', 'B', False)
    for r in (0, 1):
        run_episode(f'post_A{r}', 'A', False)
        run_episode(f'post_B{r}', 'B', False)
    for k in probe_ks:
        run_episode(f'probe_{k}', f'probe{k}', False)
    log.close()
    print('[done]', out / 'episodes.jsonl')


if __name__ == '__main__':
    main()
