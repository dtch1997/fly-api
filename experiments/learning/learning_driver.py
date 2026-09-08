"""Associative conditioning + generalization in the whole-brain LIF connectome.

Adds dopamine-gated LTD at KC->MBON synapses to the (otherwise fixed-weight)
Shiu et al. 2024 model, runs an appetitive conditioning protocol, and probes
generalization over stimulus similarity.

Rule (episodic, presynaptic, biologically standard in sign):
  after each episode where the US was delivered and PAM DANs fired above a
  gate, every KC->MBON synapse whose KC was active that episode is depressed:
  w *= (1 - eta).  Tests/probes deliver no US -> no plasticity.

Protocol (one seed):
  pre-tests:   A, B
  training:    n_pair x [A+US, B alone]
  post-tests:  A, B
  probes:      stimuli sharing {75,50,25,0}% of A's PNs (0% = novel, != B)

Outputs episodes.jsonl (one row per episode: per-MBON-type spike counts,
KC actives, PAM rate, plastic-weight mass) into --out.

Usage:
  python learning_driver.py --model-dir <shiu clone> --out runs/p1 \
      [--seed 0] [--eta 0.25] [--n-pair 3] [--drive-dans] [--epi-sec 1.0]
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-dir', required=True)
    ap.add_argument('--out', default='runs/p1')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--eta', type=float, default=0.25)
    ap.add_argument('--n-pair', type=int, default=3)
    ap.add_argument('--n-pn', type=int, default=25, help='PNs per odor')
    ap.add_argument('--pn-hz', type=float, default=100.0)
    ap.add_argument('--us-hz', type=float, default=150.0)
    ap.add_argument('--dan-hz', type=float, default=60.0)
    ap.add_argument('--epi-sec', type=float, default=1.0)
    ap.add_argument('--kc-thresh', type=int, default=3, help='spikes for a KC to count as active')
    ap.add_argument('--pam-gate-hz', type=float, default=1.0, help='mean PAM rate to gate plasticity')
    ap.add_argument('--drive-dans', action='store_true',
                    help='US drives PAM DANs directly instead of sugar GRNs')
    ap.add_argument('--overlaps', default='75,50,25,0')
    args = ap.parse_args()

    here = Path(__file__).resolve().parent
    sys.path.insert(0, args.model_dir)
    from model import create_model, default_params
    from brian2 import PoissonGroup, Synapses, Network, Hz, ms, mV, defaultclock

    rng = np.random.default_rng(args.seed)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    log = open(out / 'episodes.jsonl', 'w')

    # ---------------- circuit ids ----------------
    circ = json.load(open(here / 'circuit_ids.json'))
    sugar_ids = json.load(open(here / 'sugar_ids_783.json'))
    md = Path(args.model_dir)
    path_comp = md / 'Completeness_783.csv'
    path_con = md / 'Connectivity_783.parquet'
    df_comp = pd.read_csv(path_comp, index_col=0)
    flyid2i = {fid: i for i, fid in enumerate(df_comp.index)}

    def idx(ids):
        return np.array(sorted(flyid2i[i] for i in ids if i in flyid2i), dtype=int)

    kc_idx = idx(circ['kc'])
    pam_idx = idx(circ['pam'])
    mbon_groups = {t: idx(v) for t, v in circ['mbon'].items()}
    mbon_idx = np.unique(np.concatenate(list(mbon_groups.values())))
    pns = [i for i in circ['alpn_uni_chol'] if i in flyid2i]
    sugar_idx = idx(sugar_ids)

    # ---------------- stimuli ----------------
    pns = list(rng.permutation(pns))
    n = args.n_pn
    A = pns[:n]
    B = pns[n:2 * n]
    fresh = pns[2 * n:]  # pool for probe fill-ins, disjoint from A and B
    overlaps = [int(x) for x in args.overlaps.split(',')]
    probes = {}
    fi = 0
    for ov in overlaps:
        k = round(n * ov / 100)
        probes[f'probe{ov}'] = A[:k] + fresh[fi:fi + (n - k)]
        fi += n - k
    stim_sets = {'A': A, 'B': B, **probes}

    # ---------------- build network once ----------------
    print('[build] constructing whole-brain net...', flush=True)
    t0 = time.time()
    params = dict(default_params)
    neu, syn, spk_mon = create_model(str(path_comp), str(path_con), params)

    # stimulus delivery: one PoissonGroup unit per drivable neuron
    drivable = sorted(set().union(*stim_sets.values()) | set(sugar_ids) | set(circ['pam']))
    drive_pos = {fid: k for k, fid in enumerate(drivable)}
    drive_tgt = np.array([flyid2i[f] for f in drivable], dtype=int)
    pg = PoissonGroup(len(drivable), rates=0 * Hz)
    drv = Synapses(pg, neu, on_pre='v_post += w_drv', namespace={
        'w_drv': params['w_syn'] * params['f_poi']})
    drv.connect(i=np.arange(len(drivable)), j=drive_tgt)
    neu.rfc[drive_tgt] = 0 * ms  # mirror poi(): driven neurons lose refractory

    net = Network(neu, syn, spk_mon, pg, drv)

    # plastic synapse positions: KC -> MBON entries of the connection table
    df_con = pd.read_parquet(path_con)
    pre = df_con['Presynaptic_Index'].values
    post = df_con['Postsynaptic_Index'].values
    kc_set = set(kc_idx.tolist()); mbon_set = set(mbon_idx.tolist())
    plastic_mask = np.fromiter(((p in kc_set) and (q in mbon_set)
                               for p, q in zip(pre, post)), bool, len(pre))
    plastic_pos = np.flatnonzero(plastic_mask)
    plastic_pre = pre[plastic_pos]
    print(f'[build] done in {time.time()-t0:.0f}s; '
          f'{len(plastic_pos)} plastic KC->MBON synapses '
          f'({len(kc_idx)} KCs, {len(mbon_idx)} MBONs)', flush=True)

    w0 = np.array(syn.w[plastic_pos])  # initial plastic weights (volt)

    # ---------------- episode machinery ----------------
    prev_counts = np.zeros(len(df_comp), dtype=np.int64)

    def run_episode(name, stim, us):
        nonlocal prev_counts
        rates = np.zeros(len(drivable))
        for f in stim_sets[stim]:
            rates[drive_pos[f]] = args.pn_hz
        if us:
            if args.drive_dans:
                for f in circ['pam']:
                    if f in drive_pos:
                        rates[drive_pos[f]] = args.dan_hz
            else:
                for f in sugar_ids:
                    rates[drive_pos[f]] = args.us_hz
        pg.rates = rates * Hz
        neu.v = params['v_0']; neu.g = 0 * mV
        t1 = time.time()
        net.run(args.epi_sec * 1000 * ms)
        counts = np.array(spk_mon.count[:], dtype=np.int64)
        epi = counts - prev_counts
        prev_counts = counts

        kc_active = np.flatnonzero(epi[kc_idx] >= args.kc_thresh)
        kc_active_idx = kc_idx[kc_active]
        pam_hz = float(epi[pam_idx].mean() / args.epi_sec)
        plast_applied = False
        if us and pam_hz >= args.pam_gate_hz:
            hot = np.isin(plastic_pre, kc_active_idx)
            w = np.array(syn.w[plastic_pos])
            w[hot] *= (1 - args.eta)
            syn.w[plastic_pos] = w
            plast_applied = True

        row = {
            'episode': name, 'stimulus': stim, 'us': us,
            'plasticity_applied': plast_applied,
            'kc_active': int(len(kc_active)),
            'pam_hz': round(pam_hz, 3),
            'mbon_total_hz': round(float(epi[mbon_idx].sum() / args.epi_sec), 2),
            'mbon_by_type': {t: round(float(epi[g].sum() / args.epi_sec), 2)
                             for t, g in mbon_groups.items()},
            'plastic_w_frac': round(float(np.array(syn.w[plastic_pos]).sum() / w0.sum()), 4),
            'wall_s': round(time.time() - t1, 1),
        }
        log.write(json.dumps(row) + '\n'); log.flush()
        print(f"[epi] {name:14s} stim={stim:8s} us={int(us)} "
              f"KCact={row['kc_active']:4d} PAM={row['pam_hz']:6.2f}Hz "
              f"MBON={row['mbon_total_hz']:8.1f}Hz plast={int(plast_applied)} "
              f"wfrac={row['plastic_w_frac']:.3f} ({row['wall_s']}s)", flush=True)
        return row

    # ---------------- protocol ----------------
    meta = {'args': vars(args), 'A': A, 'B': B,
            'probes': {k: v for k, v in probes.items()},
            'n_plastic': int(len(plastic_pos))}
    (out / 'meta.json').write_text(json.dumps(meta))

    for s in ('A', 'B'):
        run_episode(f'pre_{s}', s, us=False)
    for c in range(args.n_pair):
        run_episode(f'train{c}_A+US', 'A', us=True)
        run_episode(f'train{c}_B', 'B', us=False)
    for s in ('A', 'B'):
        run_episode(f'post_{s}', s, us=False)
    for ov in overlaps:
        run_episode(f'probe_{ov}', f'probe{ov}', us=False)
    log.close()
    print('[done]', out / 'episodes.jsonl')


if __name__ == '__main__':
    main()
