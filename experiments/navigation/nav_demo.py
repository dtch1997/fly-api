"""Brain-in-the-loop navigation: the fly walks toward the odor it was
conditioned on.

Pipeline (one process):
  1. Build the stabilized olfactory-MB LIF brain (as in the learning demo),
     run appetitive conditioning on odor A, snapshot naive + trained
     KC->MBON weights.
  2. Embodied rollouts (flygym HybridTurningController, flat arena with
     two odor sources): every decision step, each antenna samples the
     local concentrations of odors A and B; the spiking brain converts
     each sniff into MBON output (avoidance drive); the fly steers toward
     the side with lower MBON valence.
  3. Same fly, same arena, two brains: naive weights vs trained weights.
     Videos + trajectories saved.

Usage:
  python nav_demo.py --model-dir <shiu clone> [--out runs/nav]
      [--rollout naive|trained|both] [--n-decisions 36] [--steer-sign 1]
      [--smoke]  (smoke: 6 decisions, no video)
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

ANN = '/tmp/claude-2038/-mnt-nw-home-d-tan-jarvis-monorepo-jarvis-os/7e623888-45f1-4fd0-9563-5048d259e964/scratchpad/flywire-ann/annotations.tsv'
SRC_A = np.array([14.0, 7.0])    # rewarded odor source
SRC_B = np.array([14.0, -7.0])   # control odor source
SIGMA = 9.0

def conc(pos, src):
    d = np.linalg.norm(np.asarray(pos[:2]) - src)
    return float(np.exp(-d * d / (2 * SIGMA * SIGMA)))


class Brain:
    """Stabilized olfactory-MB subnet with switchable KC->MBON weights."""

    def __init__(self, md, seed=0, gain=20.0, orn_hz=500.0, n_classes=6):
        sys.path.insert(0, md)
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'learning'))
        from model import default_params
        from model_ext import build_subnet
        from brian2 import PoissonGroup, Synapses, Network, Hz, ms, mV, volt
        self.br = dict(Hz=Hz, ms=ms, mV=mV, volt=volt)
        ann = pd.read_csv(ANN, sep='\t', low_memory=False)
        cc = ann.cell_class.fillna(''); sc = ann.super_class.fillna('')
        ct = ann.cell_type.fillna(''); ht = ann.hemibrain_type.fillna('')
        df_comp = pd.read_csv(f'{md}/Completeness_783.csv', index_col=0)
        flyid2i = {fid: i for i, fid in enumerate(df_comp.index)}
        def ids(mask):
            return [flyid2i[i] for i in ann.loc[mask, 'root_id'].astype(int) if i in flyid2i]
        orn_mask = sc.str.contains('sensory') & cc.str.contains('olfactory')
        groups = {'orn': ids(orn_mask), 'alln': ids(cc == 'ALLN'), 'alpn': ids(cc == 'ALPN'),
                  'kc': ids(cc.str.contains('Kenyon', case=False)), 'apl': ids(ct == 'APL'),
                  'mbon': ids(ct.str.startswith('MBON') | ht.str.startswith('MBON')),
                  'pam': ids(ct.str.startswith('PAM') | ht.str.startswith('PAM')),
                  'ppl1': ids(ct.str.startswith('PPL1') | ht.str.startswith('PPL1'))}
        keep = sorted(set(i for v in groups.values() for i in v))
        p, neu, syn, spk_mon, old2new = build_subnet(md, default_params, keep)
        g = {k: np.array(sorted(old2new[i] for i in v)) for k, v in groups.items()}
        i_arr = np.array(syn.i[:]); j_arr = np.array(syn.j[:])
        w = np.array(syn.w[:])
        w[np.isin(i_arr, np.concatenate([g['pam'], g['ppl1']]))] = 0
        w[np.isin(i_arr, g['kc']) & np.isin(j_arr, g['kc'])] = 0
        w[np.isin(j_arr, g['orn'])] = 0
        w[np.isin(i_arr, g['alln']) & (w > 0)] = 0
        kcmbon = np.isin(i_arr, g['kc']) & np.isin(j_arr, g['mbon'])
        w[kcmbon] *= gain
        syn.w[:] = w * volt
        self.plastic_pos = np.flatnonzero(kcmbon)
        self.plastic_pre = i_arr[self.plastic_pos]
        # odors
        orn = ann[orn_mask]
        counts = orn.cell_type.value_counts()
        types = sorted((t for t in counts.index if counts[t] >= 40), key=lambda t: -counts[t])
        rng = np.random.default_rng(seed)
        order = list(rng.permutation(3 * n_classes))
        A_types = [types[order[k]] for k in range(0, 2 * n_classes, 2)]
        B_types = [types[order[k]] for k in range(1, 2 * n_classes, 2)]
        by_type = {t: [old2new[flyid2i[i]] for i in orn.loc[orn.cell_type == t, 'root_id'].astype(int) if i in flyid2i] for t in types[:3 * n_classes]}
        self.A = [i for t in A_types for i in by_type[t]]
        self.B = [i for t in B_types for i in by_type[t]]
        drivable = sorted(set(self.A + self.B) | set(g['pam'].tolist()))
        self.pos = {i: k for k, i in enumerate(drivable)}
        tgt = np.array(drivable)
        pg = PoissonGroup(len(tgt), rates=0 * Hz)
        drv = Synapses(pg, neu, on_pre='v_post += w_drv',
                       namespace={'w_drv': p['w_syn'] * p['f_poi']})
        drv.connect(i=np.arange(len(tgt)), j=tgt)
        neu.rfc[tgt] = 0 * ms
        self.net = Network(neu, syn, spk_mon, pg, drv)
        self.p, self.neu, self.syn, self.spk, self.pg, self.g = p, neu, syn, spk_mon, pg, g
        self.tgt = tgt
        self.orn_hz = orn_hz
        self.prev = np.zeros(len(keep), dtype=np.int64)
        self.w_naive = np.array(syn.w[self.plastic_pos])

    def _episode(self, rates_map, dur_ms, us=False):
        Hz, ms, mV = self.br['Hz'], self.br['ms'], self.br['mV']
        r = np.zeros(len(self.tgt))
        for i, hz in rates_map.items():
            r[self.pos[i]] = hz
        if us:
            for i in self.g['pam']:
                r[self.pos[i]] = 60.0
        self.pg.rates = r * Hz
        self.neu.v = self.p['v_0']; self.neu.g = 0 * mV
        self.net.run(dur_ms * ms)
        c = np.array(self.spk.count[:], dtype=np.int64)
        e = c - self.prev; self.prev = c
        return e

    def train(self, n_pair=5, eta=0.5):
        volt = self.br['volt']
        for k in range(n_pair):
            e = self._episode({i: self.orn_hz for i in self.A}, 500, us=True)
            kc_act = self.g['kc'][np.flatnonzero(e[self.g['kc']] >= 1)]
            hot = np.isin(self.plastic_pre, kc_act)
            wp = np.array(self.syn.w[self.plastic_pos])
            wp[hot] *= (1 - eta)
            self.syn.w[self.plastic_pos] = wp * volt
            self._episode({i: self.orn_hz for i in self.B}, 500)
            print(f'[train] pairing {k}: {int(hot.sum())} synapses depressed', flush=True)
        self.w_trained = np.array(self.syn.w[self.plastic_pos])

    def set_weights(self, which):
        volt = self.br['volt']
        w = self.w_naive if which == 'naive' else self.w_trained
        self.syn.w[self.plastic_pos] = w * volt

    def valence(self, cA, cB, dur_ms=150):
        """MBON output (avoidance drive) for an odor mixture sniff."""
        rates = {i: min(cA, 1.0) * self.orn_hz for i in self.A}
        rates.update({i: min(cB, 1.0) * self.orn_hz for i in self.B})
        e = self._episode(rates, dur_ms)
        return float(e[self.g['mbon']].sum())


def rollout(brain, which, out, args):
    import matplotlib
    matplotlib.use('Agg')
    from flygym import Fly, Camera
    from flygym.arena import FlatTerrain
    from flygym.examples.locomotion import HybridTurningController

    brain.set_weights(which)
    arena = FlatTerrain()
    for name, src, rgba in [('srcA', SRC_A, (0.72, 0.48, 0.07, 1.0)),
                            ('srcB', SRC_B, (0.35, 0.35, 0.38, 1.0))]:
        b = arena.root_element.worldbody.add('body', name=name, pos=(src[0], src[1], 1.2))
        b.add('geom', type='sphere', size=(0.9,), rgba=rgba, contype=0, conaffinity=0)
    contact_sensor_placements = [f'{leg}{seg}' for leg in ['LF', 'LM', 'LH', 'RF', 'RM', 'RH']
                                 for seg in ['Tibia', 'Tarsus1', 'Tarsus2', 'Tarsus3', 'Tarsus4', 'Tarsus5']]
    np.random.seed(0)
    fly = Fly(enable_adhesion=True, contact_sensor_placements=contact_sensor_placements,
              spawn_pos=(0.0, 0.0, 0.2))
    cam = None
    if not args.smoke:
        cam = Camera(attachment_point=arena.root_element.worldbody, camera_name='birdeye_cam',
                     camera_parameters={'mode': 'fixed', 'pos': (9, 0, 32),
                                        'euler': (0, 0, 0), 'fovy': 50},
                     play_speed=0.5, window_size=(800, 608))
    sim = HybridTurningController(fly=fly, cameras=[cam] if cam else [], timestep=1e-4,
                                  seed=0, arena=arena)
    obs, _ = sim.reset(0)
    traj, log = [], []
    n_sub = int(args.decision_sec / sim.timestep)
    for d in range(args.n_decisions):
        pos = np.array(obs['fly'][0][:2])
        head = np.array(obs['fly_orientation'][:2])
        n = np.linalg.norm(head); head = head / n if n > 1e-6 else np.array([1.0, 0.0])
        lat = np.array([-head[1], head[0]])
        ant_L = pos + 1.5 * head + 1.2 * lat
        ant_R = pos + 1.5 * head - 1.2 * lat
        vL = brain.valence(conc(ant_L, SRC_A), conc(ant_L, SRC_B))
        vR = brain.valence(conc(ant_R, SRC_A), conc(ant_R, SRC_B))
        tot = vL + vR
        delta = args.steer_sign * args.steer_gain * (vL - vR) / max(tot, 200.0)
        delta = float(np.clip(delta, -0.45, 0.45))
        if tot < 80:  # signal lost (deep in the learned-suppression zone): slow down
            action = np.array([0.55, 0.55])
        else:
            action = np.array([1.0 + delta, 1.0 - delta])
        for _ in range(n_sub):
            obs, _, _, _, _ = sim.step(action)
            if cam is not None:
                sim.render()
        traj.append(pos.tolist())
        dA = float(np.linalg.norm(pos - SRC_A)); dB = float(np.linalg.norm(pos - SRC_B))
        log.append({'d': d, 'pos': pos.tolist(), 'vL': vL, 'vR': vR,
                    'delta': delta, 'dA': round(dA, 2), 'dB': round(dB, 2)})
        print(f'[{which}] dec {d:02d} pos=({pos[0]:6.2f},{pos[1]:6.2f}) '
              f'vL={vL:6.0f} vR={vR:6.0f} delta={delta:+.2f} dA={dA:5.1f} dB={dB:5.1f}', flush=True)
        if min(dA, dB) < 2.5:
            print(f'[{which}] reached a source — stopping', flush=True)
            break
    pos = np.array(obs['fly'][0][:2]); traj.append(pos.tolist())
    if cam is not None:
        cam.save_video(out / f'{which}.mp4')
    (out / f'{which}_log.jsonl').write_text('\n'.join(json.dumps(r) for r in log))
    np.save(out / f'{which}_traj.npy', np.array(traj))
    dA = float(np.linalg.norm(pos - SRC_A)); dB = float(np.linalg.norm(pos - SRC_B))
    print(f'[{which}] FINAL dA={dA:.1f} dB={dB:.1f}', flush=True)
    return dA, dB


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-dir', required=True)
    ap.add_argument('--out', default='runs/nav')
    ap.add_argument('--rollout', default='both', choices=['naive', 'trained', 'both'])
    ap.add_argument('--n-decisions', type=int, default=36)
    ap.add_argument('--decision-sec', type=float, default=0.15)
    ap.add_argument('--steer-sign', type=float, default=1.0)
    ap.add_argument('--steer-gain', type=float, default=1.2)
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    if args.smoke:
        args.n_decisions = min(args.n_decisions, 6)

    t0 = time.time()
    brain = Brain(args.model_dir)
    print(f'[brain] built in {time.time()-t0:.0f}s', flush=True)
    brain.train()
    results = {}
    for which in (['naive', 'trained'] if args.rollout == 'both' else [args.rollout]):
        results[which] = rollout(brain, which, out, args)
    (out / 'summary.json').write_text(json.dumps(
        {k: {'final_dA': v[0], 'final_dB': v[1]} for k, v in results.items()}))
    print('[done]', results)


if __name__ == '__main__':
    main()
