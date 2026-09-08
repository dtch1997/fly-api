"""Track A: perception->motor in the whole-brain LIF connectome model.

Runs the Shiu et al. 2024 Brian2 model (weights = FlyWire synapse counts,
sign = predicted NT) and demos:
  A1  sugar GRN activation -> MN9 (proboscis extension motor neuron) fires
  A2  specificity: bitter/water GRNs at same rate -> MN9 stays quiet;
      sugar+bitter co-activation suppresses MN9
  A3  dose-response: MN9 rate rises monotonically with sugar drive

Usage: python track_a_lif.py --model-dir <clone of philshiu/Drosophila_brain_model>
Writes results.jsonl (one row per condition x tracked neuron) into --out.
"""
import argparse, json, sys, time
from pathlib import Path

# ---- neuron IDs (FlyWire v630), from Shiu et al. example.ipynb / figures.ipynb
SUGAR = [  # labellar sugar GRNs, right hemisphere
    720575940624963786, 720575940630233916, 720575940637568838, 720575940638202345, 720575940617000768,
    720575940630797113, 720575940632889389, 720575940621754367, 720575940621502051, 720575940640649691,
    720575940639332736, 720575940616885538, 720575940639198653, 720575940620900446, 720575940617937543,
    720575940632425919, 720575940633143833, 720575940612670570, 720575940628853239, 720575940629176663,
    720575940611875570]
BITTER = [
    720575940621778381, 720575940602353632, 720575940617094208, 720575940619197093, 720575940626287336,
    720575940618600651, 720575940627692048, 720575940630195909, 720575940646212996, 720575940610483162,
    720575940645743412, 720575940627578156, 720575940622298631, 720575940621008895, 720575940629146711,
    720575940610259370, 720575940610481370, 720575940619028208, 720575940614281266, 720575940613061118,
    720575940604027168]
WATER = [
    720575940612950568, 720575940631898285, 720575940606002609, 720575940612579053, 720575940622902535,
    720575940616177458, 720575940660292225, 720575940622486922, 720575940613786774, 720575940629852866,
    720575940625861168, 720575940613996959, 720575940617857694, 720575940644965399, 720575940625203504,
    720575940630553415, 720575940635172191, 720575940634796536]
MN9_L, MN9_R = 720575940660219265, 720575940645521262  # proboscis extension motor neurons

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-dir', required=True)
    ap.add_argument('--out', default='runs/track_a')
    ap.add_argument('--n-run', type=int, default=10, help='trials per condition (paper: 30)')
    ap.add_argument('--n-proc', type=int, default=30)
    args = ap.parse_args()

    sys.path.insert(0, args.model_dir)
    from model import run_exp, default_params
    import utils as utl
    from brian2 import Hz

    md = Path(args.model_dir)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    config = {
        'path_res':  str(out),
        'path_comp': str(md / '2023_03_23_completeness_630_final.csv'),
        'path_con':  str(md / '2023_03_23_connectivity_630_final.parquet'),
        'n_proc':    args.n_proc,
    }

    # condition table: (name, neu_exc, rate_Hz, neu_exc2, rate2_Hz)
    conds = [(f'sugar_{f}Hz', SUGAR, f, None, 0) for f in (25, 50, 100, 150, 200)]
    conds += [
        ('bitter_150Hz',       BITTER, 150, None,   0),
        ('water_150Hz',        WATER,  150, None,   0),
        ('sugar150_bitter150', SUGAR,  150, BITTER, 150),
    ]

    for name, exc, r, exc2, r2 in conds:
        p = dict(default_params)
        p['n_run'] = args.n_run
        p['r_poi'] = r * Hz
        p['r_poi2'] = r2 * Hz
        t0 = time.time()
        kw = dict(neu_exc2=exc2) if exc2 else {}
        run_exp(exp_name=name, neu_exc=exc, params=p, **kw, **config)
        print(f'[track_a] {name} done in {time.time()-t0:.0f}s', flush=True)

    # ---- collect rates
    tracked = {MN9_L: 'MN9_L', MN9_R: 'MN9_R'}
    rows = []
    ps = [str(out / f'{name}.parquet') for name, *_ in conds]
    df_spike = utl.load_exps(ps)
    df_rate, df_std = utl.get_rate(df_spike, t_run=1.0, n_run=args.n_run)
    for name, *_ in conds:
        col = df_rate[name] if name in df_rate else None
        for fid, label in tracked.items():
            rate = float(col.get(fid, 0.0)) if col is not None else 0.0
            std = float(df_std[name].get(fid, 0.0)) if name in df_std else 0.0
            rows.append({'condition': name, 'neuron': label, 'flyid': fid,
                         'rate_hz': round(rate, 2), 'std_hz': round(std, 2)})
        # also record total active neurons as a sanity metric
        rows.append({'condition': name, 'neuron': 'n_active_neurons', 'flyid': None,
                     'rate_hz': int((col.fillna(0) > 0).sum()) if col is not None else 0, 'std_hz': None})
    with open('results.jsonl', 'a') as f:
        for row in rows:
            f.write(json.dumps(row) + '\n')
    print(json.dumps(rows, indent=2))

if __name__ == '__main__':
    main()
