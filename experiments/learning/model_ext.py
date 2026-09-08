"""Whole-brain LIF net builder with optional stabilization mechanisms.

Extends Shiu et al.'s create_model with:
  - SFA: spike-frequency adaptation (a += b_a on spike, decay tau_a)
  - STD: event-driven Tsodyks-Markram short-term depression on all synapses
        (dx/dt=(1-x)/tau_rec event-driven; on_pre: g += w*x; x -= U*x)

Returns (params, neu, syn, spk_mon). Weights syn.w keep the same semantics
as upstream (volt, sign from NT), so plasticity code is unchanged.
"""
from textwrap import dedent
import pandas as pd
from brian2 import NeuronGroup, Synapses, SpikeMonitor
from brian2 import mV, ms


def build_net(model_dir, default_params, sfa=None, std=None):
    params = dict(default_params)
    eqs_extra, rst_extra, v_terms = '', '', 'v_0 - v + g'
    if sfa:
        b_a, tau_a = sfa
        params['b_a'] = b_a * mV
        params['t_a'] = tau_a * ms
        eqs_extra += 'da/dt = -a / t_a : volt\n'
        rst_extra += '; a += b_a'
        v_terms = 'v_0 - v + g - a'
    params['eqs'] = dedent(f'''
        dv/dt = ({v_terms}) / t_mbr : volt (unless refractory)
        dg/dt = -g / tau               : volt (unless refractory)
        {eqs_extra}rfc                            : second
        ''')
    params['eq_rst'] = 'v = v_rst; w = 0; g = 0 * mV' + rst_extra

    path_comp = f'{model_dir}/Completeness_783.csv'
    path_con = f'{model_dir}/Connectivity_783.parquet'
    df_comp = pd.read_csv(path_comp, index_col=0)
    df_con = pd.read_parquet(path_con)

    neu = NeuronGroup(
        N=len(df_comp), model=params['eqs'], method='linear',
        threshold=params['eq_th'], reset=params['eq_rst'],
        refractory='rfc', name='default_neurons', namespace=params)
    neu.v = params['v_0']; neu.g = 0
    neu.rfc = params['t_rfc']

    if std:
        U, tau_rec = std
        params['U_std'] = U
        params['tau_rec'] = tau_rec * ms
        syn = Synapses(neu, neu,
                       model='w : volt\ndx/dt = (1 - x)/tau_rec : 1 (event-driven)',
                       on_pre='g += w * x; x -= U_std * x',
                       delay=params['t_dly'], namespace=params,
                       name='default_synapses')
    else:
        syn = Synapses(neu, neu, 'w : volt', on_pre='g += w',
                       delay=params['t_dly'], name='default_synapses')

    i_pre = df_con.loc[:, 'Presynaptic_Index'].values
    i_post = df_con.loc[:, 'Postsynaptic_Index'].values
    syn.connect(i=i_pre, j=i_post)
    syn.w = df_con.loc[:, 'Excitatory x Connectivity'].values * params['w_syn']
    if std:
        syn.x = 1.0
    spk_mon = SpikeMonitor(neu)
    return params, neu, syn, spk_mon


def build_subnet(model_dir, default_params, keep_idx, sfa=None, std=None):
    """Build the LIF net restricted to `keep_idx` (brian indices into the
    full Completeness_783 order). Returns (params, neu, syn, spk_mon,
    old2new) where old2new maps full-model indices to subnet indices."""
    import numpy as np
    from brian2 import NeuronGroup, Synapses, SpikeMonitor
    params = dict(default_params)
    path_comp = f'{model_dir}/Completeness_783.csv'
    path_con = f'{model_dir}/Connectivity_783.parquet'
    df_comp = pd.read_csv(path_comp, index_col=0)
    df_con = pd.read_parquet(path_con)
    keep_idx = np.asarray(sorted(set(int(i) for i in keep_idx)))
    old2new = {int(o): n for n, o in enumerate(keep_idx)}
    pre = df_con['Presynaptic_Index'].values
    post = df_con['Postsynaptic_Index'].values
    m = np.isin(pre, keep_idx) & np.isin(post, keep_idx)
    i_pre = np.array([old2new[i] for i in pre[m]])
    i_post = np.array([old2new[i] for i in post[m]])
    w_vals = df_con['Excitatory x Connectivity'].values[m]
    neu = NeuronGroup(N=len(keep_idx), model=params['eqs'], method='linear',
                      threshold=params['eq_th'], reset=params['eq_rst'],
                      refractory='rfc', name='sub_neurons', namespace=params)
    neu.v = params['v_0']; neu.g = 0
    neu.rfc = params['t_rfc']
    syn = Synapses(neu, neu, 'w : volt', on_pre='g += w',
                   delay=params['t_dly'], name='sub_synapses')
    syn.connect(i=i_pre, j=i_post)
    syn.w = w_vals * params['w_syn']
    spk_mon = SpikeMonitor(neu)
    print(f'[subnet] {len(keep_idx)} neurons, {len(i_pre)} synapses')
    return params, neu, syn, spk_mon, old2new
