# learning demo — running notes

## P0 (2026-09-08) — circuit access: PASS, with one finding

- Materialization: **783** (annotations root_ids match Completeness_783
  100%; the v630 sugar-GRN list loses 1/21 neurons to proofreading drift —
  we use the 20 survivors, saved in `sugar_ids_783.json`).
- Circuit coverage in the model: 5,177 KCs · 96 MBONs (35 types) ·
  307 PAM + 16 PPL1 DANs · 246 uniglomerular cholinergic AL PNs. All
  present.
- **Finding: sugar does not reach the mushroom body in the LIF regime.**
  20 sugar GRNs @150 Hz activate the feeding circuit (403 neurons, as in
  Track A) but PAM DANs, KCs, and MBONs stay at 0 Hz. The reward signal
  that gates learning does not emerge endogenously from gustatory drive in
  this model.
- Decision: deliver the US as **direct PAM-DAN drive** (`--drive-dans`),
  which is also what real conditioning experiments do when they substitute
  DAN optogenetic activation for sugar. Documented as a model limitation.

## Open risks

- Will 25 PNs @100 Hz recruit KCs? (KCs need convergent PN input; smoke
  run logs `kc_active`.) Tune n-pn / pn-hz if sparse or silent.
- Plasticity magnitude (eta=0.25 x 3 pairings) is a free parameter; report
  sensitivity, don't tune to a desired result.

## Finding (2026-09-08): global ignition attractor under central drive

The fixed-weight LIF model is **bistable**. From silence, driving just 10
AL projection neurons at 15 Hz for 500 ms ignites a self-sustaining global
state — ~8,400 neurons, ~480k spikes/s — that persists indefinitely after
all input stops (3+ s of silence, no decay). KC "coding" in this state is
~65% dense regardless of stimulus, which would make any learning
measurement meaningless. Zeroing DAN fast-outputs (dopamine as
neuromodulator) is correct but insufficient — the attractor rides the
global cholinergic recurrence.

Shiu et al. never hit this because their published experiments drive
peripheral sensory neurons (sugar: 403 active neurons, stable); the
attractor lives downstream of central (PN) drive. This is a real
characterization of the model's operating envelope, worth reporting.

**Remedy**: uniform spike-frequency adaptation (da/dt = -a/t_a; a += b_a
on spike) — one ubiquitous biophysical mechanism, two parameters.
Calibration criterion: smallest b_a that (i) kills the sustained state
after a PN kick, while (ii) preserving the published sugar→MN9 feeding
response as a regression test. No tuning toward the learning result.

## SFA calibration (2026-09-08): FAILED the trade-off

Spike-frequency adaptation cannot separate storm from signal: settings
that kill the sustained state (b=8mV/tau=300ms; b=16mV/tau=100ms) also
collapse the sugar regression (MN9 70Hz -> 4Hz, feeding circuit 361 ->
~200 active). Sub-critical settings (b<=8mV/tau=100ms) leave the attractor
alive. A rate-blind hyperpolarizing brake is the wrong tool.

Next: short-term synaptic depression (Tsodyks-Markram, event-driven, all
synapses) — punishes sustained high-rate transmission (what the storm is)
while sparing transient volleys (what the sensory responses are).
model_ext.py now builds the net with either mechanism.

## Dead ends logged (2026-09-08)

- **STD**: even U=0.1 collapses sugar->MN9 (70->4Hz) — the feeding chain
  itself relies on sustained high-rate relay. Fails regression.
- **E/I gain** (partial, run was stopped): i_gain=2 leaves storm alive AND
  breaks MN9. Axis unpromising.
- **Pulse-mode**: ignition latency ~0 — 6.3k neurons active in the first
  50ms bin, saturated by 100ms. No stable transient window under PN drive.

## Reframe: inject the CS at the periphery (ORNs), not at PNs

Sugar GRN drive (peripheral) is stable (403 neurons). Driving PNs directly
bypasses the antennal lobe's local-interneuron gain control — plausibly
why it ignites. Odors ARE ORN-class patterns; probe_orn.py drives 4
disjoint ORN glomerulus classes per odor at 100Hz and checks stability +
KC sparseness + A-vs-B code overlap.
