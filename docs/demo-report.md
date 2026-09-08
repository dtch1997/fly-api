# fly-connectome-demo — basic motor control + basic perception

**TL;DR.** Both demos landed, entirely on the CPU devbox. (A) The whole-brain
139k-neuron FlyWire LIF model turns taste perception into the correct motor
command through 50M real synapses with **no training**: sugar-GRN input
drives the proboscis motor neuron MN9 dose-dependently (0→92 Hz across
25→200 Hz drive), bitter input leaves it silent, and bitter co-activation
suppresses the sugar response 82→5.8 Hz (93%) — a qualitative repro of
Shiu et al. 2024. (B) The embodied NeuroMechFly body walks and turns
(16.1 mm in 1.5 s), and follows a moving object in closed loop using its
retina: object in view 100% of decision steps, mean azimuthal deviation
0.25 (normalized).

## Motivation

Daniel asked (2026-09-08): can the FlyWire fly neural architecture be
deployed in a realistic simulation, and trained? This demo is the first
increment of goal `fly-connectome-embodiment`: prove out the two mature
layers (whole-brain connectome dynamics; embodied body simulation) on our
infra before wiring them together.

## Method

**Track A (connectome perception→motor).** Shiu et al. 2024 whole-brain
leaky integrate-and-fire model (Brian2): every FlyWire v630 neuron is one
LIF unit; synaptic weight = synapse count × 0.275 mV, sign = predicted
neurotransmitter. We Poisson-stimulate labellar gustatory receptor neurons
(21 sugar / 21 bitter / 18 water GRNs, right hemisphere) and read out MN9,
the proboscis-extension motor neuron — the fly's "eat" command. Conditions:
sugar at 25–200 Hz (dose–response), water/bitter at 150 Hz (specificity),
sugar+bitter co-activation (suppression). 10 trials × 1 s each; ~19 min
total on 8 cores, ~2.5 GB/worker.

**Track B (embodied).** flygym 1.2.1 (NeuroMechFly v2 line), MuJoCo,
headless OSMesa rendering. B1: hybrid CPG+rule-based controller — straight
walking, then a left and a right turn, on flat terrain. B2: closed-loop
visual taxis — 721-ommatidia retina activity → azimuthal object position →
descending steering drives to the walking controller, following a moving
sphere (120 decision steps at 20 Hz, 6 s simulated).

## Results

### A. Perception→motor through the connectome (no training)

![dose response](figures/a3_dose_response.png)

MN9 firing rises monotonically with sugar drive: 0 / 18.1 / 67.7 / 82.0 /
92.1 Hz at 25/50/100/150/200 Hz GRN stimulation (mean of 10 trials; MN9_R
follows the same curve at lower rate — right-hemisphere stimulation,
ipsilateral bias).

![specificity](figures/a2_specificity.png)

At matched 150 Hz drive: sugar 82.0 Hz, water 5.6 Hz, bitter 0 Hz,
sugar+bitter 5.8 Hz. Identity and sign of the taste channels are carried by
the wiring diagram alone; bitter recruits inhibition that vetoes feeding.
Network-level sanity: 409 neurons active under sugar vs 81 under bitter.

### B. Embodied motor control + perception

- **B1 walking/turning** (`runs/track_b/b1_walking.mp4`, frame:
  `figures/b1_frame.png`): stable tripod gait, 16.1 mm displacement in
  1.5 s (10.7 mm/s), straight→left→right per the descending drive schedule.
- **B2 visual taxis** (`runs/track_b/b2_taxis_with_retina.mp4`, frame:
  `figures/b2_frame.png`, retina insets show what each eye sees): the fly
  keeps the moving sphere in view 100% of steps, mean |azimuthal
  deviation| 0.252; steering is computed only from ommatidia brightness.

## Discussion

The two layers our SDK thesis needs are now proven on our boxes, cheaply:
whole-brain connectome dynamics are a ~2-min-per-condition CPU job, and the
embodied simulator runs headless with software rendering. The perception
and motor demos meet in the middle — Track A's brain emits motor-neuron
commands, Track B's body consumes descending drives — so the natural next
increment is the coupling: map LIF motor-neuron populations (e.g. MN9, leg
MNs) onto flygym actuators and sensory neurons onto its retina/
proprioception, i.e. a minimal FlyGM-style loop but with the spiking
connectome instead of a trained GNN. Training (flyvis-style differentiable
relaxation, or FlyGM-style IL+RL) rides on top and needs the bellhop GPU
lane.

## Reproducibility

- Track A: `.venv` (py3.11: brian2 2.9.0, joblib) + clone of
  `philshiu/Drosophila_brain_model` (FlyWire v630 connectivity parquet
  ships in-repo); `python track_a_lif.py --model-dir <clone> --n-run 10
  --n-proc 8`. Neuron IDs (sugar/bitter/water GRNs, MN9 L/R) from the Shiu
  repo notebooks.
- Track B: same venv (flygym 1.2.1 + networkx/scipy/opencv/torch-cpu);
  `. ./setup_env.sh && python track_b_embodied.py`. OSMesa without sudo:
  `apt-get download libosmesa6 libglapi-mesa`, `dpkg -x` into a prefix,
  `LD_LIBRARY_PATH` + `MUJOCO_GL=osmesa` (see `setup_env.sh`).
- Upstream flygym 1.2.1 bugs worked around by vendoring
  (`svt_arena.py`/`svt_taxis.py`): `flygym.examples.vision.__init__`
  hard-imports torch+flyvis that simple taxis doesn't need; `VisualTaxis`
  captures `arena` but never forwards it to the simulation (KeyError
  'birdeye_cam').
- Artifacts (videos, spike parquets, trajectories):
  `gs://alignment-team-general-storage/daniel/jarvis/experiments/fly-connectome-demo/`
- flygym 2.1.0 (py3.12 venv `.venv-flygym`) is installed but unused — v2
  docs lean on an unpublished `flygym_demo` package and drop the vision
  tutorials; revisit when wiring brain↔body.
