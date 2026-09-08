# fly-api

**Deploy and train the fruit-fly connectome in realistic simulation.**

The [FlyWire](https://flywire.ai) whole-brain connectome (139,255 neurons,
~50M synapses, neurotransmitter-typed) is a complete wiring diagram of an
adult *Drosophila*. Mature open layers exist around it — whole-brain
leaky-integrate-and-fire dynamics ([Shiu et al. 2024](https://github.com/philshiu/Drosophila_brain_model)),
biomechanical fly bodies ([flybody](https://github.com/google-deepmind/mujoco_menagerie/blob/main/flybody/README.md),
[NeuroMechFly/flygym](https://neuromechfly.org)), differentiable
connectome-constrained training ([flyvis](https://github.com/TuragaLab/flyvis)),
and whole-graph RL controllers ([FlyGM](https://arxiv.org/abs/2602.17997)) —
but no connective SDK. fly-api aims to be that seam:
**load connectome → pick neuron model → attach body → train**.

## Progress so far (2026-09-08)

Demo increment: both foundation layers proven on a CPU-only devbox.

**1. Perception→motor through the connectome, zero training**
(`demo/track_a_lif.py`) — the whole-brain LIF model turns taste input into
the correct motor command:

| condition (150 Hz drive) | MN9 "eat" motor neuron |
|---|---|
| sugar GRNs | **82.0 Hz** |
| water GRNs | 5.6 Hz |
| bitter GRNs | 0 Hz |
| sugar + bitter | 5.8 Hz (93% veto) |

Dose–response is monotonic (0→92 Hz for 25→200 Hz sugar drive). See
`figures/a3_dose_response.png`, `figures/a2_specificity.png`.

**2. Embodied motor control + vision** (`demo/track_b_embodied.py`) —
NeuroMechFly body in MuJoCo: tripod-gait walking with commanded turns
(`media/b1_walking.mp4`), and closed-loop visual taxis steering only from
721-ommatidia retina readings (`media/b2_taxis_with_retina.mp4`, with
fly's-eye insets).

Full method + gotchas: [`docs/demo-report.md`](docs/demo-report.md).

## Quickstart

```bash
uv venv .venv --python 3.11
uv pip install --python .venv/bin/python brian2 flygym pandas pyarrow \
    matplotlib joblib networkx scipy opencv-python-headless "imageio[ffmpeg]"
# Track A: whole-brain LIF (connectivity data ships in the model repo clone)
git clone --depth 1 https://github.com/philshiu/Drosophila_brain_model
.venv/bin/python demo/track_a_lif.py --model-dir Drosophila_brain_model
# Track B: embodied demos (headless boxes: see demo/setup_env.sh for no-sudo OSMesa)
cd demo && . ./setup_env.sh && ../.venv/bin/python track_b_embodied.py
```

## Roadmap

1. **Brain↔body coupling** — map LIF motor-neuron populations onto flygym
   actuators, sensory neurons onto retina/proprioception (a spiking
   FlyGM-lite).
2. **The SDK seam** — one API over FlyWire-as-architecture +
   fly-body-as-env, with neuron models (rate / LIF / NODE) and training
   strategies (flyvis-style BPTT, FlyGM-style IL+RL) as plugins.
3. **Training runs** on GPU (connectome vs matched rewired/random controls,
   seeded ensembles — retraining instability is a known field problem).

## Provenance

Spun out of the JARVIS monorepo (`experiments/fly-connectome-demo`,
dtch1997/jarvis#191). Vendored `demo/svt_arena.py` / `demo/svt_taxis.py`
derive from [flygym](https://github.com/NeLy-EPFL/flygym) 1.2.1 examples
(Apache-2.0), with two fixes noted in the demo report.
