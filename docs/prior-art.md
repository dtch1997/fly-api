# What existed vs. what we added

An honest ledger for the 2026-09-08 demo (`demo/`, the
[site](https://dtch1997.github.io/fly-api/), and the demo report). Short
version: **the science all existed; our contribution is reproduction,
integration, two upstream bug finds, and packaging.**

## Existed already (not ours)

| Piece | Source |
|---|---|
| The connectome itself — 139,255 neurons, 50M+ synapses, NT predictions | [FlyWire](https://flywire.ai) (Princeton/Allen + community, *Nature* 2024) |
| Whole-brain LIF model **and the taste findings** — model equations, parameters, Brian2 code, the sugar/bitter/water GRN and MN9 neuron IDs, and the original versions of the dose–response and bitter-suppression experiments | [Shiu et al. 2024](https://github.com/philshiu/Drosophila_brain_model) |
| The fly body, its controllers, and the visual-taxis task — MuJoCo model, CPG+rules walking, retina simulation, the moving-sphere pursuit demo | [NeuroMechFly v2 / flygym 1.2.1](https://neuromechfly.org) (EPFL) |
| The wider landscape we cite but did not run | [flybody](https://github.com/google-deepmind/mujoco_menagerie/blob/main/flybody/README.md), [flyvis](https://github.com/TuragaLab/flyvis), [FlyGM](https://arxiv.org/abs/2602.17997), [BrainTrace](https://github.com/chaobrain/fitting_drosophila_whole_brain_spiking_model) |

Demo A is a **qualitative reproduction** of Shiu et al.'s published results
(at 10 trials/condition vs their 30). Demo B **runs flygym's own example
controllers** — we wrote no new neuroscience and no new control algorithms.

## What we did (ours)

| Contribution | Kind |
|---|---|
| Independent reproduction of Shiu et al.'s headline circuit results, as one scripted, conditions-as-data sweep (`demo/track_a_lif.py` → `results.jsonl`) on a commodity CPU box | verification + tooling |
| **flygym bug find #1**: `flygym.examples.vision.__init__` hard-imports `torch`+`flyvis`, breaking every vision example for users without them; vendoring workaround in `demo/svt_*.py` | upstream bug (unreported as of 2026-09-08; not in the tracker) |
| **flygym bug find #2**: `VisualTaxis.__init__` accepts `arena` but never forwards it to the simulation — the camera binds to an arena that is never compiled (`KeyError: 'birdeye_cam'`); one-line fix in `demo/svt_taxis.py` | upstream bug (unreported as of 2026-09-08) |
| No-sudo headless rendering recipe: extract `libosmesa6` from the Ubuntu .deb into a local prefix (`demo/setup_env.sh`) | ops |
| The integration existence proof: both stacks installed, run, and reproducible from a single quickstart on one GPU-less machine | integration |
| The synthesis: the layer map of the field, and the "missing SDK seam" framing that defines this project | framing |

## What would actually be novel (the bar for future work)

FlyGM already closed connectome→body — with a **rate-based GNN trained by
imitation + PPO**. The genuinely open increments this repo aims at:

1. A **spiking (LIF) whole-brain** in the closed loop with a realistic body
   — nobody has published that.
2. The **SDK seam** itself: connectome-as-architecture + body-as-env behind
   one API, neuron models and trainers as plugins.
3. **Controlled training studies** — connectome vs degree-matched rewired
   controls, seeded ensembles — addressing the field's reported
   retraining instability.

Until one of those lands, this project's claims are: "we reproduced,
integrated, and packaged" — nothing more.
