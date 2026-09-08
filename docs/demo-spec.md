# fly-connectome-demo — basic motor control + basic perception

Goal-mode demo for [goals/fly-connectome-embodiment](../../goals/fly-connectome-embodiment.md).
Requested by Daniel 2026-09-08: "a demo of basic motor control as well as
basic perception".

## Track A — perception → motor at the connectome level (Shiu LIF model)

Whole-brain leaky integrate-and-fire simulation, weights = FlyWire synapse
counts, sign = predicted neurotransmitter (Shiu et al. 2024,
github.com/philshiu/Drosophila_brain_model, Brian2).

- **A1 (perception→motor, positive):** activate sugar-sensing gustatory
  receptor neurons (GRNs) at physiological rates → expect feeding motor
  neurons (MN9 / proboscis extension circuit) to fire.
- **A2 (control, specificity):** activate bitter GRNs → expect MN9
  suppression / no activation.
- **A3 (dose–response):** sweep sugar GRN drive over ≥4 rates → MN9 firing
  rate should increase monotonically (qualitative repro of Shiu Fig. 2).

Artifacts: `results.jsonl` (per-condition firing rates of tracked neurons),
raster + rate figures, pass/fail vs published qualitative result.

## Track B — embodied motor control + perception (flygym / NeuroMechFly v2)

- **B1 (motor):** CPG-based walking controller on flat terrain; render
  video; log joint angles + ground contacts (gait diagram).
- **B2 (perception):** enable the retina camera; render what the fly sees
  (raw ommatidia activity + human-view side by side) while approaching a
  visual target; if cheap, run the built-in visual taxis example.

Artifacts: mp4 videos, gait figure, retina-view figure.

## Non-goals (this demo)

Training anything; brain↔body coupling (that's the next increment, noted in
the goal file); GPU work.

## Environment

CPU devbox, local uv venv in this dir (heavy deps: brian2, flygym/mujoco —
kept out of the workspace root venv). Large data (FlyWire connectivity
parquet/csv, ~previews) downloaded to `data/` (gitignored); provenance +
URLs recorded in report.md.
