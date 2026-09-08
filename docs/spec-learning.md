# Spec: in-silico associative learning + generalization

*Drafted 2026-09-08 from Daniel's question: can we teach the fly that some
images/colors mean food and others don't — and does it generalize?*

## The scientific setup

The Shiu LIF model is **fixed-weight** — it cannot learn as-is. Real fly
learning lives in the mushroom body (MB): Kenyon cells (KCs) sparse-code
stimuli; dopaminergic neurons (DANs) carry reward; coincidence of KC
activity + dopamine depresses KC→MBON synapses, tilting the fly's
approach/avoid balance. All of these cell types are annotated in FlyWire.
MB-only *rate* models of this loop exist (Jiang & Litwin-Kumar 2021's
prediction-error model; the programmable MB model, 2022). **What appears
undone: conditioning inside the whole-brain connectome LIF model, a
generalization gradient measured there, and any embodied version.** That's
the novelty target; everything else we reuse.

## Phases

**P0 — circuit access (feasibility gate).** Pull FlyWire cell-type
annotations; locate KCs, MBONs, PAM/PPL DANs in the Shiu model's ID space.
Check whether sugar-GRN stimulation already drives PAM DANs through the
real wiring (if yes, the US is endogenous; if no, drive DANs directly —
DAN activation substitutes for the US in real flies too).

**P1 — olfactory conditioning (grounded baseline).** Add dopamine-gated
plasticity to exactly the KC→MBON synapses in Brian2 (eligibility = KC
spike; gate = DAN activity; effect = LTD). Train: stimulus A paired with
sugar, stimulus B unpaired. Test: MBON valence balance for A vs B.
Success = discrimination that matches the sign of real fly behavior.

**P2 — generalization gradient (the question).** Probe stimuli sharing a
controlled fraction of active input neurons with A (100/75/50/25/0%
overlap). Plot learned response vs similarity. Sparse KC coding predicts a
steep gradient — measurable, falsifiable, and comparable to real-fly
generalization data.

**P3 — visual/color CS.** Swap the CS to the visual pathway: pale vs
yellow ommatidia channels (R7/R8 → visual PNs → γd KCs; visual KC inputs
are mapped in FlyWire). Condition on color A vs color B, then test
generalization across intermediate hues. This is the literal version of
"certain colors mean food." Risk: the visual→KC pathway may be too thin in
the LIF regime; a null here is itself a documented finding.

**P4 — embodied (stretch).** flygym arena with colored objects; contact
with the CS+ object triggers sugar-GRN drive; test whether the fly's
approach preference shifts and transfers to unseen similar hues. First
brain-in-body learning demo if it works.

## Cost & risks

All CPU: single trials are seconds; a conditioning protocol is a few
hundred 1-s simulations (~hours at Track-A throughput). Risks: annotation
wrangling (P0), whole-brain LIF may need DAN drive injected (accepted,
standard), visual pathway sparseness (P3), and the plasticity rule's free
parameters — mitigate by reusing published MB-model constants and
reporting sensitivity.

## Honesty note

The MB learning *mechanism* and MB-only models are prior art (see
[prior-art.md](prior-art.md)); the contribution here is running the
paradigm through the whole-brain connectome, the generalization gradient,
and (P4) embodiment.
