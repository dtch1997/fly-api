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
