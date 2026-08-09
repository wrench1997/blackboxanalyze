# Neural URL Generalization Loop 11 — Research Summary

Status: **confirmed research result** on a synthetic complete-family holdout. This is not a production vulnerability-detector claim.

## Question and constraint

Can a fixed-size neural model learn URL trust semantics from abstract primitives without seeing a single `url_scheme_downgrade` training example? The parameter budget stayed exactly **908,546**. The target family contributed **zero** training examples.

## Final preregistered confirmation

| Fresh seed | Candidate neural | Same checkpoint, set head off | Frozen neural | Gain vs frozen | C5 rule path | Worst old regression |
|---:|---:|---:|---:|---:|---:|---:|
| 20261603 | 88.83% | 60.00% | 63.83% | +25.00pp | 100.00% | -0.83pp |
| 20261621 | 88.00% | 58.83% | 62.67% | +25.33pp | 100.00% | +0.00pp |

The neural candidate averaged **88.42%**. Counterexample Top-1 averaged **88.00%**, versus a **37.50%** random baseline. Every preregistered threshold passed.

## Root cause and intervention

Diagnostics showed that the original byte Transformer relied on source-field and family shortcuts. It could recognize URL primitives but could not reliably bind the episode's positive and negative observations to a new query.

The confirmed intervention has three coupled parts:

1. Canonical URL slots serialize scheme, hostname, effective port and path without source-language field names.
2. Meta-label permutation is restricted to URL primitive families, forcing episode-consistent reasoning without perturbing unrelated tasks.
3. A learned **128-parameter set-comparison head** compares the query with positive and negative context examples. This is learned architecture, not an executable hand-written answer rule.

The strongest causal evidence is the same-checkpoint ablation: turning off only this head loses at least **28.83pp** on both fresh seeds.

## Why the rejected runs matter

- Canonical slots plus global meta-labeling reached only 54.33%; representation cleanup alone was insufficient.
- The first set-head candidate transferred strongly but was rejected because `truthiness_gate` regressed 3.83pp.
- Selective-meta v1 was rejected because `substring_origin` regressed 5.67pp.
- URL-meta v2 removed unrelated label perturbations and passed two new, untouched data seeds.

The thresholds were not relaxed after failure, and failed confirmation seeds were not reused as final evidence.

## Claim boundary

- **Neural result:** 88.42% mean on the unseen URL family, with a causal same-checkpoint ablation.
- **C5 result:** 100% from a separate zero-parameter executable rule architecture. It is not counted as neural learning.
- **Engineering result:** not established. Scaling now requires multiple languages, browser/runtime parsers, noisy black boxes, larger corpora and reliability profiling to distinguish scientific failures from implementation capacity limits.

