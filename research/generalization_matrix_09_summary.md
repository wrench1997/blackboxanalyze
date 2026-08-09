# Generalization Matrix 09

Status: **partial success with a blocking URL-semantics failure**. This is a synthetic, local research pilot—not a production vulnerability-detector claim.

Across three preregistered seeds, the paired macro score moved from 66.12% to 71.67% (+5.55%). The only training-driven effect confirmed on all three seeds is numeric coercion. The 100% history result comes from the new representation and executable rule head, not neural learning.

| Held-out axis | Frozen baseline | Primitive-adapted | Paired delta | Conclusion |
|---|---:|---:|---:|---|
| URL runtime semantics | 53.61% | 43.00% | -10.61% | confirmed_negative_transfer |
| Encoding depth | 50.00% | 59.17% | +9.17% | high_variance_not_confirmed |
| Unicode / casefold | 78.72% | 80.33% | +1.61% | stable_small_gain_below_preregistered_effect |
| Numeric coercion | 51.39% | 81.78% | +30.39% | confirmed_training_generalization |
| Rule composition | 63.00% | 65.72% | +2.72% | stable_small_gain_below_preregistered_effect |
| State / history | 100.00% | 100.00% | +0.00% | confirmed_rule_architecture_fix_not_neural_learning |

Counterexample Top-1 averaged 78.15%, versus a 64.60% random Top-1 baseline. Every old-family regression check stayed within the preregistered -2pp bound.

## Root-cause decision

- Numeric string coercion: confirmed cross-family training generalization (+28pp or more in every seed).
- Double decoding: high variance; two gains and one chance result, so not confirmed.
- State/history: solved by abstraction and executable memory, while the neural path remains near chance.
- URL semantics: confirmed negative transfer. The next experiment must factor scheme, hostname and port, and prevent a raw suffix rule from overriding structured URL evidence.
- The earlier unstratified Stage-B run is preserved for audit but excluded from all aggregates.
