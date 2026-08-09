# PG-194 evaluator-aware gate cross replay

device=cuda; gate_holdout_recall=1.0; dom_effect=3/3; sql_typed_positive=3/3; claims=0

| lane | instances | typed effect/positive | false positive |
|---|---:|---:|---:|
| Pikachu DOM | 3 | 3 | 0 |
| SQL fixture variants | 3 | 3 | 0 |

Evaluator-aware gate uses only typed availability, negative control, fresh reset, evidence hash and bounded effect state; no route or raw payload enters the gate.
