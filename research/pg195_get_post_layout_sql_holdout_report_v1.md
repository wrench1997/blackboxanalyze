# PG-195 GET/POST layout and SQL source holdout

device=cuda; surfaces=6; containers=3; GET=36; POST=18; DOM effects=3; Pikachu positives=0; SQL v4 typed=6

| lane | instances | typed effect/positive | claim allowed |
|---|---:|---:|---:|
| Pikachu GET/POST | 18 | 3 DOM effects | false |
| SQL v4 independent | 6 method runs | 6 typed | false |

The action model sees projections and failure signatures; route names and raw probe/response values are not persisted. Pikachu SQL surfaces remain unknown and abstain.
