# PG-197 risk-aware XXL decoder and dual evaluator

device=cuda; raw_holdout_unsafe=8; gated_holdout_unsafe=0; DOM agreement=3/3; SQL agreement=6/6

| lane | runs | agreement/effect | claim allowed |
|---|---:|---:|---:|
| Pikachu DOM dual oracle | 3 | 3 | false |
| Pikachu unknown SQL/POST | 6 | 6 abstain | false |
| SQL v4/v5 source pair | 6 | 6 | false |

The raw decoder remains diagnostic; learned candidate gate and cross-source evaluator agreement are required before any candidate send.
