# PG-196 redirect-chain and POST-failure action decoder

device=cuda; decoder_holdout=0.63425928; redirect_positive=3/3; post_failure=3; post_abstain=3

| lane | runs | typed effect | final action |
|---|---:|---:|---|
| Pikachu controlled redirect GET | 3 | 3 | typed candidate |
| Pikachu POST failure | 3 | 0 | abstain_unknown_oracle |

The decoder receives only bounded method/status/redirect/failure features; exact URLs, payload values and bodies are not persisted.
