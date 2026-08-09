# PG-205 field token training and replay

device=cuda; selected=standard; base parameters=101487169; field token dim=31
train=75; augmentation=60; holdout=42; replay=15
fresh containers=2; route replays=10; candidate sends=2; GET=2; POST=0
unknown abstain=6; redirect chains=2; field faults=40; network on fault=0

Request/response field tokens are bounded structure only. Local candidate sends are non-destructive canaries; no vulnerability claim is made.
