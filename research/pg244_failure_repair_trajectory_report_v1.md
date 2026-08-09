# PG-244 failure → diagnosis → repair → fresh replay

episodes=8; fresh=16; SQL=4; XSS=4; GET=8; POST=8
records=24; gold=16; hard_negative=8; replay=8; model_self_error=0
counterfactual failures are retained as hard negatives with explicit repair targets; raw wire is stdout-only.
