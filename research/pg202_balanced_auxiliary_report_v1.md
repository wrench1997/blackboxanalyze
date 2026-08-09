# PG-202 balanced auxiliary training

device=cuda; base parameters=101487169; total parameters=101528207
base train=15; augmentation=80; source-heldout=42; replay=15
holdout action=1.0; encoding=0.2857143; failure=0.71428573; unsafe=0
replay action=1.0; replay unsafe=0; forgetting=False

Augmentation is abstract and bounded; it adds no raw payload or response content.
