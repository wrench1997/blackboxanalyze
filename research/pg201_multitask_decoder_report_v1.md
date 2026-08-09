# PG-201 multitask decoder

device=cuda; base parameters=101487169; total parameters=101528207
train=15; replay=15; source-heldout=42
holdout unsafe allow=0; replay unsafe allow=0; forgetting=False

The adapter trains action, encoding and failure heads jointly while the XXL body remains frozen. No raw payload/response material enters the model or artifact.
