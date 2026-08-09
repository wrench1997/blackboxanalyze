# PG-231 feedback trajectory funnel

device=cuda; raw=478; unique=102; train=47; holdout=13; quarantine=42
lanes={'gold': 30, 'silver': 21, 'hard_negative': 9, 'quarantine': 42}; duplicates=376
selected hidden=64; holdout token accuracy=0.76244344; lane accuracy=1.0; repair accuracy=0.92307699; self-error recall=1.0

加入的是可观察过程状态，不是漏洞标签或原始 payload；分类头只读取 failure 位置之前的因果上下文。next-token loss 仍不能单独晋级。
