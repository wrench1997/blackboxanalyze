# PG-228 grounded diagnoser training

device=cuda; total=560; train=164; holdout=396
new PG-224=88; PG-226 typed SQL/result=8; PG-227 DOM/redirect=14; self-error counterfactuals=110
selected hidden=64; guarded holdout accuracy=1.0; next-step accuracy=1.0; guarded positive false accepts=0

只有 PG-226 同时通过 typed SQL effect 和 result fixture 的行标记为 payload-grounded eligible；PG-224 projection、PG-227 DOM/redirect surface 和 self-error 对照都不产生漏洞阳性或长期记忆。
