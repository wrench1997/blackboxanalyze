# PG-225 enriched large problem diagnoser

device=cuda; total=428; train=172; holdout=256; real PG-224 rows=88
selected hidden=64; guarded holdout accuracy=1.0; guarded positive false accepts=0

PG-224 的真实回放只有 projection-only oracle，所以新行只能训练 oracle_unavailable/inconclusive，不得被当作漏洞阳性。PG-223 的 frozen XXL body 仍未解冻；训练的是诊断 adapter。
