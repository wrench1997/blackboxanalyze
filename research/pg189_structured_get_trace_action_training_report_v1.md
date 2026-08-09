# PG-189 structured GET trace action training

device=cuda; train=225; dev=95; holdout=120; vocab=280; selected=large

| body | parameters | holdout accuracy | abstain recall | false candidate | forgetting |
|---|---:|---:|---:|---:|---|
| large | 19270940 | 1.0 | 1.0 | 0 | False |
| xxl | 101482780 | 1.0 | 1.0 | 0 | False |

结构化 real GET trace 只作为 safety-policy action 数据；不包含原始 payload、响应正文或漏洞阳性标签。
