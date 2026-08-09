# PG-180 process action token model

device=cuda; rows=35; vocabulary=78; runs=18

| variant | parameters | mean test accuracy | mean abstain recall |
|---|---:|---:|---:|
| small | 433230 | 0.933333 | 0.833333 |
| medium | 2442574 | 0.933333 | 0.833333 |
| moe_large | 15912026 | 0.933333 | 0.833333 |

模型只预测 allow-listed 抽象动作；不含路径、漏洞族、原始 probe/response 或 oracle 权威。网络复放不在本轮执行，所有晋升门保持 false。
