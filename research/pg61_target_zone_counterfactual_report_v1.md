# PG-61 目标区学习：随机 GET/POST 反事实

模型输入仅为 pre-oracle surface projection + candidate action；typed oracle 在动作之后才生成。

| split | target success | action accuracy | entropy | negative false accept | unknown strict abstain |
|---|---:|---:|---:|---:|---|
| dev | 1.0 | 1.0 | 0.922406 | 0 | True |
| holdout | 1.0 | 1.0 | 0.954434 | 0 | True |

硬门：`passed`；formal capability claim=false；训练/长期记忆不晋升。
