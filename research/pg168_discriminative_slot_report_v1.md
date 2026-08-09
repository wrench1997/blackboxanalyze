# PG-168 discriminative Rule-IR slots

- slot train/dev/OOD: **8000 / 1000 / 1000**
- replay-only OOD PPL: **471.30196493**
- slot-augmented OOD PPL: **2.38018788**
- dev/OOD projection overlap: **0**

该轮只验证抽象 slot 信息量与下一个 token 训练；不产生漏洞标签，不晋级长期记忆。

