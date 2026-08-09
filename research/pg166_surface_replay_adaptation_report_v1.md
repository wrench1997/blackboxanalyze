# PG-166 surface replay adaptation

- baseline base/typed PPL: **2.5717967 / 1.45351808**
- replay-anchored base/typed PPL: **2.55618482 / 1.43844417**
- surface-only base/typed PPL: **2.64894756 / 1.71369749**

该轮只比较 replay 与遗忘；表面 attestation 不作为漏洞标签，checkpoint 不晋级长期记忆。
