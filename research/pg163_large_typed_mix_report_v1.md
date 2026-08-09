# PG-163 大模型 typed mix 实验

- fresh typed episodes: **2388**（train 1224 / holdout 1164）
- mixed train sequences: **9999**；typed holdout: **203**
- device: **cuda**；vocabulary: **233**
- best typed holdout: **large_typed_mix**

模型输入只含抽象 Rule-IR token；原始 probe、响应正文、目标身份、族名和 oracle 标签均不进 token。结果仍是表示学习，不是漏洞扫描认证。
