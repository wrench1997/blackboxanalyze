# Pikachu PG-PK-04 多数据集/多采样记忆过滤

目标实例：3；数据集：3；每目标采样 seed：2；ledger 条目：126。

| candidate Rule key | status | datasets | targets | accepted | FP | reasons |
|---|---|---:|---:|---:|---:|---|
| `xss::reflected_get` | `promote` | 3 | 3 | 24 | 0 | none |
| `injection::sqli_str` | `quarantine` | 3 | 3 | 0 | 0 | pikachu-dataset-8766:insufficient_accepted_evidence,pikachu-dataset-8767:insufficient_accepted_evidence,pikachu-dataset-8768:insufficient_accepted_evidence |

晋级规则：三类授权数据集/目标实例、每个至少两个采样 seed、每个至少一条正证据、每个数据集误报率为 0 且证据哈希完整；否则长期记忆隔离并 abstain。
当前只保存 bounded projection 的结果摘要和 SHA-256 evidence；没有执行脚本、SQL 语法/延时、RCE、SSRF、XXE、上传或凭据提交。

完整 JSON：`research\pikachu_multidataset_memory_filter_v1.json`
Loop 规则：`research\loop_memory_promotion_rule_v1.json`
