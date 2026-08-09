# PG-44 Rule IR decoder

effect gate 与 family decoder 分离；未知 semantic ref 一律 `unknown_surface + abstain`。

| split | typed recall | precision | FPR | unknown abstain |
|---|---:|---:|---:|---:|
| pg40_seed_holdout | 1.00 | 1.00 | 0.00 | True |
| pg42_train_role_diagnostic | 0.62 | 1.00 | 0.00 | True |
| pg42_dev | 0.62 | 1.00 | 0.00 | True |
| pg42_implementation_holdout | 0.62 | 1.00 | 0.00 | True |
| pg42_family_holdout | 0.00 | 0.00 | 0.00 | True |
| pg42_negative_control | 0.00 | 0.00 | 0.00 | True |

正式 capability claim=false；该候选只证明已知 ontology source transfer 与未知安全 abstain。
