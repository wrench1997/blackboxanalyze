# PG-278 多族缺失观测—失败修复策略

| policy | pre transition (min) | post transition (min) | pair flip (min) | missing-safe (min) |
|---|---:|---:|---:|---:|
| coarse_process_sft | 0.000 | 1.000 | 1.000 | 1.000 |
| final_only_sft | 0.000 | 1.000 | 1.000 | 0.000 |
| enriched_process_sft | 1.000 | 1.000 | 1.000 | 1.000 |
| conservative_offline_update | 1.000 | 1.000 | 1.000 | 1.000 |
| dpo_preference_update | 1.000 | 1.000 | 1.000 | 0.000 |

gate=`passed`；可声称范围仅为受控多族 slot binding，真实靶场/长期记忆晋级仍冻结。
