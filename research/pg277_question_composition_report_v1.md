# PG-277 疑问驱动组合泛化

| variant | pre-question | positive recall | negative reject | counterfactual flip | missing-safe |
|---|---:|---:|---:|---:|---:|
| coarse_process_sft | 1.000 | 0.000 | 1.000 | 0.000 | 1.000 |
| enriched_final_only_sft | 0.000 | 1.000 | 1.000 | 1.000 | 0.667 |
| enriched_process_sft | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| conservative_offline_update | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| dpo_preference_update | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

gate=`passed`；所有能力/记忆晋级仍冻结。
