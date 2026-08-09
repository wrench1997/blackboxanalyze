# PG-271 教师候选 fresh seed 回放

fresh seed=27102；checkpoint=PG-270 guided_sft；模型评估设备=`cpu`。

| split | next-action | final belief | abstain calibration | unsupported positive |
|---|---:|---:|---:|---:|
| all 40 | 0.900 | 0.900 | 0.900 | 0 |
| family holdout | 0.800 | 0.800 | 0.800 | 0 |

fresh replay 的 typed oracle 仍是最终判定；模型只输出抽象候选动作，不生成或确认公网漏洞 payload。
capability gate: `passed`；promotion=false。
