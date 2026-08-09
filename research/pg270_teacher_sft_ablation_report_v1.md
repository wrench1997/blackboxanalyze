# PG-270 教师指导 SFT + preference/process-reward 小试验

设备：`cuda`；SFT=40；preference=40；family holdout=5。

| variant | route dev next-action | family holdout next-action | family preference win |
|---|---:|---:|---:|
| plain SFT | 0.800 | 0.800 | 1.000 |
| guided SFT | 0.800 | 0.800 | 1.000 |

教师 reference 只作为 target/label；原始 payload、响应正文和 oracle 字段不在模型 context。结果只代表训练信号消融，不能外推为可独立扫描公网的能力。

能力门：`passed`；training promotion=false；memory promotion=false。
数据集：`research/pg270_teacher_sft_dataset_v1.json`
报告：`research/pg270_teacher_sft_ablation_report_v1.json`
协议：`research/pg270_teacher_sft_ablation_protocol_v1.json`
