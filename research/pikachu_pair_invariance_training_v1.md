# Pikachu PG-PK-02 配对不变性训练

每个 split 比较无配对损失（baseline）和加入同 pair embedding consistency 的模型。输入只包含可观察的 action/probe/response shape/encoding descriptor；pair id、family、surface role 只在训练/评估 harness 中使用。

| split | model | exit | false positive | abstain | candidate consistency | embedding cosine |
|---|---|---:|---:|---:|---:|---:|
| encoding_holdout | baseline | 1.00 | 0.00 | 0.00 | 1.00 | 0.99 |
| encoding_holdout | pair_consistency | 1.00 | 0.00 | 0.00 | 1.00 | 1.00 |
| encoding_holdout | pair_encoding_invariant | 1.00 | 0.00 | 0.00 | 1.00 | 0.95 |
| encoding_holdout | pair_encoding_invariant_consensus | 1.00 | 0.00 | 0.00 | 1.00 | 0.95 |
| surface_holdout | baseline | 1.00 | 0.00 | 0.00 | 1.00 | 1.00 |
| surface_holdout | pair_consistency | 1.00 | 0.00 | 0.00 | 1.00 | 1.00 |
| surface_holdout | pair_encoding_invariant | 1.00 | 0.00 | 0.00 | 1.00 | 1.00 |
| surface_holdout | pair_encoding_invariant_consensus | 1.00 | 0.00 | 0.00 | 1.00 | 1.00 |
| joint_holdout | baseline | 0.67 | 0.33 | 0.00 | 0.33 | 0.59 |
| joint_holdout | pair_consistency | 0.67 | 0.33 | 0.00 | 0.33 | 0.71 |
| joint_holdout | pair_encoding_invariant | 0.67 | 0.33 | 0.00 | 0.33 | 0.86 |
| joint_holdout | pair_encoding_invariant_consensus | 0.33 | 0.00 | 0.67 | 0.33 | 0.86 |

解释：配对损失的目标是让同一抽象族的编码/表面变体在表示空间接近，而不是强迫模型对新表面硬猜。错误或高不确定性结果仍必须 abstain。

## 特征相关性审计

下表是 1-NN 的快速消融，不是最终模型分数；它用于发现明显的捷径或无关输入。

| split | full | no encoding | no probe | response only | no oracle shape |
|---|---:|---:|---:|---:|---:|
| encoding_holdout | 1.00 | 1.00 | 0.33 | 0.33 | 1.00 |
| joint_holdout | 0.83 | 0.83 | 0.67 | 0.67 | 0.83 |

完整 JSON：`research\pikachu_pair_invariance_training_v1.json`
