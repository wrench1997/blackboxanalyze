# PG-274 分数 + 约束离线 RL

v1 train=31；v2 holdout=36；device=`cuda`。

| variant | v2 next-action | v2 belief | positive recall | negative reject | false positives |
|---|---:|---:|---:|---:|---:|
| plain_sft | 0.889 | 0.889 | 0.333 | 1.000 | 0 |
| score_weighted_sft | 1.000 | 1.000 | 1.000 | 1.000 | 0 |
| score_rl | 0.889 | 0.889 | 0.333 | 1.000 | 0 |

gate=`blocked`；RL v2 positive recall=0.333。只有独立实现复测和误报门都通过，才有资格继续扩大 RL。
