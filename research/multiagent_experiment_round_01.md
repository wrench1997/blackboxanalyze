# 多代理实验轮次 01

本轮由三个受限 `luna_worker` 风格代理并行审计，全部使用本地离线数据；没有访问外网、外部目标或凭据，也没有改模型 checkpoint。

## 代理结论

- `luna_worker_generalization` 找到长期记忆门的真实绕过：同一 source 通过 dataset 标签重命名、同一 evidence 通过 seed 标签复制，旧门仍会晋升。
- `luna_worker_belief` 找到重复 evidence 放大 posterior、传输错误误绑定认证、派生 binding 可篡改，以及通用 active probe 偏向均匀无信息候选。
- `luna_worker_oracles` 找到顶层 `oracle_projection` 与哈希 evidence 没有一致性校验的缺口；现有离线 oracle 测试为 8 passed。

## 本轮已落地

- [PG-PK-19 triage 信号完整性](<D:\workspace\blackboxanalyze\research\pg_pk_19_triage_signal_integrity_v1.json>)：5/5。领域化失败信号不再被静默丢弃，未注册信号不会授权改模型或扩容。
- [PG-PK-21 memory source 完整性](<D:\workspace\blackboxanalyze\research\pg_pk_21_memory_source_integrity_v1.json>)：7/7。长期晋升现在要求 source hash、多 seed 独立 evidence manifest，并按唯一证据计数。
- [PG-PK-22 oracle projection 绑定](<D:\workspace\blackboxanalyze\research\pg_pk_22_oracle_projection_binding_v1.json>)：6/6。篡改顶层 projection 会被 XSS、SQL、logic/access 三类 revalidator 拒绝。

相关定向修复包括重复证据去重、传输失败保持 `unbound`、Rule IR 派生 binding 重算，以及把新门写入 `research/loop_memory_promotion_rule_v1.json`。

## 回归

`python -m pytest -q`：**175 passed, 1 warning**。PG‑17 联合回归功能门为 `pass`；`inconclusive` 只表示没有失败信号，仍禁止据此扩模或扩容。

## 下一轮

尚未改动通用 active-probe 的 entropy 规则，因为需要先预注册“候选结果分布/期望信息增益”的 outcome model。下一轮应专门测试冲突证据的显式 abstain、真正改变样本集合的 seed、以及 PG‑04 的 fresh ledger/source hash；旧 PG‑04 摘要缺少可重放 ledger，暂保持 `diagnostic_only`。
