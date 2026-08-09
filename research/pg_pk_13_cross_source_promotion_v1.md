# PG-PK-13 跨独立数据源表面迁移与 memory promotion

状态：`promote`；独立 source hash：3；样本：222；typed oracle pair：11。

promotion ledger：222 行（反事实：182，false positive：0）；共享路由直接放行 pair：0；abstain 后由 typed-oracle fallback 找回：11；source diversity gate：`pass`。

只有跨独立 source hash、不同 target/seed、双编码 pair 和 typed sink oracle 同时通过，Rule IR 才允许进入长期 memory；共享 head 不拥有正向 authority。
