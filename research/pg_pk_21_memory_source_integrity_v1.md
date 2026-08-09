# PG-PK-21 memory source integrity 实验

状态：`pass`；通过：7/7。

修复前，同一 source hash 通过 dataset 标签重命名即可满足多数据集门；同一 evidence hash 通过 seed 标签复制也会被当成独立采样。修复后，source hash、独立 seed evidence manifest 和去重后的证据计数共同决定晋升。PG-PK-15/18 的真实三 source ledger 仍通过；PG-PK-04 因缺少可重放 ledger 保持 diagnostic_only。
