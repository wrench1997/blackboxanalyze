# PG-PK-19 triage 信号完整性实验

状态：`pass`；通过：5/5。

修复前，PG-PK-17 的领域化实验失败与缺失产物信号都会被静默过滤成 `inconclusive`。修复后，信号保留在调用者指定的路径；未注册信号能正确分类，但在映射进正式 taxonomy 前不会授权改模型或扩容。

该实验只修改 triage 信号传输与测试，没有修改模型、checkpoint、payload 或族特异 oracle。
