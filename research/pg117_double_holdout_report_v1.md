# PG-117 双重实现/编码保持出

PG-117 用独立 gamma 实现和 `[url_percent, html_entity]` 双重编码链，冻结 PG-116 checkpoint 进行族外评估。

- 目标实例/episode/step：`3/12/48`；GET/POST：`24/24`。
- 族外 route 正例最终召回：`0.0`；decoy 误接受：`0`；blind 弃权：`1.0`。
- 逐步准确率：`0.895833`；宏 F1：`0.633854`。
- 结论：这是评估失败/能力诊断，不生成训练样本，不提升长期记忆。缺口指向未抽象的通用 transition-delta 特征，而不是继续记忆目标表面。
