# PG-162 数据集与模型容量实验

- fresh typed rows: **384**（GET 192 / POST 192）
- source groups: **pg116_alpha=128, pg116_beta=128, pg118_delta=128**
- classes: **{'confirmed_negative': 264, 'candidate': 72, 'confirmed_positive': 24, 'abstain': 24}**
- device: **cuda**
- best source-heldout candidate: **base**

所有输入均为抽象、脱敏的 Rule-IR projection；原始 probe、响应正文、族名与 evaluator 标签不进入模型。PG-146 Docker 真实靶场仍是 evaluation-only，因为本轮没有 typed oracle。