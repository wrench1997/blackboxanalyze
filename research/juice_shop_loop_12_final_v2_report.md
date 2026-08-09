# Loop 12 阶段结果：多 hidden family shadow 矩阵 + Rule IR 解码器

状态：**跨种子稳定性已通过；多族矩阵完成但当前 response-only 策略没有命中；Rule IR 解码器已训练并通过最低泛化 gate，仍有明确缺口。**

## 1. 先解决稳定性

v3 response head 在 4 个新数据种子上达到平均 **99.9722%**，最低 **99.8889%**；旧回归集最坏回退 **0.0pp**，接受。

## 2. 多 hidden family 矩阵

冻结协议为 [hidden matrix v1.1](juice_shop_loop_12_hidden_matrix_protocol_v1_1.json)，覆盖 observability、access_control、authentication、input_validation、injection、url_redirect、xss。一个 shadow 容器探测 28 个唯一 action，随后每个策略各重置一个 fresh target，固定 family 顺序回放 7 个动作。

| 策略 | fresh target 动作 | hidden family hit |
|---|---:|---:|
| v3 response head | 7 | 0/7 |
| response-disabled ablation | 7 | 0/7 |

两者差值为 0：响应形状 head 尚未学会跨 family 的 action search。这个结果应标记为“搜索/接口能力缺口”，不能解释成漏洞不存在。运行过程中发现并修正的三项是工程问题：启动时暂态连接拒绝、`/ftp/` 截断 Content-Length、redirect 参数触发 origin-relative guard；它们均已保留在运行器/协议记录中。

## 3. Rule IR 解码器

训练产物：[rule_ir_decoder.pt](../artifacts/rule-ir-decoder-loop-12-20260829/rule_ir_decoder.pt)，报告：[report.json](../artifacts/rule-ir-decoder-loop-12-20260829/report.json)。模型用 256 维匿名 trace projection，在 RTX 3060 CUDA 上训练 7 类 policy-slot AST：

`access_control / authentication / input_validation / injection / observability / url_redirect / xss`

输入不包含 family、`intended_output`、`is_counterexample`、challenge key、源码或 evaluator state；输出先经过 `validate_abstract_rule_ir` grammar 校验，再作为语言无关的 Rule IR 模板保存。

- IID validation：**100%**
- 3 个 fresh holdout seeds：平均 **75.0%**，最低 **73.33%**，达到预注册最低 70% gate
- 置信度门控最大 abstain rate：**50%**
- 已知缺口：`dom_double_decode` holdout 为 **0%**，说明当前 encoding-depth 特征不能稳定表达双重解码

## 4. 回归验证

使用 workspace 根目录作为 `PYTHONPATH` 执行 `python -m pytest -q`：**50 passed，1 warning**。直接调用 pytest 可执行文件时的 `ModuleNotFoundError` 是环境路径问题，不是测试失败。

下一轮应增加 encoding-depth 专门 curriculum，并让抽象槽位在矩阵 probe 后再绑定具体证据，然后重新跑七族矩阵；在此之前不把当前结果宣称为完整漏洞泛化能力。
