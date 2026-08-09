# Loop 12 v3：编码深度 + shadow 后 Rule IR slot binding

状态：**双重解码泛化已恢复，shadow 后证据绑定已接通；七族 evaluator 命中仍为 0，完整 action search 尚未解决。**

## 训练改动

- 加入 180 个 encoding-depth 样本：raw、single entity、double entity、triple entity、plain text。
- 匿名特征加入 0–3 次 unescape 后的 tag 可见性、嵌套 entity、双/三重 percent encoding。
- 加入 420 个单响应 shadow-surface 样本：action path、HTTP status、content type、body shape、body length。
- 用配额保证每类至少 60 个 shadow 样本，xss 至少 80 个 encoding-depth 样本。

结果：IID validation **100%**；3 个 fresh holdout 平均 **84.5833%**，最低 **84.5833%**；`dom_double_decode` holdout 达到 **100%**；最大 abstain rate **33.33%**。`compound_origin_role` 仍偏弱，已列为后续缺口。

产物：[v4 Rule IR checkpoint](../artifacts/rule-ir-decoder-loop-12-20260899-v4/rule_ir_decoder.pt)、[decoder report](../artifacts/rule-ir-decoder-loop-12-20260899-v4/report.json)。

## Shadow 后 slot binding

v3 矩阵协议为 [hidden matrix protocol v3](juice_shop_loop_12_hidden_matrix_protocol_v3.json)。绑定流程是：

1. shadow 只提供 sanitized status、selected headers、body shape/length、transport status 和 query；
2. decoder 输出候选 abstract policy-slot AST；
3. 只有置信度达到门槛才创建 binding；trusted origin 等不可见策略保持 `unbound`；
4. binding wrapper 标记 `executable: false`，不把证据包装成漏洞证明。

| 策略 | evaluator hit | fully bound actions | bound slots | decoder abstain |
|---|---:|---:|---:|---:|
| response head + Rule IR v4 | 0/7 | 3/7 | 6 | 4/7 |
| response-disabled ablation + Rule IR v4 | 0/7 | 2/7 | 4 | 5/7 |

因此 slot binding 已经真正落到 shadow evidence 上，但 action search 仍没有跨过 evaluator transition。普通 API surface 被预测为 input-validation 的错误也被原样记录，没有降低阈值掩盖。

## 验证

使用 workspace 根目录作为 `PYTHONPATH` 执行：**52 passed，1 warning**。

下一轮应做 family-disambiguating active probes 和 calibrated abstention/binding head；在此之前，不能把当前结果宣称为完整七族泛化。
