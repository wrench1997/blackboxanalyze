# Juice Shop External-Validity Loop 12：阶段总结

状态：**局部研究完成；响应训练的跨种子稳定性仍不足，完整泛化未确认。**

## 已确认

- 本地 Docker 重置、内部网络和证据链通过。
- 冻结 Loop 11 模型在动作接地上失败；无记忆神经组 6 次零成功，旧合成记忆组只能在第 6 次命中。
- shadow/evaluation 分离修复了 HEAD 副作用污染问题。
- 参数守恒的 8 槽响应头在一个 hidden observability surface 上选择 `/metrics`，同组件消融失败；旧合成族回归为 0.00pp。

## 尚未确认

- 新合成响应种子准确率为 89.50%、最低 88.00%，稳定性门槛未通过。
- 神经 shadow 只覆盖一个 hidden surface；其余 hidden family 和完整 24 题未跑。
- 神经头仍不输出 canonical Rule IR，抽象层尚未闭环。

下一目标：扩充组合式响应表面生成，注册多族 shadow 矩阵，在保持旧族 −2pp 门槛下恢复跨种子稳定性，再增加 Rule IR 解码实验。
