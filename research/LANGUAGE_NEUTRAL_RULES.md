# 语言无关规则与特征存储

SIFT 不把源码 token 或 AST 节点本身当作长期规则记忆。JS、Python、Rust、Java、Wasm 或其他中间语言首先经过各自的前端适配器，再统一成 `Common Semantic Rule`。

## 必须保留的抽象语义

- 布尔关系：必要条件、充分条件、AND/OR 错置、短路顺序；
- 数值语义：严格/包含边界、整数宽度、溢出、浮点 NaN/Infinity；
- 类型转换：truthiness、隐式字符串/数值转换、nullish 与缺失值；
- 字符串与路径：包含、前缀、正则、URL host/path/query 的语义差异；
- 状态与历史：状态转移、Episode 边界、历史偏移、重放与版本；
- 控制与副作用：异常、提前返回、异步顺序、source/sink 和权限不变量；
- 证据：最小反例、Oracle 指纹、覆盖范围和可重放条件。

## 为什么不能只存 JS 特征

`if (quota)`、Python 的 `if quota:` 与某些 IR 的 `ToBoolean(quota)` 表面不同，但可共享同一抽象特征：

```json
{
  "primitive_families": ["coercion", "authorization"],
  "coercions": ["numeric_to_boolean_nonzero"],
  "security_property": "authorization must not depend on quota truthiness"
}
```

相反，不能盲目把不同语言视作完全相同。整数溢出、相等比较、异常传播、Unicode、异步调度和对象属性访问必须记录在 `language_semantics` 中。通用的是规则骨架；语言差异作为显式语义参数，而不是丢失掉。

机器可验证 Schema 见 `common_semantic_rule.schema.json`。
