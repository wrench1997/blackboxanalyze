# PG-217/218 API payload human view

> 这是本地、只读 Pikachu 回放的请求视图。`<RUNTIME_CANARY>` 代表发送时由 API 临时绑定的短 canary；原始值和响应正文不落盘。它是 SQL 输入边界检测 probe，不是数据导出/时间延迟/写入 payload。

## 发送流程

1. AI 根据 route 的 method/path/fields 选择 `sql_channel_class`。
2. 运行时把抽象类绑定到对应字段，发送一个 syntax-boundary canary；独立 reference 再发一次同类 probe。
3. 另发一个普通未知值作 negative control。
4. evaluator 只比较 SQL 错误形状、negative、reference、fresh reset、数据库健康门和 SHA-256 证据。

## 路由视图

| 方法 | 路径 | 字段 | API 探针 | probe hash 前缀 | 本地 typed effect | 结果 fixture |
|---|---|---|---|---|---:|---:|
| GET | `/vul/sqli/sqli_blind_b.php` | `name` | `boolean-blind boundary (abstain)` | `a4a755aa417c` | abstain | not used |
| GET | `/vul/sqli/sqli_blind_t.php` | `name` | `timing channel (forbidden/abstain)` | `a4a755aa417c` | abstain | not used |
| POST | `/vul/sqli/sqli_id.php` | `id` | `numeric syntax boundary` | `a4a755aa417c` | yes | verified |
| GET | `/vul/sqli/sqli_search.php` | `name` | `LIKE/string syntax boundary` | `a4a755aa417c` | yes | verified |
| GET | `/vul/sqli/sqli_str.php` | `name` | `quoted-string syntax boundary` | `a4a755aa417c` | yes | verified |
| POST | `/vul/sqli/sqli_widebyte.php` | `name` | `escaped/wide-byte boundary (abstain)` | `a4a755aa417c` | abstain | not used |
| GET | `/vul/sqli/sqli_x.php` | `name` | `parenthesized-string syntax boundary` | `a4a755aa417c` | yes | verified |

## 可读的 wire 形状（占位显示）

### GET /vul/sqli/sqli_blind_b.php

```text
GET <LOOPBACK_ORIGIN>/vul/sqli/sqli_blind_b.php?name=<RUNTIME_CANARY>'&submit=submit
```

### GET /vul/sqli/sqli_blind_t.php

```text
GET <LOOPBACK_ORIGIN>/vul/sqli/sqli_blind_t.php?name=<RUNTIME_CANARY>'&submit=submit
```

### POST /vul/sqli/sqli_id.php

```text
POST <LOOPBACK_ORIGIN>/vul/sqli/sqli_id.php
Content-Type: application/x-www-form-urlencoded

id=1'&submit=submit
```

### GET /vul/sqli/sqli_search.php

```text
GET <LOOPBACK_ORIGIN>/vul/sqli/sqli_search.php?name=<RUNTIME_CANARY>'&submit=submit
```

### GET /vul/sqli/sqli_str.php

```text
GET <LOOPBACK_ORIGIN>/vul/sqli/sqli_str.php?name=<RUNTIME_CANARY>'&submit=submit
```

### POST /vul/sqli/sqli_widebyte.php

```text
POST <LOOPBACK_ORIGIN>/vul/sqli/sqli_widebyte.php
Content-Type: application/x-www-form-urlencoded

name=<RUNTIME_CANARY>'&submit=submit
```

### GET /vul/sqli/sqli_x.php

```text
GET <LOOPBACK_ORIGIN>/vul/sqli/sqli_x.php?name=<RUNTIME_CANARY>'&submit=submit
```

## 结果怎么读

PG-217 共 8/14 个 route episodes 通过本地 typed input-boundary oracle；PG-218 中 8 个同时通过只读已知记录/负对照。

`blind_b`、`blind_t` 和 `widebyte` 目前显示 abstain：不是说它们不存在问题，而是当前安全、非时间、非写入 oracle 没有足够证据。

原始 probe、原始响应、SQL 查询文本和可执行利用字符串均不写入 catalog；报告只保留 request anatomy、projection hash、probe/payload hash 和 evidence hash。
