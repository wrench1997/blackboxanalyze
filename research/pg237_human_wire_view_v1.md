# PG-237 Pikachu AI payload wire view

> 这是两个新 seed 的本地、只读、fresh-container 回放。AI candidate、独立 reference、matched negative 和结果 fixture 都实际发包；运行时值只在 loopback 容器内绑定，本文只展示可读 wire 形状和哈希前缀。

## AI 参与的流程

1. AI 选择 `sql_channel_class` 抽象探针。
2. 控制器按已观察的 method/path/field 绑定运行时 SQL 边界 probe。
3. 同一 fresh 容器分别发送 reference、negative 和只读结果 fixture。
4. 只有 typed effect、结果形状、阴性干净、fresh reset、数据库健康门和证据哈希同时成立，才标为训练 candidate；不成立则 abstain。

## 路由与结果

| method | route | field | AI sent | typed effect | result fixture | negative | payload hash | evidence hash |
|---|---|---|---:|---:|---:|---:|---|---|
| GET | `/vul/sqli/sqli_blind_b.php` | `name` | 1 | 0 | 0 | 1 | `538e7d0ceadf` | `d1d3ed6b2e49` |
| GET | `/vul/sqli/sqli_blind_t.php` | `name` | 1 | 0 | 0 | 1 | `8a3c2baef7bc` | `fe082311d0ad` |
| POST | `/vul/sqli/sqli_id.php` | `id` | 1 | 1 | 1 | 1 | `8cc587115d02` | `3bcd7a3d1e68` |
| GET | `/vul/sqli/sqli_search.php` | `name` | 1 | 1 | 1 | 1 | `8fe0274ae11d` | `6a0cf1a0cf4c` |
| GET | `/vul/sqli/sqli_str.php` | `name` | 1 | 1 | 1 | 1 | `7b3b406243f1` | `214da444f8a6` |
| POST | `/vul/sqli/sqli_widebyte.php` | `name` | 1 | 0 | 0 | 1 | `bb51f4f1bf83` | `705e13f602c9` |
| GET | `/vul/sqli/sqli_x.php` | `name` | 1 | 1 | 1 | 1 | `df5b72d70b36` | `96a46e2578a7` |

## 可读的 wire 形状（占位）

### GET /vul/sqli/sqli_blind_b.php

```text
GET <LOOPBACK_ORIGIN>/vul/sqli/sqli_blind_b.php?name=<RUNTIME_SQL_BOUND_PROBE>&submit=submit
```

### GET /vul/sqli/sqli_blind_t.php

```text
GET <LOOPBACK_ORIGIN>/vul/sqli/sqli_blind_t.php?name=<RUNTIME_SQL_BOUND_PROBE>&submit=submit
```

### POST /vul/sqli/sqli_id.php

```text
POST <LOOPBACK_ORIGIN>/vul/sqli/sqli_id.php
Content-Type: application/x-www-form-urlencoded

id=<RUNTIME_SQL_BOUND_PROBE>&submit=submit
```

### GET /vul/sqli/sqli_search.php

```text
GET <LOOPBACK_ORIGIN>/vul/sqli/sqli_search.php?name=<RUNTIME_SQL_BOUND_PROBE>&submit=submit
```

### GET /vul/sqli/sqli_str.php

```text
GET <LOOPBACK_ORIGIN>/vul/sqli/sqli_str.php?name=<RUNTIME_SQL_BOUND_PROBE>&submit=submit
```

### POST /vul/sqli/sqli_widebyte.php

```text
POST <LOOPBACK_ORIGIN>/vul/sqli/sqli_widebyte.php
Content-Type: application/x-www-form-urlencoded

name=<RUNTIME_SQL_BOUND_PROBE>&submit=submit
```

### GET /vul/sqli/sqli_x.php

```text
GET <LOOPBACK_ORIGIN>/vul/sqli/sqli_x.php?name=<RUNTIME_SQL_BOUND_PROBE>&submit=submit
```

## 解释

`<RUNTIME_SQL_BOUND_PROBE>` 不是可复用的原始字符串；它代表 API 在本地发送时临时绑定的受控边界探针。结果 fixture 只用于验证本地只读记录/阴性差分，不能据此宣称任意网站存在漏洞。

原始 payload、原始响应正文、数据库查询文本和秘密均未写入该视图。
