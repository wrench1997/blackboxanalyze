# PG-PK-09 抽象 SQL differential

本轮只发送白名单抽象 fragment class；服务端不执行 SQL、不访问数据库、不进行真实 sleep。错误、盲差分、行形状、超时和本地 side-channel 都由 AST/响应形状 oracle 给出 bounded evidence。

样本：15；严格 OOD abstain：15；全局 decoder model-only 接受：15；SQL channel decoder injection 接受：11；SQL oracle 复核样本：12（pair：6）。

| modality | pair count | revalidated |
|---|---:|---:|
| `ast_shape` | 2 | 1 |
| `blind_response` | 2 | 2 |
| `bounded_timing` | 1 | 1 |
| `local_side_channel` | 1 | 1 |
| `syntax_error` | 1 | 1 |

plain control revalidated：0；单 fixture 长期记忆晋升：`quarantine`。
这里的 revalidation 证明的是抽象通道/Rule IR 出口，不是对真实 SQL 服务发起攻击；任何真实目标仍必须保持本地授权、严格 OOD 和 abstain。

完整 JSON：`research\pg_pk_09_sql_differential_v1.json`
协议：`research\pg_pk_09_sql_differential_protocol_v1.json`
