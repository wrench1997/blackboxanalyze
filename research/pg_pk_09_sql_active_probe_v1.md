# PG-PK-09 SQL active probe

screen 阶段每个 SQL surface 只发一个 plain probe；只有 decoder likelihood 与 bounded differential evidence 形成疑点时，才补发 url-percent pair。服务端仍不执行 SQL、不 sleep、不访问数据库。

静态请求数：15；active 请求数：13；节省：2；pair 完整数：5；oracle 复核 pair：4。
最终 belief entropy：1.8680；SQL decoder abstain：1。
共享路由 head abstain：0；OOD：0；它只作为 active prior，不拥有正向 authority。

| stage | pair | variant | decoder | posterior injection |
|---|---|---|---|---:|
| `screen` | `control` | `plain` | `control` | 0.137 |
| `screen` | `sql-pair-01` | `plain` | `injection` | 0.180 |
| `screen` | `sql-pair-02` | `plain` | `injection` | 0.214 |
| `screen` | `sql-pair-03` | `plain` | `injection` | 0.242 |
| `screen` | `sql-pair-04` | `plain` | `injection` | 0.233 |
| `screen` | `sql-pair-05` | `plain` | `injection` | 0.257 |
| `screen` | `sql-pair-06` | `plain` | `injection` | 0.276 |
| `screen` | `sql-pair-safe-01` | `plain` | `control` | 0.243 |
| `refine` | `sql-pair-safe-01` | `url_percent` | `control` | 0.218 |
| `refine` | `sql-pair-01` | `url_percent` | `injection` | 0.245 |
| `refine` | `sql-pair-04` | `url_percent` | `injection` | 0.266 |
| `refine` | `sql-pair-02` | `url_percent` | `injection` | 0.283 |
| `refine` | `sql-pair-03` | `url_percent` | `injection` | 0.297 |

active controller 只决定安全探针顺序，不直接宣布漏洞；Rule IR 仍需 pair、sink/AST oracle、fresh target 和 SHA-256 evidence 全部通过。
完整 JSON：`research\pg_pk_09_sql_active_probe_v1.json`
协议：`research\pg_pk_09_sql_active_probe_protocol_v1.json`
