# PG-100 独立语义 sink oracle 回放

fresh Docker cases: 7；confirmed_positive: 7；abstain: 0。

| family | surface | method | independent oracle | status |
|---|---|---|---|---|
| `xss` | `xss_reflected_get` | `GET` | `browser_dom_execution` | `confirmed_positive` |
| `xss` | `xss_dom_source` | `GET` | `browser_dom_execution` | `confirmed_positive` |
| `xss` | `xss_reflected_post` | `POST` | `browser_dom_execution` | `confirmed_positive` |
| `injection` | `sqli_str` | `GET` | `sql_ast_differential` | `confirmed_positive` |
| `injection` | `sqli_search` | `GET` | `sql_ast_differential` | `confirmed_positive` |
| `injection` | `sqli_blind_b` | `GET` | `sql_ast_differential` | `confirmed_positive` |
| `url_redirect` | `url_redirect` | `GET` | `redirect_destination_controlled` | `confirmed_positive` |

PG-100 只验证验收通道，不把 oracle 标签放进模型输入，也不生成训练样本或长期记忆。PG-99 的已知/未知可见等价类仍然存在，因此能力门保持 blocked。

JSON: `research\pg100_semantic_sink_report_v1.json`
协议: `research\pg100_semantic_sink_protocol_v1.json`
