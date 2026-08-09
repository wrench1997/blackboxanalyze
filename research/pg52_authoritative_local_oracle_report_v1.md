# PG-52 本地权威 oracle 检测

真实 Docker cases: 7；confirmed_positive: 7；confirmed_negative: 0。
模型 family 命中 3/7；typed oracle 绑定 7/7。

| family | surface | method | model proposal | typed binding | oracle |
|---|---|---|---|---|---|
| `xss` | `xss_reflected_get` | `GET` | `xss` | `xss` | `confirmed_positive` |
| `xss` | `xss_dom_source` | `GET` | `xss` | `xss` | `confirmed_positive` |
| `xss` | `xss_reflected_post` | `POST` | `xss` | `xss` | `confirmed_positive` |
| `injection` | `sqli_str` | `GET` | `xss` | `injection` | `confirmed_positive` |
| `injection` | `sqli_search` | `GET` | `xss` | `injection` | `confirmed_positive` |
| `injection` | `sqli_blind_b` | `GET` | `xss` | `injection` | `confirmed_positive` |
| `url_redirect` | `url_redirect` | `GET` | `xss` | `url_redirect` | `confirmed_positive` |

浏览器 oracle 使用 loopback 响应的离线渲染；DOM 案例若需受控事件派发会在证据中标为 `controlled_event_dispatch`。SQL oracle 只观察只读 SELECT 的 AST 差分；重定向 oracle 不跟随目的地。原始 payload、响应正文和账号材料均未写入报告。

JSON: `research\pg52_authoritative_local_oracle_report_v1.json`
协议: `research\pg52_authoritative_local_oracle_protocol_v1.json`
