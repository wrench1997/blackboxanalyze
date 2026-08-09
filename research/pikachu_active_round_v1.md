# Pikachu PG-PK-03 主动安全探测轮

控制器先为每个表面发送一个 plain canary，只有 bounded projection 出现候选信号时，才按 belief information gain 逐个追加编码变体；请求预算和 loopback scope 均由控制器强制。

请求数：9/12；screen：6；refine：3；候选信号：4。

| 阶段 | surface | variant | signals |
|---|---|---|---|
| screen | `dom_value_source` | `plain` | none |
| screen | `reflected_get` | `plain` | marker_reflected,marker_in_attribute |
| screen | `sqli_blind_boolean` | `plain` | none |
| screen | `sqli_blind_time` | `plain` | none |
| screen | `sqli_search` | `plain` | none |
| screen | `sqli_str` | `plain` | none |
| refine | `reflected_get` | `double_html_entity` | marker_reflected,marker_in_attribute |
| refine | `reflected_get` | `html_entity` | marker_reflected,marker_in_attribute |
| refine | `reflected_get` | `url_percent` | marker_reflected,marker_in_attribute |

这是主动选择探针的工程记录，不是漏洞确认；所有变体仍是无害编码/标识符，未执行脚本、SQL 语法、RCE、SSRF、XXE 或上传。

完整 JSON：`research\pikachu_active_round_v1.json`
协议：`research\pikachu_active_round_protocol_v1.json`
