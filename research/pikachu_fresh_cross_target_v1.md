# Pikachu PG-PK-03 fresh cross-target 回放

旧实例请求数：9；新实例请求数：9。
同一表面/编码的 bounded 结果一致率：100.00%；一致条目：9/9。

| surface | variant | 旧实例 signals | 新实例 signals | 一致 |
|---|---|---|---|---|
| `dom_value_source` | `plain` | none | none | yes |
| `reflected_get` | `double_html_entity` | marker_reflected,marker_in_attribute | marker_in_attribute,marker_reflected | yes |
| `reflected_get` | `html_entity` | marker_reflected,marker_in_attribute | marker_in_attribute,marker_reflected | yes |
| `reflected_get` | `plain` | marker_reflected,marker_in_attribute | marker_in_attribute,marker_reflected | yes |
| `reflected_get` | `url_percent` | marker_reflected,marker_in_attribute | marker_in_attribute,marker_reflected | yes |
| `sqli_blind_boolean` | `plain` | none | none | yes |
| `sqli_blind_time` | `plain` | none | none | yes |
| `sqli_search` | `plain` | none | none | yes |
| `sqli_str` | `plain` | none | none | yes |

新实例由固定 SHA-256 镜像创建，端口仅绑定 127.0.0.1:8767；容器在脚本结束时停止并由 `--rm` 回收。
这仍然是 bounded signal 的跨实例稳定性实验，不是漏洞确认；没有执行脚本、SQL 语法/延时、RCE、SSRF、XXE、上传或凭据提交。

完整 JSON：`research\pikachu_fresh_cross_target_v1.json`
