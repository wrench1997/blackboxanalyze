# PG-PK-05 Juice Shop 跨应用 shadow

这是与 Pikachu 完全分开的本地应用表面留出：只发送 5 个 allow-listed GET canary，模型看不到应用/路径标签，且 Juice Shop 轨道没有已授权的族特异 oracle，因此正确行为是 abstain。

样本：5；模型-only 接受：5；oracle gate 接受：0；oracle gate abstain：100.00%。

| surface | candidate family | model p | OOD | oracle support | model-only | oracle gate |
|---|---|---:|---|---:|---|---|
| `juice_spa_shell` | `xss` | 0.941 | yes | 0.020 | accept | abstain |
| `juice_robots_text` | `xss` | 0.981 | yes | 0.020 | accept | abstain |
| `juice_search_unknown` | `xss` | 0.845 | yes | 0.020 | accept | abstain |
| `juice_search_common` | `xss` | 0.845 | yes | 0.020 | accept | abstain |
| `juice_search_percent_input` | `xss` | 0.847 | yes | 0.020 | accept | abstain |

model-only 的接受不是漏洞判断；本轮没有族特异 oracle，因此任何 Rule IR 发射都属于不合格猜测。
没有访问 `/api/Challenges`、`/snippets` 或其他 evaluator 路径；没有执行脚本、SQL 语法/延时、RCE、SSRF、XXE、上传或凭据提交。

完整 JSON：`research\juice_shop_cross_app_shadow_v1.json`
协议：`research\juice_shop_cross_app_shadow_protocol_v1.json`
