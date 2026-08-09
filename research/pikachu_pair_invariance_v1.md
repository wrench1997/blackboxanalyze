# Pikachu PG-PK-02 编码/表面配对实验

本轮把同一抽象族拆成 plain、URL percent、HTML entity、double HTML entity 四种无害表示，并跨多个本地页面表面配对。pair id 只用于训练的一致性损失和评估分组，不进入模型可见输入。

样本：24；配对组：2；表面：6；编码变体：4。

| pair | surface | variants | rule signal count |
|---|---|---|---:|
| `pk-pair-family-a-01` | `dom_value_source` | plain,url_percent,html_entity,double_html_entity | 0 |
| `pk-pair-family-a-01` | `reflected_get` | plain,url_percent,html_entity,double_html_entity | 4 |
| `pk-pair-family-b-01` | `sqli_blind_boolean` | plain,url_percent,html_entity,double_html_entity | 0 |
| `pk-pair-family-b-01` | `sqli_blind_time` | plain,url_percent,html_entity,double_html_entity | 0 |
| `pk-pair-family-b-01` | `sqli_search` | plain,url_percent,html_entity,double_html_entity | 0 |
| `pk-pair-family-b-01` | `sqli_str` | plain,url_percent,html_entity,double_html_entity | 0 |

注意：reflection 只表示 HTTP 响应回显 canary；没有浏览器执行 oracle，也没有 SQL/RCE/SSRF/XXE exploit 确认。

Catalog：`research\pikachu_paired_catalog_v1.json`
训练协议：`research\pikachu_pair_invariance_protocol_v1.json`
完整 JSON：`research\pikachu_pair_invariance_v1.json`
