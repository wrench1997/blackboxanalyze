# PG-190 dual-head action/gate local replay

device=cuda; routes=2; sent_get=2; sent_post=2; candidates=0; abstain=2; positives=0

| surface | sent GET | sent POST | candidates | abstain | manifest errors |
|---|---:|---:|---:|---:|---:|
| sqli_id_post | 1 | 1 | 0 | 1 | 0 |
| xss_stored_post | 1 | 1 | 0 | 1 | 0 |

模型输出 abstract action；POST 只使用浏览器观测字段和非执行 canary。typed oracle 不可用时一律 abstain，不能宣称漏洞或生成真实利用串。
