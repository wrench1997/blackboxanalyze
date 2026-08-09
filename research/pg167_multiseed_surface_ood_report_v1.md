# PG-167 multi-seed surface OOD

- train families: **sqli_boolean, sqli_search, sqli_timing, url_redirect, xss_dom_source**
- unseen families: **sqli_string, xss_reflected_get**
- baseline unseen PPL: **1652.60546784**
- replay seed unseen PPL mean/std: **1.57924625 / 0.03099716**
- projection overlap between train/holdout: **2**

由于投影碰撞，本轮不宣称族外泛化；先增加能区分表面族的 Rule-IR 特征。该轮不产生漏洞标签，也不晋级 checkpoint 或长期记忆。

