# PG-165 真实 Docker 安全表面 attestation

- rows: **28**；GET/POST: **14/14**
- safe reflection pairs: **1**；safe no-effect pairs: **13**
- training-eligible surface rows: **28**

attestation 只证明无执行 canary 的表面反射/无效果，不证明 XSS、SQL 注入、重定向或认证绕过；原始 probe/响应正文不进入数据集。
