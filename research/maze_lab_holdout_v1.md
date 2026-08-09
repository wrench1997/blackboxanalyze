# 规则迷宫族外读出 v1

这是 oracle/协议层测试，不是模型分数。策略未看到漏洞族标签，靶场 evaluator 也没有被伪造。

- 训练契约族：`access_control, logic, url_redirect`
- 族外族：`injection, xss`
- 族外靶场：`8`，可观察出口：`8`
- evaluator 确认：`0`（预期为 0）
- 安全门禁：`PASS`
- 观测通道：`ast_shape, blind_response, bounded_timing, local_side_channel, semantic_contract, syntax_error`

SQL 的 error、blind/row-shape、bounded timing 都是确定性模拟标记；没有数据库执行、网络访问或真实 sleep。
DOM 的 sink/DOM 差分在浏览器端使用 detached node，在 Python 端使用 HTMLParser 复核；没有脚本执行。
