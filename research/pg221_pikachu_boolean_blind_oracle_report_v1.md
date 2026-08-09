# PG-221 Pikachu boolean blind oracle

fresh=2; GET=2; AI pairs=4; reference pairs=4
boolean effect confirmed=2; false_positive=0

AI 选择的是抽象 blind_boolean 类；真假值只在 loopback 请求发送时绑定。结果只表示本地教学路由的可重复真假回显差异，不是任意站点漏洞断言。sqli_blind_t 仍没有安全、非时间型 oracle，保持 abstain。

wire 形状（占位）：GET <LOOPBACK_ORIGIN>/vul/sqli/sqli_blind_b.php?name=<RUNTIME_BOOLEAN_TRUE>&submit=submit；再发 FALSE 对照。
