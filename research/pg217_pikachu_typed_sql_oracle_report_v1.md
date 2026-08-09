# PG-217 Pikachu typed SQL oracle

device=cuda:0; fresh containers=14; GET=10; POST=4
AI sends=14; negative sends=14; reference sends=14
typed effect confirmed=8; abstain=6; restart=0
local typed routes=['POST /vul/sqli/sqli_id.php', 'GET /vul/sqli/sqli_search.php', 'GET /vul/sqli/sqli_str.php', 'GET /vul/sqli/sqli_x.php', 'POST /vul/sqli/sqli_id.php', 'GET /vul/sqli/sqli_search.php', 'GET /vul/sqli/sqli_str.php', 'GET /vul/sqli/sqli_x.php']

confirmed_positive 只表示 pinned Pikachu 本地路由通过 fresh reset、negative、reference、source hash 和证据 hash 的输入边界 oracle；不等于对任意站点的漏洞结论。原始 payload/响应均未落盘。
