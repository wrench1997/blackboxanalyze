# PG-241 Pikachu payload acceptance

device=cuda; fresh=14; GET=10; POST=4
AI sends=12; reference sends=12; confirmed positives=12; model misses=0; abstain timing=2
confirmed routes=['POST /vul/sqli/sqli_id.php', 'GET /vul/sqli/sqli_search.php', 'GET /vul/sqli/sqli_str.php', 'GET /vul/sqli/sqli_x.php', 'GET /vul/sqli/sqli_blind_b.php', 'POST /vul/sqli/sqli_widebyte.php', 'POST /vul/sqli/sqli_id.php', 'GET /vul/sqli/sqli_search.php', 'GET /vul/sqli/sqli_str.php', 'GET /vul/sqli/sqli_x.php', 'GET /vul/sqli/sqli_blind_b.php', 'POST /vul/sqli/sqli_widebyte.php']

实际 wire 仅在运行时 stdout 显示；持久化只保留哈希、响应投影和证据链。确认结果仅限本地 pinned Pikachu 源码/运行层。
