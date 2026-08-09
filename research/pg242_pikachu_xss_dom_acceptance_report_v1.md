# PG-242 Pikachu XSS/DOM browser acceptance

device=cuda; fresh=16; preflight=4; GET=14; POST=6
AI=16; reference=16; positive=12; negative_control=2; false_positive=0; external=0
positive routes=['GET /vul/xss/xss_reflected_get.php', 'GET /vul/xss/xss_01.php', 'GET /vul/xss/xss_04.php', 'GET /vul/xss/xss_dom.php', 'GET /vul/xss/xss_dom_x.php', 'POST /vul/xss/xsspost/xss_reflected_post.php', 'GET /vul/xss/xss_reflected_get.php', 'GET /vul/xss/xss_01.php', 'GET /vul/xss/xss_04.php', 'GET /vul/xss/xss_dom.php', 'GET /vul/xss/xss_dom_x.php', 'POST /vul/xss/xsspost/xss_reflected_post.php']

浏览器只观察本地 DOM marker；原始 payload/wire 只 stdout 临时显示，持久化为哈希、投影和 evidence chain。stored XSS 路由未写数据库。
