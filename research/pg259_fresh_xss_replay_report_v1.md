# PG-242 Pikachu XSS/DOM browser acceptance

device=cuda; fresh=8; preflight=2; GET=7; POST=3
AI=8; reference=8; positive=6; negative_control=1; false_positive=0; external=0
positive routes=['GET /vul/xss/xss_reflected_get.php', 'GET /vul/xss/xss_01.php', 'GET /vul/xss/xss_04.php', 'GET /vul/xss/xss_dom.php', 'GET /vul/xss/xss_dom_x.php', 'POST /vul/xss/xsspost/xss_reflected_post.php']

浏览器只观察本地 DOM marker；原始 payload/wire 只 stdout 临时显示，持久化为哈希、投影和 evidence chain。stored XSS 路由未写数据库。
