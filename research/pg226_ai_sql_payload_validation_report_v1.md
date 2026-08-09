# PG-226 AI SQL payload validation

fresh=8; GET=6; POST=2; AI=8; reference=8; negative=8
typed_effect=8; result_fixture_verified=8; training_candidate=8; false_positive=0

AI 只输出抽象 probe kind；wire 形状使用占位符，实际运行时值未落盘。typed effect / result fixture 是 pinned 本地路由证据，不是公网漏洞结论。

- POST /vul/sqli/sqli_id.php: probe=sql_channel_class; typed=True; result=True; wire=`POST <LOOPBACK_ORIGIN>/vul/sqli/sqli_id.php\nContent-Type: application/x-www-form-urlencoded\n\nid=<RUNTIME_SQL_SHAPE>&submit=submit`
- GET /vul/sqli/sqli_search.php: probe=sql_channel_class; typed=True; result=True; wire=`GET <LOOPBACK_ORIGIN>/vul/sqli/sqli_search.php?name=<RUNTIME_SQL_SHAPE>&submit=submit`
- GET /vul/sqli/sqli_str.php: probe=sql_channel_class; typed=True; result=True; wire=`GET <LOOPBACK_ORIGIN>/vul/sqli/sqli_str.php?name=<RUNTIME_SQL_SHAPE>&submit=submit`
- GET /vul/sqli/sqli_x.php: probe=sql_channel_class; typed=True; result=True; wire=`GET <LOOPBACK_ORIGIN>/vul/sqli/sqli_x.php?name=<RUNTIME_SQL_SHAPE>&submit=submit`
- POST /vul/sqli/sqli_id.php: probe=sql_channel_class; typed=True; result=True; wire=`POST <LOOPBACK_ORIGIN>/vul/sqli/sqli_id.php\nContent-Type: application/x-www-form-urlencoded\n\nid=<RUNTIME_SQL_SHAPE>&submit=submit`
- GET /vul/sqli/sqli_search.php: probe=sql_channel_class; typed=True; result=True; wire=`GET <LOOPBACK_ORIGIN>/vul/sqli/sqli_search.php?name=<RUNTIME_SQL_SHAPE>&submit=submit`
- GET /vul/sqli/sqli_str.php: probe=sql_channel_class; typed=True; result=True; wire=`GET <LOOPBACK_ORIGIN>/vul/sqli/sqli_str.php?name=<RUNTIME_SQL_SHAPE>&submit=submit`
- GET /vul/sqli/sqli_x.php: probe=sql_channel_class; typed=True; result=True; wire=`GET <LOOPBACK_ORIGIN>/vul/sqli/sqli_x.php?name=<RUNTIME_SQL_SHAPE>&submit=submit`
