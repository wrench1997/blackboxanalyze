# PG-234 Pikachu AI payload wire grounding

rows=22; GET=20; POST=2; AI=22
SQL typed+result=8; DOM effect=4; redirect effect=0; false_positive=0

以下 wire 只显示运行时占位符；实际值仅在本机 fresh container 复放期间存在。SQL 必须同时通过 typed result；DOM effect 不等于 XSS；redirect shape 不等于 open redirect。

- POST /vul/sqli/sqli_id.php [sql]: probe=sql_channel_class; validation=confirmed_local_typed_result; wire=`POST <LOOPBACK_ORIGIN>/vul/sqli/sqli_id.php\nContent-Type: application/x-www-form-urlencoded\n\nid=<RUNTIME_SQL_SHAPE>&submit=submit`
- POST /vul/sqli/sqli_id.php [sql]: probe=sql_channel_class; validation=confirmed_local_typed_result; wire=`POST <LOOPBACK_ORIGIN>/vul/sqli/sqli_id.php\nContent-Type: application/x-www-form-urlencoded\n\nid=<RUNTIME_SQL_SHAPE>&submit=submit`
- GET /vul/sqli/sqli_search.php [sql]: probe=sql_channel_class; validation=confirmed_local_typed_result; wire=`GET <LOOPBACK_ORIGIN>/vul/sqli/sqli_search.php?name=<RUNTIME_SQL_SHAPE>&submit=submit`
- GET /vul/sqli/sqli_search.php [sql]: probe=sql_channel_class; validation=confirmed_local_typed_result; wire=`GET <LOOPBACK_ORIGIN>/vul/sqli/sqli_search.php?name=<RUNTIME_SQL_SHAPE>&submit=submit`
- GET /vul/sqli/sqli_str.php [sql]: probe=sql_channel_class; validation=confirmed_local_typed_result; wire=`GET <LOOPBACK_ORIGIN>/vul/sqli/sqli_str.php?name=<RUNTIME_SQL_SHAPE>&submit=submit`
- GET /vul/sqli/sqli_str.php [sql]: probe=sql_channel_class; validation=confirmed_local_typed_result; wire=`GET <LOOPBACK_ORIGIN>/vul/sqli/sqli_str.php?name=<RUNTIME_SQL_SHAPE>&submit=submit`
- GET /vul/sqli/sqli_x.php [sql]: probe=sql_channel_class; validation=confirmed_local_typed_result; wire=`GET <LOOPBACK_ORIGIN>/vul/sqli/sqli_x.php?name=<RUNTIME_SQL_SHAPE>&submit=submit`
- GET /vul/sqli/sqli_x.php [sql]: probe=sql_channel_class; validation=confirmed_local_typed_result; wire=`GET <LOOPBACK_ORIGIN>/vul/sqli/sqli_x.php?name=<RUNTIME_SQL_SHAPE>&submit=submit`
- GET /vul/urlredirect/urlredirect.php [xss_or_redirect]: probe=http_canary; validation=no_typed_positive; wire=`GET <LOOPBACK_ORIGIN>/vul/urlredirect/urlredirect.php?url=<RUNTIME_CANARY>`
- GET /vul/urlredirect/urlredirect.php [xss_or_redirect]: probe=http_canary; validation=no_typed_positive; wire=`GET <LOOPBACK_ORIGIN>/vul/urlredirect/urlredirect.php?url=<RUNTIME_CANARY>`
- GET /vul/xss/xss_01.php [xss_or_redirect]: probe=inert_dom_markup; validation=confirmed_local_dom_surface_effect; wire=`GET <LOOPBACK_ORIGIN>/vul/xss/xss_01.php?message=<RUNTIME_CANARY>&submit=submit`
- GET /vul/xss/xss_01.php [xss_or_redirect]: probe=inert_dom_markup; validation=confirmed_local_dom_surface_effect; wire=`GET <LOOPBACK_ORIGIN>/vul/xss/xss_01.php?message=<RUNTIME_CANARY>&submit=submit`
- GET /vul/xss/xss_02.php [xss_or_redirect]: probe=inert_dom_markup; validation=no_typed_positive; wire=`GET <LOOPBACK_ORIGIN>/vul/xss/xss_02.php?message=<RUNTIME_CANARY>&submit=submit`
- GET /vul/xss/xss_02.php [xss_or_redirect]: probe=inert_dom_markup; validation=no_typed_positive; wire=`GET <LOOPBACK_ORIGIN>/vul/xss/xss_02.php?message=<RUNTIME_CANARY>&submit=submit`
- GET /vul/xss/xss_03.php [xss_or_redirect]: probe=inert_dom_markup; validation=no_typed_positive; wire=`GET <LOOPBACK_ORIGIN>/vul/xss/xss_03.php?message=<RUNTIME_CANARY>&submit=submit`
- GET /vul/xss/xss_03.php [xss_or_redirect]: probe=inert_dom_markup; validation=no_typed_positive; wire=`GET <LOOPBACK_ORIGIN>/vul/xss/xss_03.php?message=<RUNTIME_CANARY>&submit=submit`
- GET /vul/xss/xss_04.php [xss_or_redirect]: probe=inert_dom_markup; validation=no_typed_positive; wire=`GET <LOOPBACK_ORIGIN>/vul/xss/xss_04.php?message=<RUNTIME_CANARY>&submit=submit`
- GET /vul/xss/xss_04.php [xss_or_redirect]: probe=inert_dom_markup; validation=no_typed_positive; wire=`GET <LOOPBACK_ORIGIN>/vul/xss/xss_04.php?message=<RUNTIME_CANARY>&submit=submit`
- GET /vul/xss/xss_dom_x.php [xss_or_redirect]: probe=inert_dom_markup; validation=no_typed_positive; wire=`GET <LOOPBACK_ORIGIN>/vul/xss/xss_dom_x.php?text=<RUNTIME_CANARY>`
- GET /vul/xss/xss_dom_x.php [xss_or_redirect]: probe=inert_dom_markup; validation=no_typed_positive; wire=`GET <LOOPBACK_ORIGIN>/vul/xss/xss_dom_x.php?text=<RUNTIME_CANARY>`
- GET /vul/xss/xss_reflected_get.php [xss_or_redirect]: probe=inert_dom_markup; validation=confirmed_local_dom_surface_effect; wire=`GET <LOOPBACK_ORIGIN>/vul/xss/xss_reflected_get.php?message=<RUNTIME_CANARY>&submit=submit`
- GET /vul/xss/xss_reflected_get.php [xss_or_redirect]: probe=inert_dom_markup; validation=confirmed_local_dom_surface_effect; wire=`GET <LOOPBACK_ORIGIN>/vul/xss/xss_reflected_get.php?message=<RUNTIME_CANARY>&submit=submit`
