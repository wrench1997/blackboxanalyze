# PG-224 Pikachu parameter-surface collection

routes=44 seeds=2 safe_send=30 GET=26 POST=4 preflight_only=58

AI 只选择抽象 probe_kind；实际值在 loopback 请求瞬间绑定。下面的 wire 形状使用占位符，原始值和响应正文没有保存。

| method | route | fields | policy/status | wire shape |
|---|---|---|---|---|
| GET | /vul/csrf/csrfget/csrf_get_login.php | password,submit,username | preflight_only | `GET <LOOPBACK_ORIGIN>/vul/csrf/csrfget/csrf_get_login.php?password=<RUNTIME_CANARY>&submit=submit&username=<RUNTIME_CANARY>` |
| GET | /vul/csrf/csrfpost/csrf_post_login.php | password,submit,username | preflight_only | `GET <LOOPBACK_ORIGIN>/vul/csrf/csrfpost/csrf_post_login.php?password=<RUNTIME_CANARY>&submit=submit&username=<RUNTIME_CANARY>` |
| GET | /vul/csrf/csrftoken/token_get_login.php | password,submit,username | preflight_only | `GET <LOOPBACK_ORIGIN>/vul/csrf/csrftoken/token_get_login.php?password=<RUNTIME_CANARY>&submit=submit&username=<RUNTIME_CANARY>` |
| GET | /vul/dir/dir_list.php | title | completed_projection_only | `GET <LOOPBACK_ORIGIN>/vul/dir/dir_list.php?title=<RUNTIME_CANARY>` |
| GET | /vul/fileinclude/fi_local.php | filename,submit | preflight_only | `GET <LOOPBACK_ORIGIN>/vul/fileinclude/fi_local.php?filename=<RUNTIME_CANARY>&submit=submit` |
| GET | /vul/fileinclude/fi_remote.php | filename,submit | preflight_only | `GET <LOOPBACK_ORIGIN>/vul/fileinclude/fi_remote.php?filename=<RUNTIME_CANARY>&submit=submit` |
| GET | /vul/infoleak/findabc.php | password,submit,username | preflight_only | `GET <LOOPBACK_ORIGIN>/vul/infoleak/findabc.php?password=<RUNTIME_CANARY>&submit=submit&username=<RUNTIME_CANARY>` |
| GET | /vul/sqli/sqli_blind_b.php | name,submit | completed_projection_only | `GET <LOOPBACK_ORIGIN>/vul/sqli/sqli_blind_b.php?name=<RUNTIME_SQL_SHAPE>&submit=submit` |
| GET | /vul/sqli/sqli_blind_t.php | name,submit | completed_projection_only | `GET <LOOPBACK_ORIGIN>/vul/sqli/sqli_blind_t.php?name=<RUNTIME_SQL_SHAPE>&submit=submit` |
| GET | /vul/sqli/sqli_iu/sqli_login.php | password,submit,username | preflight_only | `GET <LOOPBACK_ORIGIN>/vul/sqli/sqli_iu/sqli_login.php?password=<RUNTIME_CANARY>&submit=submit&username=<RUNTIME_CANARY>` |
| GET | /vul/sqli/sqli_search.php | name,submit | completed_projection_only | `GET <LOOPBACK_ORIGIN>/vul/sqli/sqli_search.php?name=<RUNTIME_SQL_SHAPE>&submit=submit` |
| GET | /vul/sqli/sqli_str.php | name,submit | completed_projection_only | `GET <LOOPBACK_ORIGIN>/vul/sqli/sqli_str.php?name=<RUNTIME_SQL_SHAPE>&submit=submit` |
| GET | /vul/sqli/sqli_x.php | name,submit | completed_projection_only | `GET <LOOPBACK_ORIGIN>/vul/sqli/sqli_x.php?name=<RUNTIME_SQL_SHAPE>&submit=submit` |
| GET | /vul/ssrf/ssrf_curl.php | url | preflight_only | `GET <LOOPBACK_ORIGIN>/vul/ssrf/ssrf_curl.php?url=<RUNTIME_CANARY>` |
| GET | /vul/ssrf/ssrf_fgc.php | file | preflight_only | `GET <LOOPBACK_ORIGIN>/vul/ssrf/ssrf_fgc.php?file=<RUNTIME_CANARY>` |
| GET | /vul/unsafedownload/execdownload.php | filename | preflight_only | `GET <LOOPBACK_ORIGIN>/vul/unsafedownload/execdownload.php?filename=<RUNTIME_CANARY>` |
| GET | /vul/urlredirect/urlredirect.php | url | completed_projection_only | `GET <LOOPBACK_ORIGIN>/vul/urlredirect/urlredirect.php?url=<RUNTIME_CANARY>` |
| GET | /vul/xss/xss_01.php | message,submit | completed_projection_only | `GET <LOOPBACK_ORIGIN>/vul/xss/xss_01.php?message=<RUNTIME_CANARY>&submit=submit` |
| GET | /vul/xss/xss_02.php | message,submit | completed_projection_only | `GET <LOOPBACK_ORIGIN>/vul/xss/xss_02.php?message=<RUNTIME_CANARY>&submit=submit` |
| GET | /vul/xss/xss_03.php | message,submit | completed_projection_only | `GET <LOOPBACK_ORIGIN>/vul/xss/xss_03.php?message=<RUNTIME_CANARY>&submit=submit` |
| GET | /vul/xss/xss_04.php | message,submit | completed_projection_only | `GET <LOOPBACK_ORIGIN>/vul/xss/xss_04.php?message=<RUNTIME_CANARY>&submit=submit` |
| GET | /vul/xss/xss_dom_x.php | text | completed_projection_only | `GET <LOOPBACK_ORIGIN>/vul/xss/xss_dom_x.php?text=<RUNTIME_CANARY>` |
| GET | /vul/xss/xss_reflected_get.php | message,submit | completed_projection_only | `GET <LOOPBACK_ORIGIN>/vul/xss/xss_reflected_get.php?message=<RUNTIME_CANARY>&submit=submit` |
| POST | /pkxss/pkxss_install.php | submit | preflight_only | `POST <LOOPBACK_ORIGIN>/pkxss/pkxss_install.php\nContent-Type: application/x-www-form-urlencoded\n\nsubmit=submit` |
| POST | /vul/burteforce/bf_client.php | password,submit,username,vcode | preflight_only | `POST <LOOPBACK_ORIGIN>/vul/burteforce/bf_client.php\nContent-Type: application/x-www-form-urlencoded\n\npassword=<RUNTIME_CANARY>&submit=submit&username=<RUNTIME_CANARY>&vcode=<RUNTIME_CANARY>` |
| POST | /vul/burteforce/bf_form.php | password,submit,username | preflight_only | `POST <LOOPBACK_ORIGIN>/vul/burteforce/bf_form.php\nContent-Type: application/x-www-form-urlencoded\n\npassword=<RUNTIME_CANARY>&submit=submit&username=<RUNTIME_CANARY>` |
| POST | /vul/burteforce/bf_server.php | password,submit,username,vcode | preflight_only | `POST <LOOPBACK_ORIGIN>/vul/burteforce/bf_server.php\nContent-Type: application/x-www-form-urlencoded\n\npassword=<RUNTIME_CANARY>&submit=submit&username=<RUNTIME_CANARY>&vcode=<RUNTIME_CANARY>` |
| POST | /vul/burteforce/bf_token.php | password,submit,token,username | preflight_only | `POST <LOOPBACK_ORIGIN>/vul/burteforce/bf_token.php\nContent-Type: application/x-www-form-urlencoded\n\npassword=<RUNTIME_CANARY>&submit=submit&token=<RUNTIME_CANARY>&username=<RUNTIME_CANARY>` |
| POST | /vul/overpermission/op1/op1_login.php | password,submit,username | preflight_only | `POST <LOOPBACK_ORIGIN>/vul/overpermission/op1/op1_login.php\nContent-Type: application/x-www-form-urlencoded\n\npassword=<RUNTIME_CANARY>&submit=submit&username=<RUNTIME_CANARY>` |
| POST | /vul/overpermission/op2/op2_login.php | password,submit,username | preflight_only | `POST <LOOPBACK_ORIGIN>/vul/overpermission/op2/op2_login.php\nContent-Type: application/x-www-form-urlencoded\n\npassword=<RUNTIME_CANARY>&submit=submit&username=<RUNTIME_CANARY>` |
| POST | /vul/rce/rce_eval.php | submit,txt | preflight_only | `POST <LOOPBACK_ORIGIN>/vul/rce/rce_eval.php\nContent-Type: application/x-www-form-urlencoded\n\nsubmit=submit&txt=<RUNTIME_CANARY>` |
| POST | /vul/rce/rce_ping.php | ipaddress,submit | preflight_only | `POST <LOOPBACK_ORIGIN>/vul/rce/rce_ping.php\nContent-Type: application/x-www-form-urlencoded\n\nipaddress=<RUNTIME_CANARY>&submit=submit` |
| POST | /vul/sqli/sqli_del.php | message,submit | preflight_only | `POST <LOOPBACK_ORIGIN>/vul/sqli/sqli_del.php\nContent-Type: application/x-www-form-urlencoded\n\nmessage=<RUNTIME_CANARY>&submit=submit` |
| POST | /vul/sqli/sqli_header/sqli_header_login.php | password,submit,username | preflight_only | `POST <LOOPBACK_ORIGIN>/vul/sqli/sqli_header/sqli_header_login.php\nContent-Type: application/x-www-form-urlencoded\n\npassword=<RUNTIME_CANARY>&submit=submit&username=<RUNTIME_CANARY>` |
| POST | /vul/sqli/sqli_id.php | id,submit | completed_projection_only | `POST <LOOPBACK_ORIGIN>/vul/sqli/sqli_id.php\nContent-Type: application/x-www-form-urlencoded\n\nid=<RUNTIME_SQL_SHAPE>&submit=submit` |
| POST | /vul/sqli/sqli_widebyte.php | name,submit | completed_projection_only | `POST <LOOPBACK_ORIGIN>/vul/sqli/sqli_widebyte.php\nContent-Type: application/x-www-form-urlencoded\n\nname=<RUNTIME_SQL_SHAPE>&submit=submit` |
| POST | /vul/unsafeupload/clientcheck.php | submit,uploadfile | preflight_only | `POST <LOOPBACK_ORIGIN>/vul/unsafeupload/clientcheck.php\nContent-Type: application/x-www-form-urlencoded\n\nsubmit=submit&uploadfile=<RUNTIME_CANARY>` |
| POST | /vul/unsafeupload/getimagesize.php | submit,uploadfile | preflight_only | `POST <LOOPBACK_ORIGIN>/vul/unsafeupload/getimagesize.php\nContent-Type: application/x-www-form-urlencoded\n\nsubmit=submit&uploadfile=<RUNTIME_CANARY>` |
| POST | /vul/unsafeupload/servercheck.php | submit,uploadfile | preflight_only | `POST <LOOPBACK_ORIGIN>/vul/unsafeupload/servercheck.php\nContent-Type: application/x-www-form-urlencoded\n\nsubmit=submit&uploadfile=<RUNTIME_CANARY>` |
| POST | /vul/unserilization/unser.php | o | preflight_only | `POST <LOOPBACK_ORIGIN>/vul/unserilization/unser.php\nContent-Type: application/x-www-form-urlencoded\n\no=<RUNTIME_CANARY>` |
| POST | /vul/xss/xss_stored.php | message,submit | preflight_only | `POST <LOOPBACK_ORIGIN>/vul/xss/xss_stored.php\nContent-Type: application/x-www-form-urlencoded\n\nmessage=<RUNTIME_CANARY>&submit=submit` |
| POST | /vul/xss/xssblind/xss_blind.php | content,name,submit | preflight_only | `POST <LOOPBACK_ORIGIN>/vul/xss/xssblind/xss_blind.php\nContent-Type: application/x-www-form-urlencoded\n\ncontent=<RUNTIME_CANARY>&name=<RUNTIME_CANARY>&submit=submit` |
| POST | /vul/xss/xsspost/post_login.php | password,submit,username | preflight_only | `POST <LOOPBACK_ORIGIN>/vul/xss/xsspost/post_login.php\nContent-Type: application/x-www-form-urlencoded\n\npassword=<RUNTIME_CANARY>&submit=submit&username=<RUNTIME_CANARY>` |
| POST | /vul/xxe/xxe_1.php | submit,xml | preflight_only | `POST <LOOPBACK_ORIGIN>/vul/xxe/xxe_1.php\nContent-Type: application/x-www-form-urlencoded\n\nsubmit=submit&xml=<RUNTIME_CANARY>` |
| GET | /vul/csrf/csrfget/csrf_get_login.php | password,submit,username | preflight_only | `GET <LOOPBACK_ORIGIN>/vul/csrf/csrfget/csrf_get_login.php?password=<RUNTIME_CANARY>&submit=submit&username=<RUNTIME_CANARY>` |
| GET | /vul/csrf/csrfpost/csrf_post_login.php | password,submit,username | preflight_only | `GET <LOOPBACK_ORIGIN>/vul/csrf/csrfpost/csrf_post_login.php?password=<RUNTIME_CANARY>&submit=submit&username=<RUNTIME_CANARY>` |
| GET | /vul/csrf/csrftoken/token_get_login.php | password,submit,username | preflight_only | `GET <LOOPBACK_ORIGIN>/vul/csrf/csrftoken/token_get_login.php?password=<RUNTIME_CANARY>&submit=submit&username=<RUNTIME_CANARY>` |
| GET | /vul/dir/dir_list.php | title | completed_projection_only | `GET <LOOPBACK_ORIGIN>/vul/dir/dir_list.php?title=<RUNTIME_CANARY>` |
| GET | /vul/fileinclude/fi_local.php | filename,submit | preflight_only | `GET <LOOPBACK_ORIGIN>/vul/fileinclude/fi_local.php?filename=<RUNTIME_CANARY>&submit=submit` |
| GET | /vul/fileinclude/fi_remote.php | filename,submit | preflight_only | `GET <LOOPBACK_ORIGIN>/vul/fileinclude/fi_remote.php?filename=<RUNTIME_CANARY>&submit=submit` |
| GET | /vul/infoleak/findabc.php | password,submit,username | preflight_only | `GET <LOOPBACK_ORIGIN>/vul/infoleak/findabc.php?password=<RUNTIME_CANARY>&submit=submit&username=<RUNTIME_CANARY>` |
| GET | /vul/sqli/sqli_blind_b.php | name,submit | completed_projection_only | `GET <LOOPBACK_ORIGIN>/vul/sqli/sqli_blind_b.php?name=<RUNTIME_SQL_SHAPE>&submit=submit` |
| GET | /vul/sqli/sqli_blind_t.php | name,submit | completed_projection_only | `GET <LOOPBACK_ORIGIN>/vul/sqli/sqli_blind_t.php?name=<RUNTIME_SQL_SHAPE>&submit=submit` |
| GET | /vul/sqli/sqli_iu/sqli_login.php | password,submit,username | preflight_only | `GET <LOOPBACK_ORIGIN>/vul/sqli/sqli_iu/sqli_login.php?password=<RUNTIME_CANARY>&submit=submit&username=<RUNTIME_CANARY>` |
| GET | /vul/sqli/sqli_search.php | name,submit | completed_projection_only | `GET <LOOPBACK_ORIGIN>/vul/sqli/sqli_search.php?name=<RUNTIME_SQL_SHAPE>&submit=submit` |
| GET | /vul/sqli/sqli_str.php | name,submit | completed_projection_only | `GET <LOOPBACK_ORIGIN>/vul/sqli/sqli_str.php?name=<RUNTIME_SQL_SHAPE>&submit=submit` |
| GET | /vul/sqli/sqli_x.php | name,submit | completed_projection_only | `GET <LOOPBACK_ORIGIN>/vul/sqli/sqli_x.php?name=<RUNTIME_SQL_SHAPE>&submit=submit` |
| GET | /vul/ssrf/ssrf_curl.php | url | preflight_only | `GET <LOOPBACK_ORIGIN>/vul/ssrf/ssrf_curl.php?url=<RUNTIME_CANARY>` |
| GET | /vul/ssrf/ssrf_fgc.php | file | preflight_only | `GET <LOOPBACK_ORIGIN>/vul/ssrf/ssrf_fgc.php?file=<RUNTIME_CANARY>` |
| GET | /vul/unsafedownload/execdownload.php | filename | preflight_only | `GET <LOOPBACK_ORIGIN>/vul/unsafedownload/execdownload.php?filename=<RUNTIME_CANARY>` |
| GET | /vul/urlredirect/urlredirect.php | url | completed_projection_only | `GET <LOOPBACK_ORIGIN>/vul/urlredirect/urlredirect.php?url=<RUNTIME_CANARY>` |
| GET | /vul/xss/xss_01.php | message,submit | completed_projection_only | `GET <LOOPBACK_ORIGIN>/vul/xss/xss_01.php?message=<RUNTIME_CANARY>&submit=submit` |
| GET | /vul/xss/xss_02.php | message,submit | completed_projection_only | `GET <LOOPBACK_ORIGIN>/vul/xss/xss_02.php?message=<RUNTIME_CANARY>&submit=submit` |
| GET | /vul/xss/xss_03.php | message,submit | completed_projection_only | `GET <LOOPBACK_ORIGIN>/vul/xss/xss_03.php?message=<RUNTIME_CANARY>&submit=submit` |
| GET | /vul/xss/xss_04.php | message,submit | completed_projection_only | `GET <LOOPBACK_ORIGIN>/vul/xss/xss_04.php?message=<RUNTIME_CANARY>&submit=submit` |
| GET | /vul/xss/xss_dom_x.php | text | completed_projection_only | `GET <LOOPBACK_ORIGIN>/vul/xss/xss_dom_x.php?text=<RUNTIME_CANARY>` |
| GET | /vul/xss/xss_reflected_get.php | message,submit | completed_projection_only | `GET <LOOPBACK_ORIGIN>/vul/xss/xss_reflected_get.php?message=<RUNTIME_CANARY>&submit=submit` |
| POST | /pkxss/pkxss_install.php | submit | preflight_only | `POST <LOOPBACK_ORIGIN>/pkxss/pkxss_install.php\nContent-Type: application/x-www-form-urlencoded\n\nsubmit=submit` |
| POST | /vul/burteforce/bf_client.php | password,submit,username,vcode | preflight_only | `POST <LOOPBACK_ORIGIN>/vul/burteforce/bf_client.php\nContent-Type: application/x-www-form-urlencoded\n\npassword=<RUNTIME_CANARY>&submit=submit&username=<RUNTIME_CANARY>&vcode=<RUNTIME_CANARY>` |
| POST | /vul/burteforce/bf_form.php | password,submit,username | preflight_only | `POST <LOOPBACK_ORIGIN>/vul/burteforce/bf_form.php\nContent-Type: application/x-www-form-urlencoded\n\npassword=<RUNTIME_CANARY>&submit=submit&username=<RUNTIME_CANARY>` |
| POST | /vul/burteforce/bf_server.php | password,submit,username,vcode | preflight_only | `POST <LOOPBACK_ORIGIN>/vul/burteforce/bf_server.php\nContent-Type: application/x-www-form-urlencoded\n\npassword=<RUNTIME_CANARY>&submit=submit&username=<RUNTIME_CANARY>&vcode=<RUNTIME_CANARY>` |
| POST | /vul/burteforce/bf_token.php | password,submit,token,username | preflight_only | `POST <LOOPBACK_ORIGIN>/vul/burteforce/bf_token.php\nContent-Type: application/x-www-form-urlencoded\n\npassword=<RUNTIME_CANARY>&submit=submit&token=<RUNTIME_CANARY>&username=<RUNTIME_CANARY>` |
| POST | /vul/overpermission/op1/op1_login.php | password,submit,username | preflight_only | `POST <LOOPBACK_ORIGIN>/vul/overpermission/op1/op1_login.php\nContent-Type: application/x-www-form-urlencoded\n\npassword=<RUNTIME_CANARY>&submit=submit&username=<RUNTIME_CANARY>` |
| POST | /vul/overpermission/op2/op2_login.php | password,submit,username | preflight_only | `POST <LOOPBACK_ORIGIN>/vul/overpermission/op2/op2_login.php\nContent-Type: application/x-www-form-urlencoded\n\npassword=<RUNTIME_CANARY>&submit=submit&username=<RUNTIME_CANARY>` |
| POST | /vul/rce/rce_eval.php | submit,txt | preflight_only | `POST <LOOPBACK_ORIGIN>/vul/rce/rce_eval.php\nContent-Type: application/x-www-form-urlencoded\n\nsubmit=submit&txt=<RUNTIME_CANARY>` |
| POST | /vul/rce/rce_ping.php | ipaddress,submit | preflight_only | `POST <LOOPBACK_ORIGIN>/vul/rce/rce_ping.php\nContent-Type: application/x-www-form-urlencoded\n\nipaddress=<RUNTIME_CANARY>&submit=submit` |
| POST | /vul/sqli/sqli_del.php | message,submit | preflight_only | `POST <LOOPBACK_ORIGIN>/vul/sqli/sqli_del.php\nContent-Type: application/x-www-form-urlencoded\n\nmessage=<RUNTIME_CANARY>&submit=submit` |
| POST | /vul/sqli/sqli_header/sqli_header_login.php | password,submit,username | preflight_only | `POST <LOOPBACK_ORIGIN>/vul/sqli/sqli_header/sqli_header_login.php\nContent-Type: application/x-www-form-urlencoded\n\npassword=<RUNTIME_CANARY>&submit=submit&username=<RUNTIME_CANARY>` |
| POST | /vul/sqli/sqli_id.php | id,submit | completed_projection_only | `POST <LOOPBACK_ORIGIN>/vul/sqli/sqli_id.php\nContent-Type: application/x-www-form-urlencoded\n\nid=<RUNTIME_SQL_SHAPE>&submit=submit` |
| POST | /vul/sqli/sqli_widebyte.php | name,submit | completed_projection_only | `POST <LOOPBACK_ORIGIN>/vul/sqli/sqli_widebyte.php\nContent-Type: application/x-www-form-urlencoded\n\nname=<RUNTIME_SQL_SHAPE>&submit=submit` |
| POST | /vul/unsafeupload/clientcheck.php | submit,uploadfile | preflight_only | `POST <LOOPBACK_ORIGIN>/vul/unsafeupload/clientcheck.php\nContent-Type: application/x-www-form-urlencoded\n\nsubmit=submit&uploadfile=<RUNTIME_CANARY>` |
| POST | /vul/unsafeupload/getimagesize.php | submit,uploadfile | preflight_only | `POST <LOOPBACK_ORIGIN>/vul/unsafeupload/getimagesize.php\nContent-Type: application/x-www-form-urlencoded\n\nsubmit=submit&uploadfile=<RUNTIME_CANARY>` |
| POST | /vul/unsafeupload/servercheck.php | submit,uploadfile | preflight_only | `POST <LOOPBACK_ORIGIN>/vul/unsafeupload/servercheck.php\nContent-Type: application/x-www-form-urlencoded\n\nsubmit=submit&uploadfile=<RUNTIME_CANARY>` |
| POST | /vul/unserilization/unser.php | o | preflight_only | `POST <LOOPBACK_ORIGIN>/vul/unserilization/unser.php\nContent-Type: application/x-www-form-urlencoded\n\no=<RUNTIME_CANARY>` |
| POST | /vul/xss/xss_stored.php | message,submit | preflight_only | `POST <LOOPBACK_ORIGIN>/vul/xss/xss_stored.php\nContent-Type: application/x-www-form-urlencoded\n\nmessage=<RUNTIME_CANARY>&submit=submit` |
| POST | /vul/xss/xssblind/xss_blind.php | content,name,submit | preflight_only | `POST <LOOPBACK_ORIGIN>/vul/xss/xssblind/xss_blind.php\nContent-Type: application/x-www-form-urlencoded\n\ncontent=<RUNTIME_CANARY>&name=<RUNTIME_CANARY>&submit=submit` |
| POST | /vul/xss/xsspost/post_login.php | password,submit,username | preflight_only | `POST <LOOPBACK_ORIGIN>/vul/xss/xsspost/post_login.php\nContent-Type: application/x-www-form-urlencoded\n\npassword=<RUNTIME_CANARY>&submit=submit&username=<RUNTIME_CANARY>` |
| POST | /vul/xxe/xxe_1.php | submit,xml | preflight_only | `POST <LOOPBACK_ORIGIN>/vul/xxe/xxe_1.php\nContent-Type: application/x-www-form-urlencoded\n\nsubmit=submit&xml=<RUNTIME_CANARY>` |

projection-only rows do not establish a vulnerability. Typed family oracle, fresh replays and matched negative controls are required before any training promotion.
