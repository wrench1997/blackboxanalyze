# PG-191 Pikachu crawled surface matrix

device=cuda; matrix=44; selected=8; variant=xxl; sent_get=15; sent_post=4; candidates=3; abstain=5; positives=0

| route id | method | path | fields | GET | POST | candidate | abstain | manifest errors |
|---|---|---|---|---:|---:|---:|---:|---:|
| pg191-surface-001 | GET | /vul/csrf/csrfget/csrf_get_login.php | password,submit,username | 2 | 0 | 0 | 1 | 0 |
| pg191-surface-004 | GET | /vul/dir/dir_list.php | title | 2 | 0 | 0 | 1 | 0 |
| pg191-surface-005 | GET | /vul/fileinclude/fi_local.php | filename,submit | 2 | 0 | 0 | 1 | 0 |
| pg191-surface-009 | GET | /vul/sqli/sqli_blind_t.php | name,submit | 2 | 0 | 0 | 1 | 0 |
| pg191-surface-017 | GET | /vul/urlredirect/urlredirect.php | url | 3 | 0 | 1 | 0 | 0 |
| pg191-surface-018 | GET | /vul/xss/xss_01.php | message,submit | 2 | 0 | 0 | 1 | 0 |
| pg191-surface-026 | POST | /vul/burteforce/bf_form.php | password,submit,username | 1 | 2 | 1 | 0 | 0 |
| pg191-surface-042 | POST | /vul/xss/xssblind/xss_blind.php | content,name,submit | 1 | 2 | 1 | 0 | 0 |

完整爬虫矩阵只保存观测字段与哈希绑定的抽象 probe plan；回放只发送 bounded canary，未知 oracle 一律 abstain。
