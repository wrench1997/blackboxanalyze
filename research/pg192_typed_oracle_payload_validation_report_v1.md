# PG-192 typed oracle payload validation

device=cuda; routes=3; typed_positive=1; confirmed=1; training_eligible=False

| surface | typed oracle | confirmed positive | claim allowed |
|---|---|---:|---:|
| pg192_urlredirect | True | True | True |
| dom_unknown | False | False | False |
| sql_unknown | False | False | False |

只有受控 loopback redirect oracle 通过时才形成 typed positive；DOM/SQL evaluator unavailable 时保留 abstain。
