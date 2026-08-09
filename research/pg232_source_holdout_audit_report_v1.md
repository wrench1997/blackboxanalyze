# PG-232 strict source-heldout audit

usable=60; source_folds=5; strict_pass=False
pg222_observed_process: holdout=46; lane=0.95652175; repair=0.47826087; self_error_recall=1.0
pg224_real_surface_projection: holdout=3; lane=1.0; repair=1.0; self_error_recall=0.0
pg226_typed_sql_result: holdout=2; lane=1.0; repair=1.0; self_error_recall=0.0
pg227_dom_redirect_surface: holdout=3; lane=1.0; repair=0.66666669; self_error_recall=0.0
pg229_juice_shop_fresh_typed_replay: holdout=6; lane=1.0; repair=0.0; self_error_recall=1.0

任何一个 source 留出折叠不过门，就不提升长期记忆或漏洞结论。
