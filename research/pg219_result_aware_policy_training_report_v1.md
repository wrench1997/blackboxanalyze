# PG-219 result-aware process policy

device=cuda; train=42; holdout=58; route_holdout=/vul/sqli/sqli_x.php
selected=standard; variants=[('standard', 96, 1.0, 0), ('wide', 192, 1.0, 0), ('deep', 384, 1.0, 0)]

模型只读取 bounded transport/result projection；typed/result oracle 只作为监督目标或上一阶段反馈，不作为当前动作的输入。large body 冻结，adapter 未接管真实发包。
