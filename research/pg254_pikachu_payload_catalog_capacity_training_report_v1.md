# PG-254 Pikachu payload-catalog capacity training

catalog train=8; catalog holdout=7; total train=360; total holdout=47
final_judge=candidate_eligible_for_next_preprobe_replay

只训练抽象 safe-probe gate；payload wire 与 SQL/XSS oracle 结果不进入输入，真实发送仍由 PG-253/PG-250 回放验证。
