# PG-70 Trace decision head

device=cuda；train=4；dev=4；unknown holdout=8。
dev confirm recall=0.0；unknown misname=0；unknown strict abstain=True。

capability gate: `blocked`；training promotion: `false`；memory promotion: `false`。

checkpoint: `artifacts\pg70-trace-abstention\trace_decision_head.pt`
report: `research\pg70_trace_abstention_head_report_v1.json`
protocol: `research\pg70_trace_abstention_head_protocol_v1.json`
