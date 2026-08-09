# PG-PK-14 SQL v2 族外 active holdout

样本请求：117/135；target：3；seed：3；complete pair：45。

typed oracle pair：45；decoder 直接 pair：45；abstain 后 fallback 找回：0；decoder abstain：0；false positive ledger：0。

v2 本地门：`quarantine`；跨 v1+v2 source 门：`quarantine`；source hash：2（同一 source hash 的 variant 不计作独立数据集）。

v2 改变 endpoint、参数名、响应 JSON 形状和 fragment 命名；server 不执行 SQL、不访问数据库、不进行真实 sleep。decoder 只能提供 active prior，fallback typed AST oracle 才能接受 pair。
