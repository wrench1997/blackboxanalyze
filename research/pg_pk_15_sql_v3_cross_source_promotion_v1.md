# PG-PK-15 SQL v3 第三 source 跨源晋升

请求：135/135；target：3；seed：3；complete pair：63。

typed oracle pair：54；decoder 直接 pair：45；abstain 后 fallback：9；decoder abstain：18；incomplete pair：0；false positive ledger：0。

v3 本地门：`quarantine`；v1+v2+v3 跨 source 门：`promote`；source hash：3。

v3 改变 endpoint、参数名、响应协议和抽象 fragment 命名；server 不执行 SQL、不访问数据库、不进行真实 sleep。decoder/shared router 只能提供 active prior，typed AST oracle 才能作为正证据。
