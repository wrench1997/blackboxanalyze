# PG-PK-12 异构表面 + 编码 + seed 留出

样本：108；target：3；seed：3；共享路由 XSS 候选：0；非属性误报候选：0；正向 abstain：18。

typed oracle pair：9；其中路由 abstain 后由族特异 fallback 找回：9；反事实表面拒绝：45；长期记忆门：`quarantine`。

共享路由只提供候选/主动 prior；abstain 不等于停止探测。XML、JSON、文本和响应头回显 marker 不可替代 HTML attribute sink oracle；同一 fixture 的 variant 不计作独立数据集。
