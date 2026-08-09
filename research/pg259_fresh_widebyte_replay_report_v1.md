# PG-256 Pikachu wide-byte row oracle

episodes=2; AI sends=2; reference=2
typed_effect=1; confirmed=1; AI classes=['syntax_boundary', 'widebyte_escape_boundary']

AI 先选择抽象 Rule-IR class；失败反馈更新 UCB，再在新的 fresh 容器探索。reference 的宽字节 wire 仅作为独立对照；结果只表示本地只读行差分，不是公网漏洞结论。
