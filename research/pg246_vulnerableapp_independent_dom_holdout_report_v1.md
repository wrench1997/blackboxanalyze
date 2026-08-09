# PG-246 VulnerableApp independent DOM holdout

device=cuda; seeds=3; initial=18; fresh=36; GET=12; POST=6
AI sends=12; typed positives=6; secure false accepts=0; POST-405 abstain=6; replay matches=18
route positive recall=1.0; secure false accepts=0; POST abstain recall=1.0

VulnerableApp 是独立于 Pikachu 的 Java/Spring 实现；模型只选择抽象 DOM 通道，运行时绑定器使用临时 DOM marker。wire 只 stdout 显示，持久化为投影、哈希、失败归因、修复链和 fresh replay。
