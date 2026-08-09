# PG-71 Trace feature drift audit

legacy candidate/control duplicate-label pairs: `3`；observable shape differences: `4`；sparse dims: `254`；dev distance range: `0.0..1767.766968`。

根因：安全 response projection 已存在，但旧 feature extractor 丢掉了关键 bounded shape 差分；同时 per-dimension floor 放大了稀疏漂移。training/memory promotion 均关闭。
