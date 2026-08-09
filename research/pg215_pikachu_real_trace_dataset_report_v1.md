# PG-215 Pikachu real trace dataset

episodes=28 (new=14); step rows=112; train=48; holdout=64
GET rows=80; POST rows=32; clean-reset rows=112; docker restart rows=0

这些是固定派生 Pikachu 运行时的真实 HTTP/数据库健康回放，按 prior→negative_control→candidate→recovery 压成 family-free Rule-IR token；不保存原始 payload/response，也不把响应形状当作漏洞标签。
