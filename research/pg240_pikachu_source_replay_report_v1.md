# PG-240 Pikachu upstream-source replay

source_commit=5e1e8d9d14a3ba61d62f28cf35531c4df4dd24fc; fresh=14; GET=10; POST=4
AI=14; reference=14; negative=14; result_fixture=8; typed_effect=8

这是应用源码跨实现评估，不是两个独立后端运行时的证明；共享 PG-214 修复运行层。每路由 fresh reset、正负对照、typed/result oracle 全部满足前才可标记 confirmed_positive；本轮数据默认 evaluation-only，不进入训练或长期记忆。
