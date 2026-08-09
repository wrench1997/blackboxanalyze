# PG-220 live shadow replay

device=cuda; fresh=4; GET=2; POST=2
AI=4; reference=4; typed=4; result_fixture=4
shadow action match=12/12; gated unsafe=0

该轮在全新本地容器上验证 PG-219 的 shadow 过程判断；shadow 不接管网络，也不把本轮结果直接晋升训练或长期记忆。
