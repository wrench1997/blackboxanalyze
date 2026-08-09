# PG-56 因果 Trace Transformer 预训练基线

设备：`cuda`；train/dev/holdout：`322/188/120`；词表：`102`。
盲测 token loss/accuracy：`0.6974` / `0.9115`。
盲测 next-action accuracy：`1.0000`；oracle modality：`1.0000`；outcome：`1.0000`。
该结果只证明抽象轨迹预测基线已可复现；尚未证明未见漏洞族 Rule IR 泛化，也不会晋升训练集或长期记忆。
