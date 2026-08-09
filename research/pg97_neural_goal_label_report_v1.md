# PG-97 神经自动目标/标签解码

状态：`blocked`；架构：`token_presence_autoencoder_plus_two_means_kmeans`；设备：`cuda`。

seed holdout 召回：`1.0`；误报：`0`；未知族严格弃权：`False`。

打乱对照召回：`1.0`；阻塞项：unknown_family_strict_abstain。

该模型只提出通用的观测目标/标签，不输出漏洞结论，不更新 active checkpoint 或长期记忆。
