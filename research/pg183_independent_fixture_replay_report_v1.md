# PG-183 independent implementation replay

model=moe_large; surfaces=3; sent=15; candidate=6

冻结 Pikachu 模型在独立实现上只做安全 canary 复放；没有 typed oracle，因此不生成漏洞阳性。
