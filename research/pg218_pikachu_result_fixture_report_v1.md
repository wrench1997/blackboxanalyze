# PG-218 Pikachu result fixture

device=cuda:0; fresh=14; GET=10; POST=4
known positive record=12; negative clean=14; result fixture verified=8; typed effect=8

已知记录与负对照只是验证结果 oracle 可用，不是注入 payload；所有请求仍是本地只读、每路由 fresh reset，原始值/响应不落盘。
