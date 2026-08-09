# PG-186 Pikachu DOM capacity × encoding replay

models=6; episodes=36; sent=180; candidates=72; typed_surface_effects=12

冻结 small/medium/MoE、双 seed 模型在只读 GET 表面上复放多编码 inert DOM 探针；不训练目标 trace，不把 DOM effect 当漏洞阳性。
