# PG-239 alternate Pikachu GET/POST replay

image=tavenli/pikachu-labs@sha256:b32c7362bb102091bd4ef09c5c571db146bd57469d5598f0c8681ffeeb7907fe; fresh=14; GET=10; POST=4
AI=0; reference=14; negative=14; oracle_available=0; abstain=14

tavenli 镜像可返回 HTTP 页面，但容器内 PHP/SQL oracle 不可用；这是 environment_failure，不是漏洞阴性，也不进入训练或长期记忆。
