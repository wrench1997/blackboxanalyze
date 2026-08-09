# PG-248 observable feedback token capacity training

train=117; holdout=133; canary=34; pg230_new=6
variants=[512, 1024, 2048, 4096]; selected hidden=2048; holdout positive=0.96296296; abstain=0.94936709; false_send=4
canary pass=True; final_judge=blocked

输入只增加已观察的反馈投影；期望 oracle、lane、repair 和 payload_grounded_eligible 不进入模型输入。所有 Pikachu 来源仍为实现留出。
