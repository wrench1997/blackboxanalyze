# PG-188 XXL replay/action training

device=cuda; body_parameters=1024; action_train=216; lm_replay=4096

| variant | parameters | holdout accuracy | unknown abstain | false stop | forgetting |
|---|---:|---:|---:|---:|---:|
| frozen_xxl_head | 101260465 | 0.54545455 | 0.0 | 0 | False |
| replay_xxl_low_lr | 101260465 | 0.45454545 | 0.04545455 | 0 | False |
| replay_xxl_strong | 101260465 | 0.46590909 | 0.22727273 | 0 | False |

selected=None; 目标 Pikachu trace 未进入训练，仍需独立回放。
