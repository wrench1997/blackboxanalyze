# PG-235 failure-conditioned policy

device=cuda; train=63; unseen_family_holdout=19
selected hidden=128; token=0.69033233; lane=0.84210527; repair=0.57894737; abstain_recall=0.78947368; false_send=4; strict_pass=False

未见 DOM/redirect family 只能选择 abstain；只有 SQL typed+result 才允许 send_candidate 标签。next-token loss 不单独晋级。
