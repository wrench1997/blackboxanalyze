# PG-PK-22 oracle projection binding 实验

状态：`pass`；通过：6/6。

XSS、SQL、logic/access 三类未篡改 pair 均保持 accepted；只修改顶层 oracle projection 而不改变已哈希 evidence 的 pair 均被拒绝。
