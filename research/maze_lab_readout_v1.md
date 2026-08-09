# 规则迷宫实验读出 v1

日期：2026-08-02  
范围：本地合成规则迷宫 + 已冻结的本地 Juice Shop evaluator 基准

## 这次落地了什么

`app/maze_solver.py` 把一次实验表示成状态图：响应/浏览器/解释器的可观察投影是节点，动作与前后状态指纹是边；同一指纹回到当前路径是回环，没有新状态或不满足不变量的分支是死路。`MazeFrontier` 只消费未探索边，避免在死循环里重复采样。

`app/maze_labs.py` 注册了十二个安全靶场变体：类型混淆越权、Origin 重定向、业务边界、历史重放、DOM Sink/双重解码/template sink，以及 SQL 结构、错误、盲响应、时间、和本地带外代理通道。DOM 与 SQL 靶场均为合成 oracle：前者不执行脚本，后者不连接数据库。

本阶段又把 SQL 结构边界拆成四条独立观测通道：AST/嵌套结构、错误语法、盲响应（布尔/行形状）和有硬预算的时间通道。时间通道只记录确定性的模拟延迟，不调用 `sleep`；错误通道只有在基线差分和复探都成立时才上升为可观察出口。

## 出口判定

单次 `200`、文本反射、页面跳转或响应形状变化都只能是候选信号。认证要求同一受保护资源从拒绝到接受，并且会话状态改变，再以同资源复探；通用规则则按族绑定语义 oracle：

| 族 | 出口所需证据 |
|---|---|
| 认证 | protected resource transition + session signal + 复探 |
| 越权 | protected resource transition + 复探 |
| XSS | browser sink observed + DOM change + 复探 |
| SQL/注入 | controlled differential + interpreter boundary + 复探 |
| 逻辑 | invariant violation + state replay |
| 重定向 | location origin changed |

出口分为 `candidate`、`observable_success`、`evaluator_confirmed`。最后一级只能由 fresh reset 后的 evaluator 提供，不能由模型或 HTTP 启发式自行声称。

## 可复现检查

- 5 个迷宫图/出口单元测试通过：公开 200 不被当作认证出口；认证需同资源、会话信号和复探；重定向与 XSS 反射不能冒充浏览器出口；逻辑出口需重放；图能标出 forward/loop/dead-end。
- 2 个靶场注册测试通过：六个族已覆盖，DOM/SQL 规则是确定性的安全合成 oracle。
- 本次全量工程测试：`70 passed, 1 warning`；这份读出不把单元测试等价成漏洞确认。
- 族外 oracle 读出：8 个族外靶场全部通过安全门禁和可观察复探，evaluator 确认数为 `0`；这是协议层覆盖结果，不是神经模型泛化分数。

## 与 Juice Shop 结果的关系

Loop 12 v6 的真实 evaluator 矩阵此前仍是 hidden family hit `0/7`；这说明当前 HTTP/GET 策略还没有找到新的靶场确认出口。迷宫层解决的是“如何不误判、如何记录死路和回环”，不是凭空提高命中率。下一轮应为 XSS 加受控浏览器 oracle、为 SQL 加结构差分 oracle，再做族外 holdout。
