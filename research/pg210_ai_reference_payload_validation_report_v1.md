# PG-210 AI/reference payload validation

device=cuda:0; fresh containers=2; typed XSS GET routes=6
AI candidate sends=12; AI surface effects=4; reference effects=4; effect agreements=12

请求结构视图见 pg210_request_anatomy_view_v1.json：只包含 method/path/字段/编码/哈希，不含运行时值和响应正文。
DOM 双 oracle 只证明非 JS DOM 结构回显；SQL 路由因缺少 Pikachu 后端 AST oracle 保持 abstain。
