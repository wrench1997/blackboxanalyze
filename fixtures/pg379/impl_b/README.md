# PG-379 implementation B: Node native HTTP fixture

这是独立于实现 A 的动态网页靶场实现：运行时使用 Node 原生 `http`
单进程，不复用 Python handler、共享页面生成器或 A 的路由答案。它是
evaluator-only fixture，不是生产服务，也不会生成训练行或漏洞结论。

## 运行合同

- 只绑定 `127.0.0.1`，不解析外部主机、不读取凭据、不访问数据库/文件，状态只在进程内存中存在。
- 每个 role 前都应启动新容器/进程；若同一进程内复放，先 `POST /__reset`，再用 `GET /__health` 检查 `state_clean=true` 和新的 `target_instance_digest`。
- Docker 运行必须使用预先审核的镜像摘要、`--network none`、tmpfs 工作目录、无 bind/volume；不要把本 fixture 暴露到公共接口。
- `GET /__manifest` 只给 evaluator 路由形状和安全投影；原始输入、响应体和 evaluator answer 不进入模型上下文。
- 任何样本的 `training_allowed`、`memory_promotion_allowed`、`payload_catalog_promotion_allowed`、`vulnerability_claim_allowed` 都固定为 `false`。

## 十二个动态 route class

实现了六个 GET 与六个 POST：

| method | route class | 输入来源 | 响应形状 |
| --- | --- | --- | --- |
| GET | `get_query_html_text` | query | `html_text` |
| GET | `get_path_dom_text` | path segment | `html_dom_text` |
| GET | `get_fragment_js_navigation` | query | `html_fragment` |
| GET | `get_json_shape` | query | `json_shape` |
| GET | `get_redirect_control` | query | `redirect_shape` |
| GET | `get_failure_feedback` | query | `error_shape` |
| POST | `post_form_dom_update` | form | `html_dom_text` |
| POST | `post_json_state_transition` | JSON | `state_delta` |
| POST | `post_redirect_control` | form | `redirect_shape` |
| POST | `post_attribute_shape` | form | `html_attribute` |
| POST | `post_parser_failure` | JSON | `error_shape` |
| POST | `post_replay_shape` | form | `replay_shape` |

固定 route path 和参数角色以 `manifest.json` 为准。evaluator 使用自身生成的
`PG379B_CANARY_<id>` 安全标记；该标记只会以 escaped/摘要形式出现在响应中。
超长、标记化的过滤类别或缺失输入分别返回有界 `filtered`/`missing` 形状，
不会执行脚本、改变外部状态或发起后续请求。`post_json_state_transition` 的
计数只是内存中的 disposable state，reset/进程重启即清除。

请求形状（参数名是抽象角色名，不是攻击字符串）：

```text
GET  /pg379/b/get-query-html-text?query_text=<safe-canary>
GET  /pg379/b/get-path-dom-text/<safe-canary>
GET  /pg379/b/get-fragment-js-navigation?fragment_identifier=<safe-canary>
GET  /pg379/b/get-json-shape?json_value=<safe-canary>
GET  /pg379/b/get-redirect-control?view_mode=<safe-canary>
GET  /pg379/b/get-failure-feedback?query_term=<safe-canary>
POST /pg379/b/post-form-dom-update       form_field=<safe-canary>
POST /pg379/b/post-json-state-transition JSON {"json_value":"<safe-canary>"}
POST /pg379/b/post-redirect-control      view_mode=<safe-canary>
POST /pg379/b/post-attribute-shape       attribute_value=<safe-canary>
POST /pg379/b/post-parser-failure        JSON {"structured_value":"<safe-canary>"}
POST /pg379/b/post-replay-shape          record_cursor=<safe-canary>
```

上面的占位符只表示 evaluator 运行时安全标记；它们不是固定样本，也不会写入
模型上下文。对 `post_parser_failure`，故意不可解析的 JSON 只用于观察
`parser_error` 反馈，不执行任何副作用。

## 确定性构建/启动

1. 在授权机器上预拉取并记录 `node:20.11.1-alpine3.19` 的 image digest；构建
   时禁止联网（`docker build --network=none --pull=false`），并把最终 image digest 写入
   evaluator attestation。推荐把摘要直接作为 build arg，避免 tag 漂移：

   ```text
   docker build --network=none --pull=false --build-arg NODE_BASE_IMAGE=node:20.11.1-alpine3.19@sha256:<attested-base-digest> -t pg379-impl-b:attested .
   ```
2. 在本目录运行 `node --check server.js`，再用固定 `PORT` 启动：

   ```text
   node server.js
   ```

   容器复放使用 `--network none` 和 tmpfs（不发布公共端口；由授权的
   loopback relay 或容器内 evaluator 连接）：

   ```text
   docker run --rm --network none --tmpfs /tmp:rw,noexec,nosuid,size=16m pg379-impl-b:attested
   ```

3. 仅允许 loopback relay 连接；停止进程后再销毁容器。`package.json` 没有依赖，
   因而不需要 `npm install`，源文件和 manifest 的 SHA-256 应保存到 attestation。

静态 contract test 只读取 `manifest.json`/`server.js`，不会启动 Docker、网络或训练。
