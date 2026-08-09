# Response intervention v1：失败的工程诊断

这次运行不计入干预成功。它发现了两个协议错误：

1. Juice Shop 的 `HEAD /metrics` 也会触发 `exposedMetricsChallenge`，因此不能把 HEAD 视为天然无副作用。
2. 仅按“`text/plain` + 非小长度”判断运营指标，把 `.well-known/security.txt` 错判为 `observability`；这是观察投影过粗造成的假阳性。

修正内容：按通用 Prometheus 媒体类型 `text/plain; version=0.0.4` 区分指标面；每一步记录评估器状态变化；结果另存为 v2，不覆盖本次失败证据。
