# PG-268A Pikachu browser parameterized crawl

pages=62; routes=105; GET=85; POST=20; forms=38
with_parameter_context=43; missing=62; elapsed=26.984s
本阶段只做同源 DOM 发现，不提交表单；PG-268B 才逐路 fresh reset 回放并记录参数化响应、302 链和有限 oracle。
