# PG-PK-16 logic/access v2 族外与跨族 guard

请求：180/180；complete pair：90；logic oracle pair：36。

logic decoder model-only：63；logic control candidate：0；SQL decoder 在 logic surface 上的 injection candidate：0；shared router injection route：0。

SQL 跨族 guard：`pass`；v2 本地 logic/access memory：access_control=quarantine, logic=quarantine；v1+v2 跨 source：access_control=quarantine, logic=quarantine。

v2 更换 route、query 词汇、JSON 字段和响应长度；SQL/shared 输出只作诊断 prior，不具备 logic/access 正向权威。
