# PG-185 Pikachu DOM surface replay

model=moe_large; routes=2; sent=10; candidates=4; typed_dom_effects=2

模型参与角色选择，控制器只向浏览器清单中的 GET 参数发送不执行脚本的 inert DOM 标记；typed DOM effect 不等于 XSS 阳性。
