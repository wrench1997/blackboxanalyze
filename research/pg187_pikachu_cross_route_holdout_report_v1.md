# PG-187 Pikachu unseen-route double holdout

models=6; episodes=24; sent=120; candidates=48; typed_surface_effects=12

xss_01/xss_04 是浏览器清单中的真实 GET 参数表面，但未用于 PG-185 回放；同时留出多层编码，检查路由泛化和错误阳性。
