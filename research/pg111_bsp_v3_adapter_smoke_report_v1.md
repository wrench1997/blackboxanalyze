# PG-111 Python BSP v3 结构与回放 smoke

状态：`passed_bsp_v3_structure_adapter_smoke`。本轮把 BSP v3 的 Page/Node/Expert 结构契约和 NumPy Python reference core 迁入 `blackboxanalyze`，不读取 dog 项目的权重，也不启动普通话基础训练。

- contract SHA-256: `baceba05f8bafa38a3493aff55cc591a9195b99197d622e09c42f11cab12c9f4`
- replay SHA-256: `458f5171066eada4449ce5d517053a14dbd741fa09d4fe2c47606a2994994401`
- lineage: `68b2c853305ae5317d23c45b18ce6125c93240c371cdd19be8c87674c15ba3ae`（fresh；旧 checkpoint 拒绝）
- Python split forward 最大误差：`0.000e+00`；merge round-trip 最大误差：`8.674e-19`；page mass 最大误差：`0.000e+00`。
- CPU/CUDA 只比较结构签名，不宣称数值 forward parity。
- Rule IR 仍等待 evaluator-only typed oracle；不生成训练样本、不写长期记忆。
