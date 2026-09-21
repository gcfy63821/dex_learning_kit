# 资产来源与许可（provenance）

机器人为 **Franka FR3 臂 (7 DOF) + Sharpa Wave 手 (22 DOF)**，共 29 DOF。
上一代 Sharpa HA4 手的资产**已不随库发布**（其网格带有保密标记）。

本目录放**两个公开上游模型**、由它们生成的**合并 URDF**，以及由该 URDF 构建的**机器人 USD**。
改动资产前请先读这里。

## 一览

| 路径 | 来源 | 许可 | 谁在用 |
|---|---|---|---|
| `franka_fr3/` | Franka FR3 description（URDF + meshes） | Apache-2.0（见 `LICENSE`） | 合并 URDF 的臂侧输入 |
| `sharpa_wave/` | `github.com/sharpa-robotics/sharpa-urdf-usd-xml` 的左右手子集 | Apache-2.0（见 `LICENSE.txt` / `NOTICE.txt`） | 合并 URDF 的手侧输入；`hand_imitation/envs/sharpa.py` 的 retarget FK |
| `generated/fr3_with_{right,left}_sharpa_wave.urdf` | **生成物**，`scripts/build_merged_urdf.py` | 随本库 (MIT) | 机器人 USD 的构建输入；retarget 与 deploy v3 的 `pk` 运动链 |
| `robot/fr3_with_{right,left}_sharpa_wave/` | **生成物**，`scripts/build_robot_usd.py`（各 ~26 MB） | 随本库 (MIT) | **机器人 spawn 源** |

生成物已提交入库，所以训练不需要构建步骤。改了任一输入后重新运行：

```bash
python scripts/build_merged_urdf.py --side both
python scripts/check_asset_equivalence.py --reference_usd <旧 USD> --side right   # 可选回归门禁
```

## URDF 与 USD 的分工

两个都入库，各司其职：

- **URDF**（84 KB，可读可 diff）是**源**。改机器人从这里改。
- **USD**（各 ~26 MB）是**实际 spawn 的产物**，由 `scripts/build_robot_usd.py` 从 URDF 构建。

之所以把 USD 也入库，不是为了省一次转换时间，而是因为**下一节的自碰撞过滤对
无法用 URDF 表达**，只能在转换后写到 USD 上。若每次运行时重建，任何绕过
`franka_sharpa_robot_usd()` 直接 spawn URDF 的代码路径都会静默丢掉它们
——`scripts/retarget.py` 就曾经这样丢过一次。固化成资产后，语义在产物里。

改了 URDF 之后**必须重新构建 USD**：

```bash
python scripts/build_merged_urdf.py --side both     # 改了上游模型时
python scripts/build_robot_usd.py  --side both      # 改了 URDF 之后，必跑
```

构建产物里的 `.source_hash` 记录来源 URDF 的 sha256。`shipped_robot_usd()` 每次加载时比对，
不一致会**大声告警**（但不中断）——资产与源悄悄不同步正是这套代码反复踩过的失效模式。

> 没有构建产物时会回退到运行时转换（同样应用过滤对），所以不带 USD 的 checkout 也能跑，
> 只是首次慢、且依赖本地 Isaac Lab 版本产出相同结果。回退缓存目录由 `usd_cache_dir()` 钉住，
> 可用 `DEXX_USD_CACHE` 覆盖；缓存共享，并行开多进程前先用单 env 预热一次。

体积：上游 49 MB + USD 产物 52 MB ≈ 98 MB（相比最初的 250 MB 预转换资产仍减少六成），
最大单文件 26 MB，未触及 GitHub 的大文件限制。

## 自碰撞过滤：URDF 丢不得的语义

旧的预转换资产显式排除了 **7 对**自碰撞（手掌 `hand_C_MC` ↔ 食指/中指/无名指/小指近节
与拇指掌骨；小指掌骨 ↔ 小指近节；拇指掌骨 ↔ 拇指近节），共 14 条有向记录。
这些 link 在静止姿态下就相互接触。

**URDF 格式表达不了碰撞过滤，Isaac Lab 的 `UrdfConverterCfg` 也没有对应字段**
（只有一个 `self_collision` 布尔开关）。所以转换产物必然丢掉它们，而我们的
`enabled_self_collisions=True` 会让手掌顶住手指根部 —— **手指物理上合不拢**。

失败是**完全静默**的：没有 NaN、没有告警、没有限位越界，episode 存活步数正常，
物体就是一动不动。实测（shipped teacher，cube_small @5cm，256 ep）：

| 配置 | 成功率 |
|---|---|
| 旧预转换 USD + 旧增益（基线） | 95.3% |
| 旧预转换 USD + 现在的显式增益 | 93.0% |
| URDF 转换，**无**过滤对 | **37.9%** |
| URDF 转换，关闭自碰撞（诊断用） | 94.9% |
| URDF 运行时转换 + 写回过滤对 + 自碰撞开启 | 93.8% |
| **入库 USD 资产（当前默认）** | **94.9%** |

修复在 `franka_sharpa_env_cfg.py`：`SELF_COLLISION_FILTER_PAIRS` +
`franka_sharpa_robot_usd()`。后者显式驱动 `UrdfConverter`（保留懒转换与
`.asset_hash` 缓存），转换后用 `UsdPhysics.FilteredPairsAPI` 双向写入，再由
`UsdFileCfg` spawn。

⚠️ **必须写进 USD 文件，不能 spawn 后补**：`spawn_from_urdf` 带 `@clone` 装饰器，
克隆到各 env 已经发生，事后给源 prim 打补丁不会传播。

⚠️ 换手或改动 link 命名后要复核这 7 对是否仍然正确。`_apply_self_collision_filters()`
找不到对应 prim 时会直接抛错，不会静默跳过。

> 教训：资产等价性**不能只靠静态比对**。本次两套资产在增益、armature、力矩上限、
> 关节限位、碰撞近似、质量、惯量七个维度上逐项零差异，唯一差别就是这组过滤对，
> 而它造成了 57pp 的性能差。`check_asset_equivalence.py` 只能验运动学，
> 换资产后务必对着已知 checkpoint 跑一次成功率回归。

## armature 是承重项，不要删

URDF 无法表达转子惯量，所以 URDF 导入的手 **armature = 0**。手指 link 的惯量在
1e-6 kg·m² 量级，0.2 N·m 的力矩就足以让关节在一个 1/120 s 步内**直接穿透 PhysX 限位**。
实测：零动作跑 100 步后 29 个关节里有 14 个跑出限位，其中一个在 `[0, 1.75]` 限位下到了 15 rad。

因此增益/armature/摩擦全部由 `src/dexx/robot_constants.py` 的 `HAND_GAINS`、
`ARM_ARMATURE`、`ARM_FRICTION` 显式注入 actuator cfg，**不再从资产继承**。
数值取自厂商 USD（已从 USD 的“每度”驱动约定换算，×180/π）。

臂的 stiffness/damping 不在那里——它们留在 `franka_sharpa_env_cfg.py`，
用的是 2026-04-23 阶跃响应标定的那一套，本库所有 checkpoint 都是对着它训的。

## 挂载变换：标定值，不是调参旋钮

手通过 fixed joint 接在 `fr3_link7` 上，`xyz = 0 0 0.142`，yaw 右手 `+3π/4` / 左手 `−π/4`。
这是对着旧资产标定出来的：改了它整只手就位移，**每一条 retarget demo 和每一个相机外参都会失效**。

`check_asset_equivalence.py` 是便宜的回归门禁（~0.2 s，不需要模拟器）：

- **右手**：42 个 link 全部对上，最大 0.0006 mm / 0.0003 deg → `EQUIVALENT`
- **左手**：最大 0.0494 mm / 0.0228 deg → 按默认容差报 `MISMATCH`，但这是**有意的修正**：
  旧资产把 yaw 写成四舍五入的 `-0.785`，这里用精确的 `-π/4`。差值 0.023 deg
  在指尖是 0.06 mm，比 retarget 求解残差 1.6 mm 小一个量级，不影响任何 demo。

## 已知问题

`src/dexx/tasks/franka_sharpa/franka_sharpa_force_deploy_env_v2.py` 在 `__init__` 里
无保护地加载 `assets/tactile_ha4_map/*.npy`，该目录从未随本库发布 ——
**构造 deploy env 会 FileNotFoundError**。这是 HA4 时期的触觉传感器 UV map，
Wave 没有对应物，需要重新标定或把该路径改成可降级。

## 第三方许可

两个上游目录均为 Apache-2.0，其 `LICENSE` / `LICENSE.txt` / `NOTICE.txt` 已原样保留，
请勿删除。本库其余部分为 MIT（见根目录 `LICENSE`）。
