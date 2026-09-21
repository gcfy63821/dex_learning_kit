# Wave 手：历史与 elastomer 标定

> **本文大部分内容已被取代。** 资产的当前做法见 **[`assets/ASSETS.md`](../assets/ASSETS.md)**。
> 这里只保留两件仍有用的东西：为什么换手、以及 elastomer collision-shape id 的标定方法。
> 更新: 2026-09-16。

## 1. 为什么从 HA4 换到 Wave

上一代 **Sharpa HA4** 手的网格带有保密标记，不能随开源库发布；
新 **Sharpa Wave** 手有公开仓库（`github.com/sharpa-robotics/sharpa-urdf-usd-xml`，Apache-2.0）。

兼容性上二者几乎等价：**22 个受控关节名、link 名完全一致**（根 link 都是 `{side}_hand_C_MC`），
所以 pipeline 里所有按名字寻址的依赖都不受影响。运动学差异实测在**指尖 0.2 mm** 量级，
关节限位完全相同 —— 差别在几何/动力学（碰撞网格、惯量、elastomer 形状），不在运动学。

## 2. 已废弃：预转换的合并 USD

2026-07 曾用一个临时脚本把 FR3 臂与 Wave 手在 USD 层 `stage.Flatten()` 成
`{Right,Left}_Wave_combined.usd`（各 ~26 MB）。该路线已于 2026-09 废弃，原因：

- 那份合并 USD 里**臂完全没有碰撞几何**（26 个碰撞 prim 全在手上，臂 0 个）；
- 合并脚本是临时文件，未入版本控制，已遗失，产物无法复现；
- 预转换 USD 加上游包共 250 MB。

现在改为提交**合并 URDF**（`scripts/build_merged_urdf.py` 生成、可复现、有回归门禁），
由 Isaac Lab 在首次 spawn 时转换并缓存。资产降到 49 MB，臂的碰撞体由转换器统一生成。

## 3. elastomer collision-shape id 标定

摩擦域随机化要给 5 个指尖 elastomer 施加"软"摩擦，靠 **shape 索引**寻址
（`env_cfg.material_elastomer_ids`，消费点在 `franka_sharpa_env.py`）。

**不能靠数 USD 里的碰撞 prim**：PhysX 的凸分解会使 shape 数 ≠ CollisionAPI prim 数，
USD 里也没有可区分 elastomer 的摩擦授权信息。

### 工具：`tools/calibrate_elastomer_ids.py`

探针法：对每个 elastomer body 用它自己的 physx rigid-body view 设一个标记摩擦值，
再读全局 material array，看哪几行变了 —— 那几行就是该 body 的 shape。

```bash
python tools/calibrate_elastomer_ids.py --side right --headless
# -> material_elastomer_ids = [...]
```

⚠️ **切换资产后必须重标**。id 是相对当前机器人资产的 shape 布局而言的：

| 资产 | n_shapes | elastomer ids |
|---|---|---|
| 旧 HA4 预转换 USD | 34 | `[27,28,30,32,33]` |
| 已废弃的预转换 Wave 合并 USD | 64 | `[57,58,60,62,63]` |
| **当前：合并 URDF → 转换 USD** | **34** | **`[27,28,30,32,33]`**（2026-09-16 实测） |

当前管线产出 34 个 shape（臂 8 + 手 26），与旧 HA4 资产相同，所以 id 也回到了那一组。
曾经一度把默认值设成 64-shape 时期的 `[57,58,60,62,63]`，5 个 id 全部越界、被静默丢弃，
摩擦 DR 完全没生效 —— 这正是下面那道过滤的危险之处。

`franka_sharpa_env.py` 里有 `_elast_ids = [i for i in ids if i < n_shapes]` 这道过滤：
**若 id 全部越界，摩擦 DR 会静默失效而不报错**。所以重标之前不要假设它在生效。

## 4. 其他遗留

- **左手全 env 训练未测**：示例数据只有右手 demo。有左手 retarget 数据后可补测。
- ~~未重新加旧资产的碰撞过滤 `filteredPairs`~~ —— **已修复，且这曾是一个 57pp 的静默回归**。
  旧资产排除了 7 对自碰撞（手掌 ↔ 四指近节/拇指掌骨等）；这些 link 静止时就相互接触，
  少了过滤 + `self_collision=True` 时手指合不拢，teacher 成功率从 ~95% 掉到 37.9%，
  且**没有任何 NaN 或告警**。URDF 表达不了碰撞过滤，Isaac Lab 的 `UrdfConverterCfg` 也没有该字段，
  现由 `franka_sharpa_env_cfg.SELF_COLLISION_FILTER_PAIRS` 在转换后写入 USD。详见 `assets/ASSETS.md`。
