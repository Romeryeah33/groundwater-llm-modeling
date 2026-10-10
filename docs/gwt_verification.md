# 溶质运移（GWT）链路验证记录

本文记录 GWT 全链路打通后的验证基线、判定依据，以及一处**已知且已接受**的数值伪影。
后续修改这条链路时，请对照本文的基线回归。

- 验证驱动脚本：`scripts/verify_gwt_cases.py`（Case A/B）、`scripts/verify_east_texas.py`（Case C）
- 回归测试：`tests/test_gwt_pipeline.py`（52 项，含两个真实 mf6 算例）
- 环境：Flopy 3.10.0 / mf6 6.3.0（Windows）

---

## 1. 算例与判定结果

### Case A — 一维土柱，对照解析解

网格 `nrow=1, ncol=500, nlay=1`，`delr=0.2`（域长 100 m），两端 CHD（h=1.0/0.0，C=1.0/0.0），
`al=1.0 m`，`perlen=1250 d`，`nstp=2500`。参考流速 `v = 0.04008 m/d`。

| 判据 | 结果 | 阈值 |
|---|---|---|
| `max\|C_num − C_解析\|` | **0.0022** | < 0.01 |
| C=0.25 穿透点相对误差 | 0.116% | < 1% |
| C=0.50 穿透点相对误差 | 0.094% | < 1% |
| C=0.75 穿透点相对误差 | 0.012% | < 1% |
| 域内总质量 vs 纯对流输入 `v·t` | 50.0801 vs 50.1002（0.040%） | < 1% |
| 负浓度 / 超调 | 无 / 无 | — |

**为什么对照"对流型入口"的单项解，而不是经典两项式 Ogata–Banks：**

MF6 的弥散只在**相邻单元之间的内部界面**上计算，外部边界面上没有弥散通量。
所以 CHD 入口供质纯粹是对流项 `Q·C_aux`，不含 `−D·∂C/∂x|₀`。

- 对流型入口（零边界弥散）解析解：`C = 0.5·erfc((x−vt)/(2√(Dt)))`，域内总质量 `= v·t`
- 第一类（Dirichlet）入口的两项式多送进来的质量恰好是 `αL·C0`，表现为前沿整体下移 `αL`

实测佐证（αL=1 m，v·t=50.1 m）：MF6 自身质量预算 `CHD IN = 12.5251 = q·C0·T`，
恰好是对流量、没有任何额外弥散供质；前沿位移恒为 1.00 m，不随推进距离（8→67 m）
和 Δx（0.1/0.2/0.4）变化，且 `δ/αL = 1.01`——是刚性平移而非数值弥散。
`max|C_MF6 − C_单项| = 0.0022`，`max|C_MF6 − C_两项| = 0.0414`。

### Case B — RCH + AUX + AUXMIXED + nlay=2

`nrow=10, ncol=40, nlay=2`，`delr=delc=10`，`top=30, bottom=10`，RCH 覆盖 layer0 的 col 0..9
（`rate=1e-4, conc=500`），`perlen=365 d, nstp=365`。

| 判据 | 结果 |
|---|---|
| `gwt.nam` 含 `MST6` | ✅ |
| GWF/GWT 两份 `.dis` 的 griddata 逐行一致 | ✅ |
| botm 写成 `LAYERED` 分层形式 | ✅ |
| `rch` 声明 `AUXILIARY CONCENTRATION` | ✅ |
| `ssm` SOURCES = `RCHA / AUXMIXED / CONCENTRATION` | ✅ |
| RCH 累计注入 vs 理论值 1.825e5 | 偏差 **0.0000%** |
| GWT 质量离散度 | 0.27% |
| layer1 浓度 > 0（垂向路径连通） | ✅ |
| 负浓度 / 超调 | 无（min=6.5e-13）/ 无 |

### Case C — East Texas 本体

`data/sessions/default.json` 的 IR 有若干物理设计缺陷，直接跑只能得到浓度恒为 0 的
"成功"结果。`scripts/verify_east_texas.py` 在**内存里**补齐后再跑，**不改动用户的 session 文件**：

1. 顶层 `stress_periods` 是裸整数 `10`，真实时间信息在
   `transport.stress_periods = {nper:10, perlen:15000}`。`perlen` 按 MF6 原生语义是
   "每个应力期的长度"，直接用会变成 10×15000=150000 d。脚本显式给 `period_data`：
   10 期 × 1500 d = **15000 d 总计**。
2. `transport` 缺 `al` → 显式给 40.0 m
3. `transport` 缺 `porosity` → 给 0.2
4. 无任何浓度源 → 加一片 81 格 RCH 面源（西北角），浓度 20000 mg/L，`rate=1e-4 m/d`
5. `vka = {"value": 20, "unit": "m"}` 单位错误且是 hk 的 3.6 倍 → 改为 `hk/10 = 0.55 m/d`
6. `hclose` 0.01 → 1e-3
7. 初始水头给 215 m（介于 GHB 水位 212/208 与 IC=230 之间）

网格 103×218×2，跑满 15000 d。

| 判据 | 结果 |
|---|---|
| `gwt.nam` 含 `MST6` | ✅ |
| GWF/GWT botm 逐行一致 | ✅ |
| `rch` 声明 `AUXILIARY CONCENTRATION` | ✅ |
| `ssm` SOURCES 含 `RCHA / AUXMIXED` | ✅ |
| GWT 质量离散度 | 0.32% ~ 0.38%（阈值 1%） |
| 浓度场非零 | ✅ 峰值 7145 mg/L |
| 无超调 | ✅ |
| 污染羽外扩 | ✅ 0 → 558 格，逐帧单调不减 |
| 负浓度 | ⚠️ 见第 2 节 |

浓度峰值的物理合理性核对：补给量 `1e-4 m/d × 15000 d = 1.5 m` 水柱，源格水当量
`27.5 m × 0.2 = 5.5 m`，完全混合下应达 `1.5/5.5 × 20000 ≈ 5455 mg/L`。实测 7145 mg/L，
同量级且偏高（源格还额外接收弥散汇入），合理。

### Case D — 源区定位回归锁（不跑 mf6）

**为什么必须有这一条。** Case B / Case C 都在 IR 里**显式给了 `geometry.grid_indices`**，
走的是 mapper 的优先级 1 直接透传 —— 于是"从 `position: "landfill"` + 上传 SHP 解析出源区"
这条链路**在验证套件里从未被执行过**。而真实 LLM 产出的 IR 恰恰不给 `grid_indices`。

该缺口的实际后果（2026-09-17 定位）：East Texas 会话的 IR 里
`{"type":"rch","name":"landfill_boundary","position":"landfill","parameters":{"rate":1e-4,"concentration":20000}}`
在旧代码下经历三步静默降级 —— `model_mapper` 只读不存在的 `grid_indices`
→ `_infer_edge_indices` 只认方位词、`landfill` 一个都不匹配 → 落到
`elif conc is not None:` 分支，**把点源展开成 layer 0 全网格 22454 格（103×218）**，
只写一句到不了任何 UI 组件的 `logger.info`。

网格 103×218、`delr = delc = 20 m`，故单格 400 m²：

| 量 | 缺陷态 | 正确态 | 倍数 |
|---|---|---|---|
| `.rch` 逐格记录 | 22454（= 103×218 全域） | **132** | 170.1× |
| 启用的格子面积 | 8 981 600 m² | **52 800 m²** | 170.1× |
| 补给总量 | 898.16 m³/d | **5.28 m³/d** | 170.1× |

（填埋场多边形的真实面积 `landfill_arcs.shp` 测得为 54 225 m²，与栅格化后的
52 800 m² 相差 2.6%，是格子离散化的正常差异，不是错误。）

这正是用户看到"污染羽像水流模型结果、从北部整条边界补给"的根因：**不是一个 132 格的
点源在泄漏，而是整个模型域在均质地加浓度**。已在 `app/modeling/source_zone.py` 中修掉，
三重定位（IR `grid_indices` → IR `coordinates`/`bbox` → 上传 SHP 名称匹配）全失败时
**硬拒绝**（`SourceZoneUnresolved`），绝不回落全网格。

**Case D 断言的是写出来的 `.rch` 文件本身**（不是中间对象）——只有落盘文件才是用户真正
跑的东西。用 `build_simulation` + `write_simulation`，不跑 mf6，几秒完成。

```text
.rch 逐格记录数 = 132                    ← 期望 132，不是 22454
row 恰好 52..62 / col 恰好 137..148      ← 集中在填埋场，不是铺满全域
全部落在 layer 0                          ← RCH 不接受跨层
rate 全为 1e-4 m/d / conc 全为 20000 mg/L
总补给 = 5.2800 m³/d（= 132 × 1e-4 × 400 m²）
```

注意 `parse_rch_cells` 把文件里的 1-based cellid **减 1 转成 0-based**，好与
`tests/test_source_zone.py` 的 0-based 锁（row 52..62 / col 137..148）逐字对齐 ——
两处各用一套基准，是最容易被自己的 ±1 骗过去的地方。

**锁的有效性已验证**（不是同义反复）：把 `ET_ORIGIN_X/Y` 故意偏移 500 m，
row/col 断言即红（27..37 / 112..123）；偏移 20 m（恰好一格）时 col 断言仍红（136..147）。
这同时证明 `domain.origin_xy` 是**承重**字段而非装饰。

**端到端实测（`ET_D_RUN=1`，真跑 mf6，2026-09-17）**：

```text
解析：landfill_boundary 132 格, area=54225.0 m², from=shp_name_match:landfill_arcs
.rch：132 条 = row 52..62 × col 137..148, layer 0, rate 1e-4, conc 20000, 合计 5.2800 m³/d
tdis：NPER 10, perioddata 1500 × 10, nstp 10（Courant 从 1 提升）→ 总时长 15000 d, Δt = 150 d
水头：212.03 ~ 215.0 m（含 CHD 215，说明 north_boundary 已生效）
t = 300 d：污染格 277，源区 132 格**全部已污染**，277 格**全部连通到源区**（无孤立污染团）
t = 15000 d：污染格 2289 / 22454 = 10.19%（修复前是 100% 的全域毯子）
```

`Δt = 150 d` 顺带解决了"想看 360 天"落空的问题：输出时刻为 150/300/450…，360 d 落在
300 与 450 之间，单帧交互图可插值。

判据刻意**不判迁移方向**，原因见 §5 第 5 条（row 方向两套约定）。"源区整块污染" +
"污染区连通到源区"这两条与方向约定无关，才是可靠的那一半。

---

## 2. 已知且已接受的数值伪影：源片边缘的负浓度

**现象**：Case C 中 `C_min ≈ −4.8 mg/L`，出现在源片（row 8–17, col 18–27）边缘外侧，
300 帧里 299 帧都有。量级为**源浓度的 0.024%**、峰值的 0.067%。

**已排除的成因**（均为实测，非推断）：

| 假设 | 实验 | 结果 | 结论 |
|---|---|---|---|
| 时间步截断误差 | `nstp` 30 → 60 | 负值 −4.8245 → −4.8828 | 排除：基本不变 |
| 弥散导致的下冲 | `al` = 5 / 40 / 120 | 负值 −10.51 / −4.82 / −0.71 | 排除：弥散越大负值**越小**，与假设相反 |
| 求解器容差过松 | `hclose` 1e-3 → 1e-7 | 负值 −4.8286 | 排除：不变 |
| 对流格式超调 | 检查 `.adv` | `SCHEME upstream` | 排除：upstream 本就不产生对流超调 |
| 横向弥散交叉项 | `trpt=trpv=0` | 模拟不收敛 | 实验无法完成，未定论 |

**判定**：MF6 的 GWT 是**隐式**求解，弥散张量的交叉项会破坏系数矩阵的 M-矩阵性质，
在陡峭的源片硬边界处产生远小于源浓度量级的负值。这是方法的固有特征，不是本链路的缺陷——
同样配置下 Case A 的 `min = 4.7e-07`、Case B 的 `min = 6.5e-13`，都干净无负值。

**处置**（用户 2026-09-15 决定）：接受为已量化的数值伪影，写入本节。
`scripts/verify_east_texas.py` 使用容差 `NEG_TOL_FRAC = 1e-3`
（即 `|C_min| ≤ 源浓度的 0.1%`，相对实测 0.024% 留 4 倍余量）。
**Case A / Case B 仍使用严格判据 `C.min() ≥ −1e-9`。**

后续若升级到 MF6 6.5.x，可考虑改用 ATS 的 Courant 控制重测本节结论。

---

## 3. 回归基线

```text
tests/test_gwt_pipeline.py               52/52  通过   （46 → 52，+6 为 total_time 相关）
tests/test_water_budget.py               23/23  通过   ← 本轮新增（水量平衡改由 .lst 权威解析）
tests/test_source_zone.py                97/97  通过   ← 本轮新增（源区定位 + 方位/时间约定）
tests/test_prompt_templates.py           52/52  通过   ← 本轮新增（模板能否 .format 渲染）
tests/test_plume_visualization.py        76/76  通过   （65 → 76，+11 为"确定重绘"接线锁）
tests/test_model_builder.py              全部通过
tests/test_physics_engine.py             全部通过
tests/test_boundary_pairs.py             56/56  通过
tests/test_error_handling.py             23/23  通过 (100.0%)
tests/test_subsystem_integration.py      90/90  通过 (100.0%)  ← 方位约定修正后由 89/90 转全通过
tests/test_all_imports.py                30/31  通过 (96.8%)  ← 既有失败
```

```bash
# 脚本式验证（非 pytest）
python scripts/verify_east_texas.py D      # Case D：源区回归锁，只构建不跑 mf6（几秒）
```

`test_subsystem_integration` 的 `I2.1: Schema符合率` 依赖**真实 LLM 网络调用**（`extract_entities`
/ `classify_intent` 会走网络），Mock IR 每次运行内容不同，故该套件的总数会在 89~90 之间浮动；
2026-09-17 复跑为 90/90。

一项既有失败与本轮无关：

- `test_all_imports` 的 30/31，在 Git HEAD 上即失败。

---

## 4. 复现命令

```bash
export PYTHONIOENCODING=utf-8          # Windows GBK 控制台必需

python tests/test_gwt_pipeline.py                    # 全部 52 项（含真实 mf6）
SKIP_MF6_TESTS=1 python tests/test_gwt_pipeline.py   # 只跑 Gate 0 纯函数，秒级

python scripts/verify_gwt_cases.py AB                # Case A + B
python scripts/verify_east_texas.py                  # Case C
python scripts/verify_east_texas.py D                # Case D：源区定位回归锁

# Case C 参数试验（输出写独立目录，避免 Windows 文件锁）
ET_NSTP=60   ET_WS=data/outputs/gwt_verify/t1 python scripts/verify_east_texas.py
ET_AL=120    ET_WS=data/outputs/gwt_verify/t2 python scripts/verify_east_texas.py
ET_HCLOSE=1e-7 ET_WS=data/outputs/gwt_verify/t3 python scripts/verify_east_texas.py
```

环境变量一览：`MF6_EXE`（mf6 可执行文件路径）、`SKIP_MF6_TESTS`、`ET_NSTP`、`ET_AL`、
`ET_TRPT`、`ET_TRPV`、`ET_HCLOSE`、`ET_NEG_TOL`、`ET_WS`，以及 Case D 专用的
`ET_SHP`（`landfill_arcs.shp` 路径；缺省按「仓库内 → `%TEMP%\gm_zip_*`」自动查找）、
`ET_ORIGIN_X` / `ET_ORIGIN_Y`、`ET_WS_D`。

---

## 5. 已知的其余风险（未在本轮解决）

1. **East Texas 的 IR 物理设计仍需对话层引导**。2026-09-17 复核后现状是四条边界：
   `chd north_boundary 215 m`、`ghb south_boundary 212 m`、`ghb east_boundary 208 m`，
   以及填埋场 RCH；实跑水头范围 212.03 ~ 215.0 m，**已有梯度驱动**，比早期"只有两个 GHB、
   无梯度驱动"好得多。但域内流速仍极低（约 0.02 m/d），15000 d 平流位移仅约 300 m，
   弥散占比偏高。这是 IR 的物理设计问题，靠改代码解决不了。
2. **`vka` 单位未做校验**：`vka = {"value": 20, "unit": "m"}` 单位应是 m/d。建议在
   `validator.py` 加"垂向渗透系数单位必须是 m/d"的检查。
3. **`_build_gwt` 的初始浓度只支持单一常数**，真实案例常需"局部高浓度 + 背景浓度"分区。
4. **MF6 版本倒挂**：flopy 3.10.0 的 dfn 面向比 6.3.0 更新的 MF6，传 6.3.0 不认识的关键字
   会被 `store_error(terminate=.TRUE.)` 硬终止（已知：ADV 的 `UTVD`、`ats_percel`）。
   升级到 MF6 6.5.x 是更彻底的解法。
5. **方位名与网格行的方向曾上下颠倒 —— 2026-09-17 已修。**

   修复前仓库内有两套相反的约定：

   | 位置 | 修复前 | 修复后 |
   | --- | --- | --- |
   | `app/modeling/source_zone.py` `infer_edge_indices` | `north` → **row 0**，`south` → row nrow-1 | `north` → row nrow-1，`south` → row 0 |
   | `app/utils/grid_coords.py` `grid_cell_centers`（未改） | row 索引随 y 增大 ⇒ row 0 在图的最下方 | 同左 |

   两者单独看都自洽，放在一起就意味着**名为 `north` 的边界被写到最南一排、画在图的底部**。
   当时的证据链：

   - `data/sessions/default.json` 的 `south_boundary` 是 GHB stage 212，实测写进
     `auto_model_gwf.ghb` 的记录是 `1 103 <col> 212 ...`（1-based row 103 = 0-based 102
     = nrow-1）；
   - 读 09/15 那次运行的 `auto_model_gwf.hds` 末帧：最高水头 211.997 出现在
     **(row 102, col 0)**，最低 208.019 在 (row 0, col 217)；
   - 绘图路径 `contourf(X, Y, head)` 全链路**没有任何翻转/转置**
     （`grep flipud|np.flip|[::-1]` 在 `app/` 下零命中），故 row 102 落在图的上方。

   **选定方向：以 `grid_cell_centers`（y 递增）为准，即"row 0 在南、row nrow-1 在北"**，
   判据是与 SHP 独立对齐：`landfill_arcs` 在 y 上占 (3428453-3427412)/1871 = **55.6%**，
   栅格化后是 rows 52..62（中心 55%）——只有"row 越大越靠北"这一套能同时对上 SHP 坐标、
   可视化底图与 `.rch` 的实测行号。因此改的是 `infer_edge_indices`，不是 `grid_cell_centers`
   （改后者会把已对齐 SHP 的底图整体镜像）。

   修复后的实测（用迁移后的真实会话 IR 端到端重跑）：

   ```text
   auto_model_gwf.chd : rows(1-based) 103..103  (= 0-based 102 = nrow-1)  值 [215.0]  ← north_boundary
   auto_model_gwf.ghb : 212.0 落在 row 0（=0-based 0 = 最南）                          ← south_boundary
   ```

   回归锁见 `tests/test_source_zone.py` 的 5.9a~5.9f：不只断言"north → nrow-1"，
   还把两套约定钉在一起断言 `y(row nrow-1) > y(row 0)` —— 单看任何一套都自洽，
   只有这一条能挡住它们再次漂移。`test_subsystem_integration` 因此从 89/90 转为 90/90。

   Case D 的羽流判据仍刻意**不判**迁移方向（源区是否整块污染 + 污染区是否连通到源区），
   因为它要回答的是"污染物是不是从填埋场出来的"，与南北无关。

---

## 6. 时间离散的界面可见性（150000 d → 15000 d）

用户报"模拟时长展示的还是 150000 天，并没有修改为 15000 天"。复核结论：**显示没错，是配置错了**，
而且"配置错了"这件事在界面上**从来没被说出来过**。三层叠加：

| # | 层 | 现象 | 处置 |
| --- | --- | --- | --- |
| 1 | 会话 IR | `data/sessions/default.json` 的 `transport.stress_periods` 是 `{nper:10, perlen:15000}`，按 MF6 语义是 **10 × 15000 = 150000 d** | 已迁移为 `{nper:10, total_time:15000, nstp:1}` |
| 2 | 校验层 | `total_time` 警告**能算出**（"模拟总时长为 150000 d（410.7 年）"），但界面只渲染 `error_count > 0` 的错误，警告只计数进 `monitor.info` | 已渲染：琥珀色卡片 + 聊天回复 |
| 3 | 抽取链 | `_merge_ir` 的顶层白名单在**每一层递归**都生效，把 LLM 已产出的 `total_time` 静默删掉 | 已修（白名单限 `_depth == 0`） |

第 2 层是最值得记的一点：提示**算得出、也生成了**，却没有任何 UI 组件消费它 ——
和 `logger.info` 孤点提示是同一类失效，说了，但没人听得见。

界面侧新增两处（`app/frontend/gradio_app.py`）：

- 结果面板多一行 **`计划模拟时长（来自 IR）`**，与跑出来的时间轴末刻并排，配不上对时当场可见；
- 校验警告渲染为琥珀色卡片，并**按信噪比折叠**：East Texas 的 IR 一次出 12 条警告，其中 9 条是
  "参数'PCG'为非数值字符串，跳过范围检查（可能是布尔/枚举值）"这类枚举型参数的例行提示，
  全长铺开会把唯一要紧的那条淹掉，因此低优先项折叠成一行计数。

### 用户会话的迁移（一次性）

`data/sessions/default.json` 被 `.gitignore` 忽略（属本地运行状态），改动前已备份为
`default.json.bak-20260917`。迁移内容：

```json
// 迁移前
"stress_periods": 10,
"transport": {"stress_periods": {"nper": 10, "perlen": 15000, "nstp": 1, ...}}

// 迁移后（两份都改：transport 那份会被后续轮次当作 current_json_ir 再喂给 LLM）
"stress_periods": {"nper": 10, "steady_state": false, "total_time": 15000, "nstp": 1},
"transport": {"stress_periods": {"nper": 10, "steady_state": false, "total_time": 15000, "nstp": 1}}
```

依据是用户提示词第 9 行原文"模拟时长15000天，均匀分为10个应力期，每个应力期1个时间步"。

### 迁移后的端到端实测

用**迁移后的真实会话 IR**（`domain.origin_xy=[612284, 3427412]`、103×218 网格、真实
`landfill_arcs.shp`）完整建模并跑 mf6：

```text
TDIS NPER = 10
TDIS perlen = 1500 x 10 期, nstp = 10  ->  总时长 = 15000 d, Δt = 150 d
RCH  MAXBOUND = 132, 记录数 = 132, rows(0-based) 52..62  cols 137..148
CHD  rows(1-based) 103..103 (= 0-based 102 = nrow-1)  值 [215.0]   ← north_boundary
GHB  212.0 落在 row 0（最南）                                      ← south_boundary
.ucn 输出时刻: 100 帧, 150 .. 15000 d
360 d 可插值区间: (300.0, 450.0)
```

最后一行就是用户"想看 360 天"的落点：360 d 落在 300~450 d 之间，单帧交互图可线性插值。

用**修复前的旧形态**（顶层裸 `10` + `transport.perlen=15000`）复跑校验，卡片渲染为：

```text
参数校验警告 12 条 （3 条需留意）
· total_time: 模拟总时长为 150000 d（410.7 年），请确认是否符合预期。（若 15000 天其实是**总**模拟时长，请改写为 "total_time": 15000，否则会被按每期 15000 天解释）
另有 9 条枚举/字符串类参数未做范围检查的低优先提示（不影响建模）
```

---

## 7. 单时刻污染羽的色标上下限改为「确定重绘」

**现象**：想在下限框输入 `300`，敲下 `3` 的那一刻程序就开始算，用户必须等这次结果刷出来
才能接着键入 `0` —— 只要想输入多位数就必然被卡住。

**根因是事件语义，不是计算慢**：原来三个控件都挂了 `.change`：

```python
for _w in (plume_auto_range, plume_vmin, plume_vmax):
    _w.change(_on_plume_number, [...], [...])
```

`gr.Number.change` 是**逐字符**触发的，每次键入即一次完整重绘。

**改法**：新增 `✅ 确定重绘` 按钮 + `_on_plume_apply`，事件接线收敛为

| 触发源 | 事件 | 说明 |
|---|---|---|
| `plume_time_slider` | `release` | 拖动结束才触发（原样保留） |
| `plume_time_num` | `submit` | 回车即改（原样保留） |
| `plume_apply_btn` | `click` | **新增**：输入完上下限再点 |
| `plume_vmin` / `plume_vmax` | `submit` | **新增**：回车等价于点确定 |
| `plume_auto_range` | `change` | 复选框单击一次即生效，不涉及键入 |

浓度框自身不再有任何 `change` 绑定，键入过程完全不触发计算。顺带消除了一类隐患：
程序回写输入框不再引发新事件，不存在回环。

`_on_plume_apply` 的时刻取值：数字框有有效值用它，否则回退滑块 —— 刻意复用
`_render_plume` 而不是复制一份逻辑，色标解析（`_pick_vmin_vmax`）的既有语义
（勾自动忽略输入框、单框留空回退自动、上限小于下限自动纠正）一并继承。

**回归锁**：`tests/test_plume_visualization.py` 9.1~9.11（65 → 76）。与纯函数测试不同，
这里断言的是**组件图上的边**：

- 9.7~9.9：能按标签定位两个浓度框、能扫到 change 事件、`_on_plume_apply` 恰好 4 条绑定
- 9.10：**没有任何 change 事件挂在浓度框自身上**
- 9.11：**判据自检** —— 把坏边原地接回去，扫描必须报警

9.11 是必需的，因为本节初版的 9.10 是**空过的**：判据按 `fn.inputs` 里的 id 解析输入，
而 Gradio 存的是**组件对象**，永远得空集 → 接回坏边照样报绿。另外判据从"输入含浓度框"
收窄为"**触发源**是浓度框"：`plume_auto_range.change` 的输入里也含这两个框（处理器要读
它们），那是单击复选框这一下，属合法。

---

## 8. 界面与图件外观改版（不影响判定结果）

本轮（2026-09-17）做了界面美化与图件配色改版，**不改任何数值结果**，但有一处变化
会让"和上一轮截图对不上"：

| 项 | 改前 | 改后 |
|---|---|---|
| 单时刻污染羽色阶 | `Reds` | `viridis` |
| 通用浓度色阶 | `YlOrRd` | `viridis` |
| 阈值线颜色 | 硬编码 `red` | NPG 红 `#DC0000` |
| 穿透曲线 / 过程贡献分类色 | `plt.cm.tab10` / `Set2` | NPG 10 色（`categorical(n)`） |
| 图件字体 | 整张图统一用一个 CJK 字体 | 逐字形回退：西文 Arial、中文微软雅黑 |
| 图件字号 | 10 / 12 / 11 pt | 9.5 / 11 / 10 pt（期刊插图尺度） |

**判定基线不变**：Case A~D 断言的是数值（水头范围、`.rch` 格数、浓度、行/列索引），
与色阶无关。改版后复跑八个套件，计数与 §3 基线逐项一致。

改动的复核方式与其他项不同：色阶与字体是**看得见**的，所以直接渲染成 PNG 用眼核对
（本机无 playwright/selenium，web 界面无法截图，只能对已服务的页面做内容断言）。
核对点：中文无方块、西文为 Arial、污染羽高低浓度可分辨、色盲模拟下仍可分辨、
灰度打印条件下仍能分档。

已知**未做**：去上/右边框（matplotlib 无对应 rcParam，逐轴去边要改 9 个绘图函数）、
聊天记录的"扁平无气泡"样式（需覆盖 Gradio 内部类名，无浏览器环境无法确认其存在）。
详见 `docs/调试日志-202606.md` 第十六节。

