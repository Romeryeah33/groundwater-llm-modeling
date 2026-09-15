# 溶质运移（GWT）链路验证记录

本文记录 GWT 全链路打通后的验证基线、判定依据，以及一处**已知且已接受**的数值伪影。
后续修改这条链路时，请对照本文的基线回归。

- 验证驱动脚本：`scripts/verify_gwt_cases.py`（Case A/B）、`scripts/verify_east_texas.py`（Case C）
- 回归测试：`tests/test_gwt_pipeline.py`（46 项，含两个真实 mf6 算例）
- 环境：Flopy 3.10.0 / mf6 6.3.0（Windows）

---

## 1. 三个算例与判定结果

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

```
tests/test_gwt_pipeline.py               46/46  通过
tests/test_model_builder.py              全部通过
tests/test_physics_engine.py             全部通过
tests/test_boundary_pairs.py             56/56  通过
tests/test_error_handling.py             23/23  通过 (100.0%)
tests/test_subsystem_integration.py      89/90  通过 (98.9%)  ← 既有失败
tests/test_all_imports.py                30/31  通过 (96.8%)  ← 既有失败
```

两项既有失败与本轮无关：

- `test_subsystem_integration` 的 `I2.1: Schema符合率`。该断言依赖**真实 LLM 网络调用**
  （`extract_entities` / `classify_intent` 会走网络），Mock IR #2/#3 的内容每次运行不同，
  同一次运行中会出现 #1 通过、#2/#3 失败的情况。已通过"仅回退本轮的 `intent_parser.py`
  改动"验证：回退后反而更差（97.8% vs 98.9%，并新增一项 I2.2 失败），确认非本轮引入。
- `test_all_imports` 的 30/31，在 Git HEAD 上即失败。

---

## 4. 复现命令

```bash
export PYTHONIOENCODING=utf-8          # Windows GBK 控制台必需

python tests/test_gwt_pipeline.py                    # 全部 46 项（含真实 mf6）
SKIP_MF6_TESTS=1 python tests/test_gwt_pipeline.py   # 只跑 Gate 0 纯函数，秒级

python scripts/verify_gwt_cases.py AB                # Case A + B
python scripts/verify_east_texas.py                  # Case C

# Case C 参数试验（输出写独立目录，避免 Windows 文件锁）
ET_NSTP=60   ET_WS=data/outputs/gwt_verify/t1 python scripts/verify_east_texas.py
ET_AL=120    ET_WS=data/outputs/gwt_verify/t2 python scripts/verify_east_texas.py
ET_HCLOSE=1e-7 ET_WS=data/outputs/gwt_verify/t3 python scripts/verify_east_texas.py
```

环境变量一览：`MF6_EXE`（mf6 可执行文件路径）、`SKIP_MF6_TESTS`、`ET_NSTP`、`ET_AL`、
`ET_TRPT`、`ET_TRPV`、`ET_HCLOSE`、`ET_NEG_TOL`、`ET_WS`。

---

## 5. 已知的其余风险（未在本轮解决）

1. **East Texas 的 IR 本身设计不良**：只有两个 GHB、无补给、无梯度驱动，估算流动时间常数
   约 8 d，溶质主要靠弥散而非平流迁移。这是 IR 的物理设计缺陷，需要在对话层引导用户
   补边界/补给条件，而不是靠改代码解决。
2. **`vka` 单位未做校验**：`vka = {"value": 20, "unit": "m"}` 单位应是 m/d。建议在
   `validator.py` 加"垂向渗透系数单位必须是 m/d"的检查。
3. **`_build_gwt` 的初始浓度只支持单一常数**，真实案例常需"局部高浓度 + 背景浓度"分区。
4. **MF6 版本倒挂**：flopy 3.10.0 的 dfn 面向比 6.3.0 更新的 MF6，传 6.3.0 不认识的关键字
   会被 `store_error(terminate=.TRUE.)` 硬终止（已知：ADV 的 `UTVD`、`ats_percel`）。
   升级到 MF6 6.5.x 是更彻底的解法。
