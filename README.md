# simple_disease_simulation

基于 PyTorch + matplotlib 的**逐个体（agent-based）传染病模拟**：SEIR 框架上扩展出
无症状（A）与住院（H）状态，支持按疾病档案（JSON schema v1/v2/v3/v4）切换自然史、
v3 社会响应（检测 / 追踪 / 隔离 / 行为反馈 / 报告延迟）与 v4 个体异质性
（超传播 / 依从性差异 / 年龄同配混合），并在 matplotlib 动画中实时观察空间分布、
人群构成曲线、病毒参数演化与再感染率。

## 背景

最早是为回答知乎问题[《对于致死率几乎百分百的病毒和细菌，那些在感染下幸存下来的人是因为基因很牛逼还是因为运气好？》](https://www.zhihu.com/question/1987928986068723299)写的小玩具，后来逐步扩展成现在这个可切换疾病档案的 agent-based 模拟器。

## 快速开始

```powershell
# 默认 covid19 档案（schema v4，全部机制开启）
python simple_disease_simulation.py

# 换档案 / 换人口 / 换种子 / 限帧数（CLI 优先于环境变量）
python simple_disease_simulation.py --profile toy
python simple_disease_simulation.py --profile covid19 --population 100000 --seed 7
python simple_disease_simulation.py --frames 3000          # 等价 SIM_FRAMES=3000
```

窗口内点击 **LOCKDOWN** 按钮可切换封控（`toggle`），左上标题实时显示计数、医疗压力、
免疫/再感染率、时钟与场所构成。

环境变量（与历史版本一致）：

| 变量 | 含义 | 默认 |
| --- | --- | --- |
| `DISEASE_PROFILE` | 档案名（等价 `--profile`，CLI 优先） | `covid19` |
| `DISEASE_POPULATION` | 人口规模（自动按 √r 缩放面积、按 r 缩放场所数） | 配置内 `N_AGENTS` |
| `DISEASE_SEED` | 随机种子 | 配置内 `SEED`=42 |
| `SIM_FRAMES` | 覆盖运行帧数 | 配置内 `FRAMES` |
| `CONTACT_MODE` | `auto` / `dense` / `grid` 接触对算法 | `auto` |
| `SIM_RENDER_EVERY` | 每 K 帧重绘一次（`1`=每帧；模型仍逐帧推进） | `1` |
| `SIM_RENDER_POINTS` | 散点抽样上限（大人口下限制 Agg 绘制成本） | `10000` |

渲染节流也可用命令行：`python simple_disease_simulation.py --render-every 4`；
或 `--async-render`（默认 K=2，模型逐帧推进、每 2 帧重绘，10k 下约 1.6× 有效帧率）。

### 改超参数做自己的实验（五种方式，按侵入性从低到高）

| 想改的东西 | 做法 | 例子 |
|---|---|---|
| 人口 / 种子 / 帧数 | CLI 或环境变量 | `python simple_disease_simulation.py --population 50000 --seed 3 --frames 6000` |
| 现有档案里的数字（R0、无症状比例、检测能力、σ、家庭乘子、依从性…） | 复制 `diseases/<档>.json` 为新文件（**文件名 = `profile_id`**），改数值后 `--profile <新名>`；`tools/profile_check.py --profile <新名>` 会检查派生自洽 | 见 [diseases/README.md §7](diseases/README.md) 的 9 步流程 |
| 一次性试验（不想改仓库） | 用 `model_overrides` 简档（v1 风格）写一个临时 JSON，直接覆盖 CONFIG 键 | `diseases/toy.json` 就是这种简档 |
| 关掉 v3/v4 看差异 | 用现成对照档 `covid19_v2_noiv`（无社会响应）/ `covid19_homogeneous`（无 v4 异质性），或在自己的档案里把块设 `"enabled": false` | `--profile covid19_homogeneous` |
| 统计而不是单条轨迹 | `tools/ensemble.py` 多 seed 分位带（p10/p50/p90），可加 `--track-secondary` 看超传播 | `python tools/ensemble.py --population 10000 --frames 3000 --seeds 1-12` |

> 改完档案用 `python tools/profile_check.py --profile <新名>` 自查派生自洽
> （解析式与 CONFIG 是否一致、R0/代际是否落在合理量级）。实验档是独立文件，
> 不影响其它档案。

## 文档地图

| 你想知道 | 看 |
| --- | --- |
| 怎么跑起来、怎么改参数做实验 | 本文（快速开始 / 改超参数） |
| 目录结构、模块职责、一帧数据流、兼容层原理 | [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) |
| 每个机制怎么实现（状态机 / 自然史 / 医疗 / v3 社会响应 / v4 异质性） | [MECHANISMS.md](MECHANISMS.md) |
| 参数取值、量级、缩放、性能与吞吐实验 | [SCALING.md](SCALING.md) |
| 哪些地方不像真实世界（以及为什么不打算拟合） | [REALISM.md](REALISM.md) |
| 怎么新增一种疾病 / 档案字段与校验规则 | [diseases/README.md](diseases/README.md) |
| 有哪些测量工具、怎么读它们的结果 | [tools/README.md](tools/README.md) |

## 工程结构

```
simple_disease_simulation.py   # 薄入口（装配 + 给分析工具转发模块级名字，~100 行）
disease_sim/                   # 主包
├── config.py                  # CONFIG 缺省值 / 档案应用 / 人口缩放
├── state.py                   # SimState 容器 + build_state()
├── activity.py                # 日程出行模型
├── contacts.py                # 接触对生成（dense / grid）
├── disease.py                 # 传染力曲线 LUT
├── interventions.py           # v3 检测/追踪/行为反馈/报告延迟
├── simulation.py              # update(sim, frame) 帧编排
├── render.py                  # 图元、按钮、toggle
├── cli.py                     # FuncAnimation 入口
└── profiles/                  # 档案子系统（schema/loader/validate/derive/apply）
diseases/                      # 档案 JSON
tools/                         # 分析/测量工具（见 tools/README.md）
benchmarks/                    # 基准与多 seed 结果（本地数据，不入库）
docs/ARCHITECTURE.md           # 模块职责、依赖方向、数据流
```

> `tools/` 里的脚本通过“加载入口 + 读取模块级名字”驱动模拟，仓库根没有测试目录；
> 想验证改动是否破坏机制，直接跑 `python tools/profile_check.py` 或
> `python tools/ensemble.py` 看轨迹/分布是否还合理。

依赖方向单向、无环：`cli → simulation → {activity, contacts, disease, interventions, render}
→ {state, config}`；`profiles` 完全独立。`tools/ensemble.py` 在独立子进程里跑多个 seed
（不改模型），详见 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)。

## 疾病档案

| 档案 | schema | 说明 |
| --- | --- | --- |
| `diseases/covid19.json` | v4 | 自然史 + 医疗 + interventions + v4 异质性/混合全开 |
| `diseases/covid19_homogeneous.json` | v4 | 同上但 `heterogeneity`/`mixing` 关闭：**同质对照**（isolate v4 的方差效应） |
| `diseases/covid19_v2_noiv.json` | v3 结构、无 `interventions` 段 | 自然史+医疗开、社会响应全关（等价 v2 行为），常驻对照档 |
| `diseases/toy.json` | v1 | 全部新机制关闭的玩具档案，快速演示与统计回归用 |

档案字段与派生规则见 [diseases/README.md](diseases/README.md)；疾病机制见
[MECHANISMS.md](MECHANISMS.md)；参数与文献对照见 [REALISM.md](REALISM.md)；规模化见
[SCALING.md](SCALING.md)。

## 性能（RTX 4060 Laptop 8G，covid19 档案，400 帧，seed=42）

**优化幅度（同进程交替 A/B，抗机器漂移；covid19 档案，5 轮 × 120 帧取中位数）**

| 人口 | before（原始单文件） | after（当前实现） | 变化 |
|---|---|---|---|
| 10k | 14.319 ms/帧 | 11.303 ms/帧 | **−21.1%** |
| 100k | 28.518 ms/帧 | 22.048 ms/帧 | **−22.7%** |
| 1M | 33.570 ms/帧 | 26.737 ms/帧 | **−20.4%** |

toy 档案（满负荷疫情）10k 同口径：22.434 → 18.054 ms/帧（**−19.5%**）；较早一轮
曾测得 −16.8% 与 −29.6%，故 toy 10k 的稳健区间约 **−17% ~ −30%**。

**绝对口径（bench.py，covid19，400 帧，当前机器状态）**

| 人口 | update 均值 | update 中位 | draw+blit | CUDA 峰值 | wall |
|---|---|---|---|---|---|
| 10k | 21.126 ms | 18.974 ms | 231.5 ms | 12.19 MB | 100.8 s |
| 100k | 60.861 ms | 59.424 ms | 259.6 ms | 167.72 MB | 127.9 s |
| 1M | 57.773 ms | 33.340 ms | 237.0 ms | 369.46 MB | 117.6 s |

`bench_before_*.json`（优化前，同日更早时刻）与上表**不可直接相减**：机器在长时间
GPU 负载后整体变慢（draw+blit 代码几乎未改却从 133 ms 涨到 232 ms），故改进幅度以
同进程 A/B 为准。峰值显存三档基本持平（369.72 → 369.46 MB）：1M 峰值由常驻状态
张量决定，而非逐帧临时量；被修复的是**主机内存泄漏**（`mobility_stats` 累积容器，
1M 下原本每帧新增 1.4 万条记录，45 天将增长到 GB 级，现已封顶 65536 条并改为按日聚合）。

口径说明：`update` 含 `torch.cuda.synchronize()`（比只测 launch 的口径更严苛）；
`draw+blit` 为 15 个 artist 逐个重绘 + blit；before 为重构前的单文件实现（`_orig_sim.py` 副本），
after 为当前包实现。测量期间 GPU 非严格独占（桌面约 0.3 GB 占用），三条基准顺序执行。
注意 `bench.py` 的 `draw+blit` 是**每帧全量 `canvas.draw()`** 的最坏口径；交互式
`FuncAnimation(blit=True)` 的真实成本约为其 1/3（见下）。

### 第二轮优化（S1 算子压缩 / S2 吞吐 / S3 渲染）

| 优化点 | 手法 | 实测效果 |
|---|---|---|
| 分段线性曲线 | dense LUT（0.25 天）+ 2 次 1D gather，数值等价（≤1.2e-7） | 每帧 aten 事件 **1222 → 967（−21%）** |
| 统计段 | 24 维向量合并、整帧一次 `.cpu()` 同步 | 同步点由 15.6 次/帧降到 1 次 |
| 布尔掩码索引 | 复用索引 / 掩码写 / 主机侧计数门控空分支 | 纯计算 A/B **−6.1%**（10k，噪声地板 ±2%） |
| 渲染节流 | `--render-every K` + 抽样上限 10000 | 10k 交互 74.8 → 42.2（K=2）→ 26.7（K=4）ms/帧 |
| 多进程吞吐 | `tools/sweep.py`（K 个独立世界并行） | 10k K=8 285.0 帧/s（3.43×）、util 95%；1M K=4 76.5 帧/s、util 95%（第三轮后复测） |

多进程吞吐矩阵（covid19，`SIM_RENDER_EVERY=0`）与渲染三口径的完整表格见
[SCALING.md](SCALING.md) §6.4–§6.6；其中也解释了"为什么单进程吃不满 GPU"
（每帧约 1000 个小 kernel 的 launch 开销主导，而非算力受限）。

### 第三轮优化（Wave 1：整幅化 / 索引复用 / 同步消除）

在第二轮基础上继续压缩"每帧 kernel 数与主机同步"，语义不变（对外仍是同一套状态机、
概率、曲线、床位规则与可视化 contract）：

| 优化点 | 手法 | 实测效果 |
|---|---|---|
| 活动模型整幅化 | 停留段去 `nonzero`+gather+掩码写；位置 clamp 合并；`lerp` 推进；原地 `add_(bool)` | 活动段 index 类算子显著减少 |
| 病程/传播 | `_inf_lut` 一次 gather 取代 3–5 个比较；住院空床数主机侧记账（省 2 次 `.sum().item()` 同步）；`p_pair` 先合成感染者侧系数 | 同步点进一步减少 |
| 统计段 | 状态以 `uint8` 拷回；再感染率前缀和只算显示窗口尾部 | 1M 传输 8 MB→1 MB；长跑不再 O(T) |
| v3 门控 | 检测/追踪/隔离倒计时用上一帧主机计数整块跳过 | 无症状期这些流程零开销 |

同进程交替 A/B（`tools/ab_iter.py`，8 轮 × 240 帧、丢弃前 2 轮、配对中位比，
对照为本轮改动前的快照）：covid19 10k **−11.8%**、covid19 1M **−14.7%**、
toy 10k **−4.0%**（toy 两侧 RNG 流分叉，末帧工作集不同，可比性差）。
算子证据：covid19 10k 每帧 aten 事件 **978 → 863（−11.8%）**。
同机噪声地板约 ±3%，故只把 >3% 的配对差视为真实收益。详见
[SCALING.md](SCALING.md) §6.2c 与 [MECHANISMS.md](MECHANISMS.md) §11.2。

### 第四轮优化（Wave 2：算子瘦身 / 接触对 / 渲染）

用户在本轮前明确"不再要求逐位一致"，于是可以把不可观测写与冗余 kernel 一并删掉：

| 优化点 | 手法 | 实测效果 |
|---|---|---|
| 删除不可观测写 | 死亡/康复者的 5 个 `virus_*` 字段清零整段删除（没有任何代码读它、传染力只读 E/I/A） | 疫情期每帧省 20–45 次 `index_put` |
| 变异/免疫批量 | 三组病毒参数变异合成 `(3,n)` 张量一次算；免疫增益改 `index_add_`+`clamp_` | 变异块 kernel ≈39 → 17 |
| 掩码去重与折叠 | `status==3` 每帧一次；`p_pair` 感染者侧系数先合成；v3 三条流程改索引化（去 `bool(any())` 与整幅 N 布尔） | covid19 10k 内核 **877 → 833** |
| 接触对 grid | 1 格外边距（越界邻格落到空 padding 格，免越界掩码）+ 步长缓存；另做过 5 组随机形态 parity 校验 | 1M 接触段 4.216 → **3.617 ms（−14.2%）** |
| 渲染 | 主散点改小方块 marker（`s=2`，顶点数约 1/6）；新增 `--async-render[=K]` | 10k 交互 **37.3 → 22.2 ms/draw（−40%）**；K=2 amortized 20.9 ms（**47.9 fps**） |

同进程交替 A/B（对照 = 第四轮前快照）：covid19 10k **−15.0%**（配对中位；
medians −19.3%）、covid19 1M **−8.3%**（medians −11.7%）、toy 10k −22.4%（轨迹分叉，
仅参考）。当时全量回归 6/6 PASS（接触对 parity + 结构断言；测试套件已移除，
数字仅作历史）。详见 [SCALING.md](SCALING.md) §6.2d（含 §6.5 渲染三口径更新）。

### 第五轮（v4 机制）：个体异质性与接触混合（不是性能优化）

均匀混合 + Poisson 型传播几乎不会熄火、也长不出超级传播者，而"同 R0、不同个体差异 →
有的种下去就灭、有的炸开"正是本类 toy model 的教学点。v4 只加三个最小结构：

| 机制 | 档案键 | 效果 |
|---|---|---|
| 传染力个体异质性 | `heterogeneity.infectiousness_sigma=1.4` | 每人一个固定对数正态乘子（均值 1）；12 seeds 攻击率 1.3%–9.2%、二代 var/mean 中位 **2.62**（同质对照 1.21） |
| 依从性个体异质性 | `compliance_mean=0.8 / compliance_sd=0.15` | 隔离、追踪、行为反馈的概率各乘本人 compliance（只改阈值，随机数个数不变） |
| 接触混合 | `household_contact_mult=3.0 / age_assortativity=0.5 / symptom_stay_home_fraction=0.6` | 同户接触×3、4 组年龄同配（归一化保 R0 不变）、部分有症状者自发居家 |

- 关闭路径（`toy`/`covid19_v2_noiv`）**不消耗任何随机数**，toy 300 帧 counts 在 v4 前后逐位不变
  `[18,102,820,6659,1632,769,0,0]`；解析 R0 仍 3.00、代际 4.54 天、倍增 2.34 天。
- 观测层：`tools/ensemble.py` 多 seed 分位带（p10/p50/p90）+ 可选 `--track-secondary`
  逐感染者二代统计；示例输出 `benchmarks/ensemble_covid19_10k_12_after_audit.{json,csv,png}`
  （审计修复后重跑；隔离总量从恒 0 变为 p50 346）。
- 观测层对照：`covid19`（异质性开）与 `covid19_homogeneous`（全中性）两档案，
  比较分布层方差与二代过散布；详见 [MECHANISMS.md](MECHANISMS.md) §12 与
  [SCALING.md](SCALING.md) §6.2e。

## 自检与复现实验

项目不含测试套件：验证方式是跑一次冒烟、看档案派生自洽、看多 seed 分布是否合理。

```powershell
python simple_disease_simulation.py --profile toy --frames 30        # 冒烟：能跑通、能出图
python tools/profile_check.py --profile covid19                      # 档案派生 / R0 / 代际时间自洽
python tools/bench.py --profile covid19 --population 10000 --frames 400   # 单帧耗时与显存
python tools/ensemble.py --profile covid19 --population 10000 --frames 3000 --seeds 1-12 `
    --json benchmarks/ensemble_covid19_10k_12.json --png benchmarks/ensemble_covid19_10k_12.png
```

> **验收口径（2026-10 起）**：第二轮性能优化改变了浮点运算顺序与 RNG 路径，
> **不再要求逐位复现**。判断改动好坏看三件事：每帧 `sum(counts)==N` 守恒、
> toy/covid19 的攻击率与病死率量级合理、同 seed 重跑轨迹一致（无自动断言，
> 靠 `tools/ensemble.py` 的分布与两次运行对比观察）。

历史金标与基准数据在 `benchmarks/`（本地数据、不入库，S1 之前采集，作长跑与量级参照）：

| 文件 | 内容 |
| --- | --- |
| `golden_toy_300.json` | toy 300 帧，末帧 counts `[13,102,962,6345,1586,992,0,0]` |
| `golden_v3_300.json` / `golden_v3_500.json` | covid19 300/500 帧 |
| `golden_v3_3000.json` / `golden_v3_10800.json` | covid19 长跑（12.5 天 / 45 天；45 天攻击率 75.95%、床位崩塌） |
| `golden_v2_noiv_400.json` | 自然史开、社会响应关，末帧 `[9977,0,23,0,0,0,0,0]` |
| `bench_before_*.json` / `bench_after_*.json` / `sweep_*.json` | 优化前后与吞吐基准 |

## 已知限制 / 未完成项

- 行为反馈层较弱：仅调节每日外出概率与休闲场所接触乘数，不影响工作/学校/家庭接触。
- 检测只覆盖有症状者；无二代追踪、无疫苗/免疫逃逸、无口罩与学校关闭。
- **v3 追踪/密接隔离已于 2026-11 审计修复**（此前 `trace_timer` 无递减 + 门控恒 0，
  `_resolve_traces` 从不执行、`quarantined` 恒 0）。修复后 covid19 的隔离首次真正生效；
  机制说明见 [MECHANISMS.md](MECHANISMS.md) §7.3。
- 同轮审计还修了两处：病房康复释放的床位在同帧不可用（多判 `untreated`）、
  死者不释放 `quarantined` 标记（隔离计数虚高）。
- `covid19_v2_noiv` 的末帧 counts 因此从历史快照的 `[9977,0,23,0,0,0,0,0]`
  变为 `[9969,0,31,0,0,0,0,0]`（补收更多 `untreated` 是修复的预期行为）。
- **第二批审计修复（同日，不改 covid19 轨迹）**：① 派生层 `hospitalization_rate=0` 会
  除零崩溃（现阈值置 0），且阈值重复计算忽略 `medical.beds_per_1000` 覆盖；② 世界坐标
  裁剪原先两轴都用 `WIDTH`，非方形世界 y 越界（现按轴使用 `WIDTH/HEIGHT`，默认方形
  世界逐元素等价）。
- `tools/profile_check.py` 的解析检查针对 v2 自然史参数，尚未感知 v3 四层机制。
- v3 长跑（45 天）卫生经济学/干预效果尚无系统实验矩阵，仅单条金标轨迹。
- 接触对生成在 `dense` 路径仍是 O(N_inf×N_sus) 全对判定（小规模校验用）；`grid` 路径
  为分块空间网格，内存有界。
- **不再逐位复现**：同 seed 同配置的两次运行在前数百帧通常完全一致，但 v2/v3
  机制介入后会出现浮点末位差异累积；确定性靠两次重跑对比观察，没有自动测试守护。
- 单进程 GPU 利用率 16–44%（10k/1M），不是算力受限而是小 kernel launch 受限；
  `tools/diag.py` 实测每帧 877–1444 个 torch 算子、CPU 侧每算子约 9–13 µs，
  单帧 CPU 占比 76–97%（见 [SCALING.md](SCALING.md) §6.7）。压榨 GPU 的路径是
  `tools/sweep.py` 多进程并行（已实现）或 B-world 向量化（仅设计，见
  `docs/ARCHITECTURE.md`）。
