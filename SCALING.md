# 人口规模扩展（10k → 100k → 1M）

> 文档分工：本文件讲**参数、缩放与性能实验**；机制见 [MECHANISMS.md](MECHANISMS.md)，
> 结构与数据流见 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)。
> 性能数字的测量口径极易误读，引用前先看 §6.2（抗漂移 A/B）与 §6.7（诊断）。

本文件说明同一份代码、同一组疾病参数如何通过环境变量 `DISEASE_POPULATION`
扩展到 10 万 / 100 万：缩放规则、接触对枚举、渲染抽样、性能基线（**重构前
口径**）与限制。代码已拆分为 `simple_disease_simulation.py` 薄入口 +
`disease_sim/` 模块 + `diseases/*.json` 档案。

## 1. 目标与约束

- 目标：在个人电脑（CUDA GPU）上运行 **10 万 ~ 100 万** 个体的模拟，语义与 10k 版本一致。
- 约束：疾病参数不重标定；同一档案在任意人口下派生结果一致（档案在人口缩放之前应用）。
- 默认人口仍是 10k；10k 路径走稠密接触矩阵（`CONTACT_MODE="auto"` 自动选择），行为与旧版一致。
- 疾病参数由 `diseases/*.json` 档案提供（默认 `covid19`，`--profile` /
  `DISEASE_PROFILE` 可切换；见 [diseases/README.md](diseases/README.md)），在人口缩放**之前**应用。
- 档案中"每万人基准"的量（`INITIAL_SEEDS`、旧版 `MEDICAL_THRESHOLD`）随后按
  `DISEASE_POPULATION` 等比缩放；`beta/gamma/mu` 等每帧速率与人口无关。
- v3 档案（schema 3）的床位总数 `beds = beds_per_1000 × N / 1000`、医院数
  `n_hospitals_per_100k × N / 100000`、**检测能力**
  `test_capacity_per_100k_per_day × N / 100000` 也按人口缩放，因此
  "床位/人口"与"检测能力/人口"比例保持不变。

## 2. 启动方式

```powershell
# 默认 10k
python simple_disease_simulation.py

# 10 万
$env:DISEASE_POPULATION="100000"; python simple_disease_simulation.py

# 100 万
$env:DISEASE_POPULATION="1000000"; python simple_disease_simulation.py

# 组合疾病档案（默认 covid19；toy 为旧演示参数；v2_noiv 关闭社会响应）
python simple_disease_simulation.py --profile toy
$env:DISEASE_POPULATION="1000000"; python simple_disease_simulation.py --profile covid19
python simple_disease_simulation.py --profile covid19_v2_noiv
```

> 注意：`covid19` 档案的模拟长度是 45 天 = 10800 帧（默认人口 10k 下约 18~21 分钟）；
> 1M 人口下同一长度会显著更久，建议用 `$env:SIM_FRAMES="2000"` 做快速查看。

`DISEASE_POPULATION` 非法或缺失时回退到 CONFIG 默认值（10000）。启动时会打印：

```
Population: 1000000 (x100.0), world: 1000.0x1000.0, workplaces: 2000, schools: 800, medical threshold: 37500
```

## 3. 缩放规则（保持密度与个体体验不变）

设 `r = DISEASE_POPULATION / 10000`，`s = sqrt(r)`。

| 量 | 缩放 | 10k | 100k | 1M |
|---|---|---|---|---|
| 世界边长 WIDTH/HEIGHT | `× s` | 100 | 316.2 | 1000 |
| 工作场所 / 学校 / 商店 / 公园 | `× r` | 20 / 8 / 12 / 6 | 200 / 80 / 120 / 60 | 2000 / 800 / 1200 / 600 |
| 居住区中心 | `× r` | 12 | 120 | 1200 |
| 场所散布 VENUE_SPREAD | `× s` | 8/8/4/10 | 25.3/25.3/12.6/31.6 | 80/80/40/100 |
| 出行速度 TRIP_SPEED | `× s` | 8 | 25.3 | 80 |
| 床位总数 beds_per_1000 × N / 1000（v2/v3） | `× r` | 30 | 300 | 3000 |
| 检测能力 tests/day = 800 × N / 100000（v3） | `× r` | 80 | 800 | 8000 |
| 医疗阈值（打印值；v2/v3 由档案派生，10k 基准 375；v1 legacy 用 CONFIG `MEDICAL_THRESHOLD`） | `× r` | 375 | 3750 | 37500 |
| 初始感染种子 INITIAL_SEEDS | `× r` | 20 | 200 | 2000 |

**不随人口缩放**的量：每帧传播/病程/死亡速率、自然史时长、检测/追踪/行为/
报告的延迟与乘子、行为阈值（每 10 万人 7 天发病率）、隔离时长、单场所人数、
通勤距离（帧数）。因此人口密度、单工作场所/学校人数、通勤时间、症状比例、
疫情时间尺度都保持与 10k 版本相当；医疗崩溃判定条件（住院人数达到床位容量）
与检测排队比例（检测能力按人口同倍放大）也保持一致。

初始化部分已向量化（家庭规模抽样、家庭构成、年龄/就业/入学、场所分配），
1M 人口导入约 2.2-2.5 s（重构前记录）。

## 4. 接触对枚举：dense / grid / auto

`CONFIG["CONTACT_MODE"]`：

- `"dense"`：`torch.cdist` 全矩阵 + 距离阈值，10k 默认走此路径
  （`n_inf*n_tar ≤ CONTACT_DENSE_LIMIT=30,000,000` 时 auto 选择）。
- `"grid"`：GPU 空间网格（格宽 = `INFECT_RADIUS`），只对**目标**（易感/康复）
  建网格；感染者按 `GRID_INF_CHUNK=100,000` 分块，9 个邻格一次性展开
  （单次 `repeat_interleave`），距离筛选按 `GRID_CHUNK=2,000,000` 分块。
- `"auto"`（默认）：按 `n_inf*n_tar` 自动选择。

等价性：dense 与 grid 曾在 5 组构造状态上比对 `(src, tar)` 集合完全一致（脚本已删），
差异只可能来自恰好落在半径边界上的浮点比较。10k 默认走 dense，轨迹不受影响。

## 5. 渲染抽样

`CONFIG["RENDER_MAX_POINTS"]=10000`：人口超过该值时只渲染随机抽样的 10000 个
个体（固定随机子集，不逐帧抖动），状态颜色按真实状态着色。图中散点密度因此在
大人口下仍可读，且 `update()` 的绘图开销受控；统计/标题/曲线仍使用全体人口的
真实数据。可用环境变量 `SIM_RENDER_POINTS=N` 覆盖（例如交互演示时设 5000
可把 10k 的 Agg 绘制耗时从 ~59 ms 降到 ~29 ms）。

## 6. 性能基线

### 6.1 重构前基准（before，Lead 提供；本文引用口径）

`python tools/bench.py --profile covid19 --frames 400`，结果来自
`benchmarks/bench_before_10000.json`、`bench_before_100000.json`、
`bench_before_1000000.json`：

| 人口 | update 均值 | update 中位 | draw+blit 均值 | CUDA 峰值显存 | wall |
|---|---|---|---|---|---|
| 10k | 16.933 ms | 16.095 ms | 133.064 ms | 12.32 MB | 59.849 s |
| 100k | 21.887 ms | 20.634 ms | 154.248 ms | 169.0 MB | 70.278 s |
| 1M | 26.585 ms | 25.518 ms | 167.449 ms | 369.72 MB | 77.42 s |

口径：

- `bench.py` 用 Agg 后端 + `DummyFuncAnimation` 打桩，不跑 GUI 事件循环；
  以 `importlib` 导入主入口并记录 `load_s`；
- 首帧 `update(0)` 含懒初始化（单独记为 `first_update`），再跑一次 warm；
- 随后逐帧 `update` 计时，CUDA 路径每帧 `torch.cuda.synchronize()`；
- `draw+blit` 单独计时；`wall` 含加载与绘图；CUDA 峰值取
  `max_memory_allocated`（另记 `reserved`）；
- 机器：RTX 4060 Laptop GPU、torch 2.5.1+cu124、Python 3.12.5。

读法（同一台机器、同一档案）：

- `update` 随人口增长主要来自状态向量与接触对规模；1M 时 400 帧均值仍只有
  26.6 ms，说明传播/移动是向量化的；
- `draw+blit` 在人口 ≥20000 后基本恒定（只画固定抽样点），三档 133–167 ms，
  其中大头是 matplotlib 的绘制；
- `wall` 的 60–77 s 主要花在 400 帧的 draw+blit 与加载上，不是 `update`。

### 6.2 重构后性能（同进程交替 A/B，抗漂移口径）

绝对值会随机器热状态漂移（本次记录中，同一份 draw 代码在 1.5 小时内从
133 ms 涨到 232 ms，说明整机吞吐在退化），因此**优化幅度以同进程交替 A/B 为准**：
`tools/_ab.py` 在同一进程内加载原始单文件实现与当前包实现，各热身 20 帧后交替
跑 5 轮 × 120 帧，取中位数（两边受同一漂移影响）。

covid19 档案、seed=42：

| 人口 | before（原始单文件） | after（当前实现） | 变化 |
|---|---|---|---|
| 10k | 14.319 ms/帧 | 11.303 ms/帧 | **−21.1%** |
| 100k | 28.518 ms/帧 | 22.048 ms/帧 | **−22.7%** |
| 1M | 33.570 ms/帧 | 26.737 ms/帧 | **−20.4%** |

toy 档案（满负荷疫情，dense 接触路径最重）10k 同口径曾测得
before 22.945→after 19.084（−16.8%）与 28.172→19.821（−29.6%）两轮，取中位数区间
约 **−17% ~ −30%**（见 `benchmarks/_ab_*.json` 与运行日志）。

`bench.py` 的绝对耗时（covid19，400 帧；2026-11-01 复测，每档重复 2–3 次取中位）：

| 人口 | update 中位（`--no-draw`） | update 中位（draw 交织） | draw 实测 | CUDA 峰值 |
|---|---|---|---|---|
| 10k | **8.2 ms**（7.6–8.4） | 9.2–14.9 ms | 111–191 ms | 12.5 MB |
| 100k | **13.2 ms**（11.6–18.1） | 12.8–49.7 ms | 142–176 ms | 161.8 MB |
| 1M | **19.6 ms**（18.5–20.2） | 32.7–35.8 ms | 192–221 ms | 422.5 MB |

注意：**绝对毫秒在本机不可跨时段比较**。早先记录的 `21.1 / 60.9 / 57.8 ms` 与
`231.5 / 259.6 / 237.0 ms` 采自整机异常慢速的一段时间——同一份代码的 `draw+blit`
（纯 CPU 工作）也从 104 ms 漂到 232 ms，且 `bench.py` 的均值会被单帧尖峰（max 450 ms）
抬高；今天同 commit 复测，update 中位回到 8.2 / 13.2 / 19.6 ms。判断改进幅度请用
同进程交替 A/B（§6.1、§6.2b–§6.2f）；判断单帧成本请只比 `--no-draw` 的 update 中位。

峰值显存 12.5 / 161.8 / 422.5 MB：1M 峰值主要由常驻状态张量决定（v4 新增 3 个 N 长
浮点数组），而非逐帧临时量；逐帧临时量的下降体现在 `reserved` 与 kernel 数上。

### 6.2b 第二轮优化（S1，2026-10-31 晚）与验收口径变化

第二轮改动集中在"每帧算子数"：分段线性曲线改 0.25 天步长的 dense LUT + 2 次
1D gather（数值等价，最大偏差 1.2e-7）、统计段 24 维向量一次拷回（整帧唯一一次
`.cpu()` 同步）、布尔掩码索引改索引/整幅算术、v3 三个辅助流程用主机侧计数门控。

实测（同进程 A/B，covid19 10k）：

| 指标 | before | after | 变化 |
|---|---|---|---|
| torch profiler aten 事件数/帧 | 1222 | 967 | **−21%** |
| 纯计算 A/B（`SIM_RENDER_EVERY=0`，8×150 帧中位数） | 17.434 ms/帧 | 16.370 ms/帧 | **−6.1%** |
| 含渲染 A/B 第 1 轮（5×120） | 15.756 | 13.679 | −13.2% |
| 含渲染 A/B 第 2 轮（8×150） | 17.331 | 17.498 | +1.0%（落在噪声内） |

同机噪声地板实测为 ±2%（两边都指向同一入口时的 A/B 差值），且同一条轨迹在单次
运行内可从 12 ms 漂到 19 ms。因此**单世界墙钟收益无法稳定量化，报告以"算子数
−21% + 纯计算 −6.1%"为准**，不宣称更高的百分比。

由此，**验收口径从"逐位回归"改为"分配守恒 + 统计区间 + 确定性复跑"**：
正式测试套件已移除；`golden_*.json` 保留为历史快照与长跑对照，只作参照。

### 6.2c 第三轮优化（Wave 1：整幅化 / 索引复用 / 同步消除）

本轮按"减少每帧 kernel 数与隐式同步"推进，均为语义等价的实现替换（不改变状态迁移、
概率、曲线、年龄表、床位规则与可视化 contract）；由于前一轮已取消逐位验收，
随机数流允许改变，但仍保持每帧抽取的**次数与顺序结构**不变。

改动要点：

- **活动模型**：停留段整幅化（去掉 `nonzero` + 3 次 gather + 2 次 index_put）、
  位置 clamp 合并为一次 elementwise、旅途推进改用 `torch.lerp`、`desired` 用预分配
  缓冲原地清零、`traveling/trip_count` 原地写、`torch.add_(bool)` 免 dtype 转换。
- **病程/传播**：`_inf_lut` 一次 gather 取代 3–5 个比较/或；住院空床数改用主机侧
  记账（省每帧 2 次 `.sum().item()` 同步）；`p_pair` 先合成感染者侧系数
  （少 2 次 gather、2 次乘法）；每帧 `torch.tensor(...)` 常量改用标量版 `torch.where`。
- **统计段**：状态以 `uint8` 传输后查颜色 LUT（1M 下传输量 8 MB→1 MB）；再感染率
  前缀和只计算显示窗口所需尾部（长跑不再每帧 O(T) 累加）。

同进程交替 A/B（`tools/ab_iter.py`，两进程交替、每轮 240 帧、丢弃前 2 轮、取配对
中位比；对照 = 本轮改动前的工作区快照，测量完成后已删除，可用 `git stash`/备份重建）：

| 场景 | before | after | 配对中位变化 | 备注 |
|---|---|---|---|---|
| covid19 10k | 9.183 ms/帧 | 8.415 ms/帧 | **−11.8%** | 早期帧工作集接近，可比性最好 |
| covid19 1M | 19.329 ms/帧 | 16.460 ms/帧 | **−14.7%** | 前者计数 S=997814，后者 S=997788 |
| toy 10k | 11.528 ms/帧 | 11.240 ms/帧 | **−4.0%** | 两侧 RNG 流分叉，末帧感染态不同（E+I 1180 vs 1036），工作集不可比 |

operator 级证据（torch profiler，covid19 10k，第 20–50 帧）：

| 指标 | before | after |
|---|---|---|
| aten 事件数/帧 | 978 | 863（**−11.8%**） |
| `aten::copy_`/帧 | 39 | 27 |
| `aten::index`/帧 | 36 | 31 |
| `aten::_index_put_impl_`/帧 | 7 | 6 |

说明：同机噪声地板（两侧都指向同一份代码时）实测约 ±3%，因此只有超过 3% 的配对差
才算真实收益；toy 的配对值受轨迹分叉污染，此处只作参考，不作为结论依据。

### 6.2d 第四轮优化（Wave 2：算子瘦身 / 接触对 / 渲染，2026-10-31 深夜）

前提变更：用户明确"不再要求逐位一致"（m00995），因此本轮以**每帧算子数与 CPU 派发
时间**为目标，允许随机数流改变，但状态迁移、曲线、概率、年龄表、床位规则与可视化
contract 全部不变。

#### W2-A 算子瘦身（semantic-preserving）

决定性诊断（§6.7）显示每帧 877 个 kernel、CPU 占比 76–97%，故本轮直接削 kernel 数：

- **删除不可观测写**：没有任何代码读取 `virus_*` 字段；传染力只按 E/I/A
  索引读取，均值只对感染者求，再感染时全量覆写 → **死亡/康复个体的 5 个病毒字段清零
  整段删除**（疫情期每帧 20–45 次 `index_put`）。
- **掩码去重**：`status == 3` 每帧只算一次（隔离倒计时/社区结局共用）；
  `advance_mobility` 死亡清理改 `logical_and_` + `masked_fill_`。
- **变异块合并**：三组 `beta/gamma/mu` 变异由"3×(clone+掩码写+clamp)"改为
  `(3, n)` 批量 `stack → rand → 加性 delta → clamp_ → 3 次写回`（约 39 → 17 kernel），
  常量（scale/bounds）首次调用时按档案缓存。
- **免疫增益**：改为 `index_add_` + `clamp_`（少 2 个 where 链）。
- **系数折叠**：`p_pair` 的感染者侧系数 `beta_factor*status_mult*(BASE*lockdown)` 先合成，
  少 2 次 gather、2 次乘法。
- **v3 路径索引化**：`_process_tests`/`_resolve_traces`/`_advance_quarantine` 去掉
  `bool(x.any())` 同步与整幅 N 的 `sel` 布尔数组（改为 `nonzero` + `index_select` +
  `index_put`）；`_expand_members` 直接 `(key>=0).nonzero()`；`_daily_step` 用一次
  `sum().item()` 兼作判空与名额。
- **常量提升**：网格邻域步长/空索引提为模块常量（此前每帧重建）。

| 场景 | before | after | 配对中位变化 | 备注 |
|---|---|---|---|---|
| covid19 10k | 14.365 ms/帧 | 11.591 ms/帧 | **−15.0%**（medians −19.3%） | 达标（目标 −8~12%） |
| covid19 1M | 25.505 ms/帧 | 22.515 ms/帧 | **−8.3%**（medians −11.7%） | 达标 |
| toy 10k | 17.571 ms/帧 | 15.506 ms/帧 | −22.4%（medians −11.8%） | 轨迹分叉，参考值 |

算子证据（`tools/diag.py`，covid19 10k）：内核数 **877 → 833（−5%）**，仍未达到
"≤760"的预设目标——剩余 kernel 主要分布在 `simulation.js` 的病程/传播细粒度掩码写与
F 段统计拼装，继续压缩需要"活跃集合索引化"重构（改变数据布局与遍历方式），风险高于
本轮收益，故留待下一轮。

#### W2-B 接触对（parity 先行）

- 在 5 组构造状态（均匀/聚集/边界/重合点/空集）上比对 dense 与 grid 的
  `(src, tar)` 集合完全相等、grid 无重复对（校验脚本已删，属一次性验证）。
- `_contact_pairs_grid` 改为**带 1 格外边距的环形步长**：越界邻格自然落到计数为 0 的
  padding 格，省掉 4 个比较 + 3 个与 + 1 次 cast；邻格偏移步长 `dx + dy*gxp` 按 jxp
  缓存在模块字典。

| 指标（covid19 1M，`tools/phase_probe.py`） | before | after |
|---|---|---|
| 接触段 | 4.216 ms/帧 | **3.617 ms/帧（−14.2%）** |
| 整帧配对 A/B（1M，W2-A+B 合计） | 18.329 ms | 17.572（medians −4.1%，配对 −1.5%≈噪声地板） |

即接触段本身达到 −14%，但在 1M 整帧里只占 ~20%，故整帧收益被噪声淹没；10k 的 dense
路径（`cdist`+阈值+`nonzero`）实测已接近最优（`cdist 1.38 + lt 0.41 + nonzero 0.39 ms`），
未做改动。

#### W2-C 渲染解耦

- 主散点 marker 由默认圆形改为**小方块**（`s=2, marker='s'`）：Agg 每个 marker 的顶点数
  约降到 1/6，2 px 下肉眼几乎无差别，15 artist / 4 axes 的可视化 contract 不变。
- 新增 `--async-render[=K]`：模型逐帧推进、每 K 帧重绘（默认 K=2），与 `--render-every`
  同源但语义更明确；`--render-every` 优先。

| 10k 交互 blit 口径（`tools/render_prof.py covid19 10000 40 K`） | 优化前 | marker='s' | marker='s' + K=2 |
|---|---|---|---|
| 每帧绘制耗时 | 37.29 ms/draw | **22.21 ms/draw（−40%）** | 24.14 ms/draw |
| amortized/帧 | 47.38 ms（21.1 fps） | 30.60 ms（32.7 fps） | **20.90 ms（47.9 fps，1.57×）** |

1M + 10000 抽样点在 marker='s' 后为 30.14 ms/draw、amortized 47.28 ms/帧。
至此"默认 10k 绘制 ≤40 ms"与"`--async-render` 交互 ≥1.5×"两项 W2 承诺均达成。

当时全量回归 6/6 PASS（v4 之前口径，测试套件已移除，数字仅作历史参照）。

### 6.2e 第五轮：schema v4 个体异质性与接触混合（2026-11-01 凌晨）

这一轮**不是性能优化**：它补的是"方差"——均匀混合 + Poisson 型传播几乎不会熄火、
也长不出超级传播者，而"同 R0、不同个体差异 → 有的熄火有的爆发"正是本 toy model 想讲的
教学点。实现清单与原理见 [MECHANISMS.md](MECHANISMS.md) §12，这里只记录**缩放与成本**。

**成本**：v4 全部改动都在关闭路径之外，关闭时零新增 kernel（toy 300 帧 counts 在 v4 前后
逐位不变：`[18,102,820,6659,1632,769,0,0]`）。开启时每帧新增约 6–8 个算子
（感染者侧 `virus_scale` gather+乘、pair 级 `same_hh` 比较 1 + gather 2 + 乘 1、
症状居家一次 `(status==3) & symp_home` 合并），相对 ~890 kernel 约 1%：

| 10k，covid19，400 帧，`--no-draw`（`benchmarks/gate_bench_v4_*.json`） | update mean | CUDA 峰值 |
|---|---|---|
| v4 开（`covid19`） | 13.229 ms | 12.51 MB |
| v4 关（`covid19_homogeneous`） | 12.734 ms | 12.37 MB |

→ 差值 +0.50 ms/帧（+3.9%），其中一部分来自"异质性改变了轨迹与工作集"（感染更多 →
传播段更重），而不是纯算子开销；仍在本机 A/B 噪声地板（±3%）上沿。

**R0 不变量**：年龄同配矩阵满足 `Σ_g Σ_h n_g n_h φ = 1`、`virus_scale` 归一化到样本均值 1、
compliance 只改阈值不改随机数个数 → `python tools/profile_check.py --profile covid19`
的解析 R0 仍 **3.00**，代际 4.54 天、Euler–Lotka 倍增 2.34 天不变。

**12-seed ensemble 分位带**（10k，3000 帧 = 12.5 天；审计修复后重跑，
`benchmarks/ensemble_covid19_10k_12_after_audit.json`）：

| 指标 | p10 | p50 | p90 | min | max |
|---|---|---|---|---|---|
| 攻击率 | 0.033 | **0.040** | 0.088 | 0.013 | 0.092 |
| 峰值感染 | 295 | 368 | 812 | 112 | 862 |
| 峰值住院 | 0 | 1 | 3 | 0 | 3 |
| 累计感染 | 307 | 384 | 858 | 110 | 898 |
| 隔离总量 | 316 | **346** | 467 | 194 | 630 |
| 死亡 | 0 | 0 | 0 | 0 | 0 |
| 达峰天数 | 12.5 | 12.5 | 12.5 | 12.5 | 12.5 |

→ 攻击率跨度 1.3%–9.2%（约 7×），首批 1–2 个超级传播者是否出现决定了整条轨迹，
这正是 v4 想表达的行为。（修复前同一实验：隔离总量恒 0、攻击率 p50 0.050——
追踪失效时传播略高；修复后追踪压低了传播，见 §6.2f。）

**超传播过散布**（逐感染者二代计数，`--track-secondary`；审计修复后口径）：

| 口径 | var/mean p10/p50/p90 | k 估计中位 | 零二代 | ≥5 二代 |
|---|---|---|---|---|
| completed（传染期已结束，主口径） | 1.86 / **2.62** / 5.26 | **1.31** | 24.9% | 10.4% |
| all（含仍在传染期者，仅参考） | 2.14 / 2.73 / 3.42 | 0.56 | 56.9% | 3.8% |
| 同质对照 covid19_homogeneous（completed） | 1.08 / 1.21 / 1.28 | 9.6–26 | — | — |

⚠️ **口径警告**：只有"传染期已结束"的感染者适合做过散布统计——把仍在 E/I/A 的人算进去
会被尾部截断压平/扭曲方差（同质对照也会 >2），`tools/ensemble.py --track-secondary`
采用 completed 口径。

**分层代价**（1M 时新增张量）：`virus_scale`/`compliance`/`symp_home` 各 1 个 float32/bool
向量，1M 合计约 9 MB；`mix_flat` 仅 16 个元素。GPU 显存峰值基本不变。

### 6.2f 审计修复对 covid19 轨迹的影响（2026-11-01）

一次全量隐藏 Bug 审计修了 3 个 P1 级问题（详见 [MECHANISMS.md](MECHANISMS.md) §7.3 与
[README.md](README.md)「已知限制」）：

| 缺陷 | 机制 | 修复后的可观测变化 |
|---|---|---|
| 床位释放漏算 | D.3/D.4 计算空床时只扣死亡、不扣同帧康复出院者 | v2_noiv 400 帧 counts `[9977,0,23]`→`[9969,0,31]`（补收更多 `untreated`）；covid19 300 帧不变 |
| 追踪死锁（双重） | `trace_timer` 无递减 + D.5 门控 `==0` 恒假 → `_resolve_traces` 永不执行 | 隔离总量 0 → **p50 346**；攻击率 p50 0.050→0.040；二代 var/mean p50 2.32→2.62（k 1.31） |
| 死者隔离标记泄漏 | 死亡路径不清 `quarantined`/`quarantine_timer` | 隔离计数与标题 Q 不再含死者；`_advance_quarantine` 不再被死者吊住 |

- 关闭路径（`toy`、`covid19_v2_noiv` 的社会响应）**不受影响**：递减不消耗随机数，
  `_resolve_traces` 只在 `TRACING_ON` 时执行，toy 300 帧 counts 仍逐位 `[18,102,820,6659,1632,769]`。
- 性能：追踪真正生效后每帧多出 `_resolve_traces` ~2–4 个 kernel；10k 单帧
  `update mean` 仍在 12–14 ms 量级（12-seed 复跑 20–24 ms/帧是 3 进程并行争 GPU 的口径）。
- 回归锁（已删）：当时有 4 个用例（床位记账、死者标记、2600 帧追踪长跑、toy 不变式）。

**第二批修复（同日，非轨迹级）**：同一轮审计继续修了派生层与"验证工具自身"的问题，
它们不改变 covid19 轨迹，但会崩非默认档案、或让"没跑"看起来像"通过"：

| 类别 | 问题 | 修复 | 可观测影响 |
|---|---|---|---|
| 派生层 | `hospitalization_rate=0`（schema 允许）在派生阈值前就除零，`apply` 直接抛 `ZeroDivisionError` | 阈值改为 `beds_per_1000*10/hosp_rate if hosp_rate>0 else 0.0` | covid19 阈值仍 **375.0**；0 住院率档案可正常应用 |
| 派生层 | 阈值重复计算两次，第二次忽略 `medical.beds_per_1000` 覆盖值 | 合并为一次、统一用覆盖值 | 有覆盖值的档案阈值与床位口径一致 |
| 状态层（潜在） | 坐标裁剪两个轴都用 `WIDTH`；`WIDTH/HEIGHT` 是独立键，非方形世界 y 越界 | `_clip_world(pos, margin, W, H)` 按轴裁剪 | 默认 `W==H` 逐元素等价 → 轨迹不变 |



单进程吃不满 GPU（见 §6.6），用 K 个独立进程并行跑不同随机种子可提升吞吐。
covid19、600 帧（10k）/300 帧（1M），`SIM_RENDER_EVERY=0`（纯计算，无绘图）：

| 人口 | 并行度 K | 单进程 ms/帧 | 总吞吐 帧/s | 加速比 | GPU util | 功耗 | 显存峰值 |
|---|---|---|---|---|---|---|---|
| 10k | 1 | 12.05 | 83.0 | 1.00× | 11.5% | 13.7 W | 515 MB |
| 10k | 2 | 11.49 | 174.1 | 2.10× | 80.3% | 28.9 W | 757 MB |
| 10k | 4 | 15.86 | 252.3 | 3.04× | 90.5% | 31.1 W | 1018 MB |
| 10k | 8 | 28.06 | 285.0 | 3.43× | 95.1% | 33.0 W | 1587 MB |
| 1M | 1 | 21.43 | 46.7 | 1.00× | 37.7% | 33.8 W | 999 MB |
| 1M | 4 | 52.30 | 76.5 | 1.64× | 95.4% | 54.0 W | 2725 MB |

（上表为第三轮优化后的最新测量，原始数据 `benchmarks/sweep_w1_covid19_10000.json`、
`sweep_w1_covid19_1000000.json`；第二轮对应数据见 `sweep_covid19_*.json`：10k K=1 13.06 →
本次 12.05 ms（+8% 吞吐）、K=8 270.7 → 285.0 帧/s；1M K=1 21.72 → 21.43 ms、K=4 68.3 →
76.5 帧/s。）

原始数据：`benchmarks/sweep_covid19_10000.json`、`sweep_covid19_1000000.json`。
读法：10k 在 K=2 后 GPU 已近饱和（util 76%→89%），加速比来自"填满 kernel 空隙"；
1M 单进程已能推进 36% 利用率、K=1 就比优化前快 30%（31.17→21.72 ms），因此
并行加速比数字变小，但**绝对吞吐**仍是上升的。定位到"K* 建议"：10k 用 K≈4–8，
1M 用 K≈2–4（受显存与功耗上限约束，非算力）。

### 6.5 渲染的三种口径与节流

同一份绘图代码在不同调用协议下成本差 2–3 倍，报告中必须区分：

| 口径 | 调用方式 | 10k 实测 |
|---|---|---|
| 全量重绘（`bench.py` 默认） | 每帧 `fig.canvas.draw()` + `blit(artists)` | 133–232 ms/帧 |
| 交互 blit（`FuncAnimation(blit=True)`） | 缓存背景 → 每帧 `restore_region` + `draw_artist` + 每 axes 一次 `blit` | **22.2 ms/帧**（marker='s' 后；其中 update 8–10 ms） |
| 节流渲染（`--render-every K` / `--async-render`） | 模型每帧推进，只每 K 帧重绘 | **K=2: 20.9 ms/帧（47.9 fps）**；K=4/8 见下 |

逐 artist 归因（10k，marker='o' 时）：主散点 PathCollection 39.6 ms、三行 Text 14.7 ms、
8+5 条 Line2D 共 2.0 ms。第四轮把主散点改成小方块后 PathCollection 降到约 7–9 ms
（绘制 37.3 → 22.2 ms/draw），Text 仍是第二大项（每帧标题都会变，无法跳帧）。
`100k` + 10000 点 + K=2 时 amortized 37.3 ms/帧（10k 等价人口下约 27 fps，为第三轮数据；
第四轮后同口径约 25–28 ms）。

用法：

```powershell
python simple_disease_simulation.py --profile covid19 --render-every 4
python simple_disease_simulation.py --profile covid19 --async-render      # 默认 K=2
$env:SIM_RENDER_EVERY="2"; $env:SIM_RENDER_POINTS="5000"; python simple_disease_simulation.py

# 复现上表（交互 blit 口径 + 节流）：profile / population / frames / K / points
python tools/render_prof.py covid19 10000 40 2
python tools/render_prof.py covid19 100000 20 2 10000
```

`--render-every` / `--async-render` 只影响重绘频率，不影响模型推进（每帧都 `update`），
因此统计、标题与曲线数据仍然逐帧更新，只是画面每 K 帧刷新一次。

### 6.6 为什么单进程吃不满 GPU

RTX 4060 Laptop（24 SM，TGP≈115 W）实测：10k 单进程纯 update 循环 GPU util
16–31%、功耗 8–19 W；1M 时 util 31–44%、功耗 27–37 W；而 4/8 个进程并行可把
util 推到 91–99%，功耗仍只有 31–33 W（10k）/ 55–59 W（1M）。即 **GPU 被"大量
小 kernel 的 launch 开销"占满，而不是被算力占满**：每帧约 1000 个 aten 算子，
每个只处理 10k–1M 元素的小张量，Windows/WDDM 下每次 launch+隐式同步的固定
成本与算术量同量级。

这是设计取舍而非缺陷：

- 模型语义要求逐帧串行（第 t 帧的状态决定 t+1 帧的随机数抽取），无法把多帧
  批成一个大 kernel；跨个体向量化已经做了，跨时间并行会改变语义。
- 融合内核/`torch.compile` 需要 triton（本环境未安装，属新增依赖）；按项目约定
  不引入新依赖。
- 交互模式的主成本是 matplotlib 的 CPU 侧绘制（§6.5），GPU 在绘制期间空闲。
- 要真正压榨 GPU，需要"多世界并行"（本仓库 `tools/sweep.py` 已实现进程级并行，
  见 §6.4）或"单进程内 B-world 向量化"（设计见 `docs/ARCHITECTURE.md` §7，
  未实现）。

复现：`python tools/gpu_util.py <population> <frames>`（纯 update 循环，外部
`nvidia-smi` 采样利用率/功耗），或 `python tools/sweep.py ... --jobs-list 1,2,4,8`。

### 6.7 决定性诊断：CPU 派发才是单帧瓶颈（2026-10-31 晚）

用 `tools/diag.py`（同时测"逐帧 sync 的 wall"、"不逐帧 sync 的 CPU 入队速率"、
"进程 CPU 时间"、"CUDA 内核数与时长的 CUPTI 分布"、cProfile、主机同步点）
在三个负载上做的时间账：

| 负载 | wall 中位 | 入队/帧 | CPU 时间/帧 | CPU 占比 | 内核数/帧 | 主要结论 |
|---|---|---|---|---|---|---|
| covid19 10k | 14.8 ms | 15.4 ms | 11.7 ms | 76% | **877** | CPU 受限 |
| toy 10k（疫情满负荷） | 11.3 ms | 10.9 ms | 10.6 ms | **97%** | **1130** | 纯 CPU 受限 |
| covid19 1M | 16.1 ms | 16.7 ms | 15.6 ms | **93%** | **1444** | 仍是 CPU 受限 |

> 第四轮（W2-A）后复测 covid19 10k：内核数 **877 → 833（−5%）**，wall 中位 9.3 ms、
> CPU 入队 8.4 ms —— 每算子 CPU 成本仍是 9–13 µs 量级，说明"少一个算子就省一份固定成本"
> 的规律仍然成立，但要再降 30% 需要活跃集合重构（见 §6.2d）。

换算下来 **每次 torch 算子的 CPU 侧成本约 9–13 µs**（toy：10.55 ms / 1130 ≈ 9.3 µs；
covid19 10k：11.7 / 877 ≈ 13 µs）。而 GPU 侧 util 只有 16–44%（§6.6），说明 GPU
大部分时间是空闲的，**瓶颈是"每帧几百上千次 launch/派发"，不是算力**。
CUPTI 报的"GPU 内核时间"在该量级下被记录开销放大 4 倍以上（>wall），不可直接引用，
只用于**算子计数与相对归因**。

单帧算子构成（covid19 10k，次/帧）：`as_strided` 110、`view` 45、`to` 41、
`reshape` 40、`empty` 38、`resize_` 37、`index` 31、`copy_` 27、`__and__` 27、
`bitwise_and` 27、`eq` 23、`empty_strided` 21。即约 **330 个是视图/分配类元操作**
（纯 CPU 元数据，~1–2 µs 每个），真正带 GPU kernel 的约 550 个。

推论（Wave 2 的靶子因此不是"算法"，而是"算子数"）：

1. 视图/分配类元操作可减半：复用 scratch 缓冲、避免每帧 `torch.stack/cat/zeros`、
   避免同一 dtype 反复 `to()`（41 次/帧）。
2. 布尔掩码类（`eq`+`and` 共 77 次/帧）可在帧首算一次并跨段复用；
   已有 `_inf_lut` 这条路线可继续（`_alive`/`_symp` 等）。
3. toy/1M 的额外 ~250–500 个算子主要来自 grid 接触对（`repeat_interleave`、
   `bincount`、9 邻格展开），是接触对优化的直接证据。
4. 主机同步已经不是问题：每帧仅 3 次 `.cpu()`（统计向量 + pos + status），
   合计 0.2–1.1 ms，不值得再动。

复现：`python tools/diag.py <profile> <population> <seed>`，JSON 落盘
`benchmarks/diag_<profile>_<population>.json`。

### 6.8 长跑金标（历史采集，2026-10-31）

> 本节采集于 v4 机制与后续优化**之前**，只作"量级与长跑合理性"参照，
> 不代表当前代码的绝对耗时或末帧 counts（口径差异见 §6.2 与 §6.7）。

| 金标 | 帧数 | update 均值 | CUDA 峰值 | 末帧 counts `[S,I1,E,I,D,R,A,H]` |
|---|---|---|---|---|
| `golden_v3_3000.json` | 3000（12.5 天） | 22.101 ms | 30.91 MB | `[9537,0,312,61,0,46,44,0]` |
| `golden_v3_10800.json` | 10800（45 天） | 29.482 ms | 125.81 MB | `[2405,0,422,779,30,5970,364,30]` |

45 天结果：累计感染 7595/10000（75.95%），死亡 30，住院 30（床位 30/30，`Med: COLLAPSE`），
免疫 0.357，再感染率 16.7%。**注意**：这批金标采集于 S1 优化之前（当时仍逐位
可复现），优化后 numpy/torch 的浮点运算顺序与 RNG 路径改变，长跑数值会与上表
有统计等价但非逐位的差异；上表仅作"长跑数量级与崩塌点"的历史参照。

- 验收：当时用统计式回归（分配守恒/量级/重跑一致）与一套结构断言核对；S1 之前还跑过
  toy 300、covid19 300/500、`covid19_v2_noiv` 400 的四档逐位对照（见 MECHANISMS.md §11）。
- v3 社会响应的额外开销集中在**日界**：每天一次 `_daily_step`（归档、报告抽样、
  行为乘子、检测排程），检测排程只对"当日有症状且愿意检测"的子集做 argsort；
  检测出结果/追踪到期也是按标记子集操作（S1 优化后进一步用主机侧计数整块跳过
  空分支）。传播循环里 v3 只多乘若干因子，不增加接触对数量。因此 v3 的
  **唯一新增缩放量是检测能力随人口线性放大**（80 → 800 → 8000 人/日），
  其余延迟/乘子按帧不变。

## 7. 复现实验命令

```powershell
python -m py_compile simple_disease_simulation.py

# 入口冒烟（toy，30 帧）
python simple_disease_simulation.py --profile toy --frames 30

# 档案派生/解析式 R0 自洽
python tools/profile_check.py --profile covid19

# 多 seed 分布（分位带；--track-secondary 加超传播诊断）
python tools/ensemble.py --profile covid19 --population 10000 --frames 3000 --seeds 1-12 --jobs 4

# 性能：单世界 A/B 与多进程吞吐
python tools/bench.py --profile covid19 --population 10000 --frames 400
python tools/sweep.py --profile covid19 --population 10000 --frames 600 --jobs-list 1,2,4,8
python tools/diag.py covid19 10000 42        # 决定性诊断：wall/入队/CPU/kernel 数/aten 明细

# 渲染节流/抽样与环境变量旋钮
$env:SIM_RENDER_EVERY="2"; $env:SIM_RENDER_POINTS="5000"
python simple_disease_simulation.py --async-render
```


## 8. 已知限制

- **GPU 建议**：100k 以上在 CPU 上会非常慢（网格/掩码为逐帧全量操作），CUDA GPU
  是 100k–1M 的可行前提；显存需求约 370 MB 实测峰值（1M），另加少量逐帧临时量。
- **网格近似**：空间网格用 9 邻格枚举候选对，再按精确距离筛选（曾在 5 组构造
  状态上与 dense 比对集合一致）。10k 默认走 dense，不受影响。
- **渲染抽样**：大人口下散点只显示 10000 人（统计与曲线是全体人口）；如需更多
  可调 `RENDER_MAX_POINTS` 或 `SIM_RENDER_POINTS`，但绘制时间线性增加。
- **不再逐位可复现**：S1 之后同 seed 同配置的跑次之间，前几百帧通常仍逐位一致，
  但 v2/v3 机制介入后会出现浮点末位差异累积（LUT 曲线、统计向量合并）。确定性
  （同一份代码、同一 seed、两次独立进程结果相同）仍然成立。
- **医疗阈值/种子/检测能力按人口放大**：这是"保持 10k 版动态"的选择；若希望
  1M 城市配固定数量床位/检测能力（现实政策约束），需另行按政策改档案或缩放规则。
- **v3 社会响应是简化机制**：检测只覆盖有症状自愿者、无无症状筛查；追踪只覆盖
  家庭与同工作/学校场所；行为乘子全人群共享且只影响休闲接触/出行；详见
  [MECHANISMS.md](MECHANISMS.md) §9 与 [REALISM.md](REALISM.md) §5。
- **初始化内存**：1M 人口在导入期分配约 220 MB CUDA 张量并保留；初始化约 2-3 s。
- **确定性**：同一 `SEED`（或 `DISEASE_SEED`）下结果可复现（家庭/场所/初始感染
  均由种子驱动；v3 关闭时不消耗随机数）。
- **未做**：多 GPU / CPU 多线程 / 分块感染更新；人口 >1M 需进一步把移动与
  状态更新也分块。
- **工具口径**：`profile_check.py` 不感知 v3，经验 R0 按无社会响应路径校准；
  §6.1 为重构前基准、§6.2 为当前实现与抗漂移 A/B 对照，两者测量时刻不同、不可直接相减。
