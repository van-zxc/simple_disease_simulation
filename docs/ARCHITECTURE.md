# 架构说明（disease_sim 包）

本文件说明重构后的模块划分、依赖方向、一帧的数据流、兼容层原理、渲染路径与
"多世界并行"的设计草案。

## 1. 目录结构

```
simple_disease_simulation.py     # 薄入口：装配 + 给分析工具转发模块级名字（~100 行）
disease_sim/
├── __init__.py                  # build_simulation(argv) = init_session → build_state → build_render
├── config.py                    # CONFIG 缺省值、档案应用、env 旋钮、人口缩放
├── state.py                     # SimState 容器 + build_state()（所有逐帧可变状态）
├── activity.py                  # 日程/出行（resample_daily_plan、advance_mobility）
├── contacts.py                  # 接触对枚举（dense 全对 / grid 空间网格）
├── disease.py                   # 传染力曲线（dense LUT + lerp）
├── interventions.py             # v3：检测、追踪、行为反馈、报告延迟
├── simulation.py                # update(sim, frame)：一帧编排 + 传播/统计
├── render.py                    # 图元、按钮、toggle
├── cli.py                       # FuncAnimation 入口（含 --render-every / --async-render）
└── profiles/                    # schema / loader / validate / derive / apply
diseases/                        # 档案 JSON（covid19 v4 / covid19_homogeneous / v2_noiv / toy）
tools/                           # 分析/测量工具（bench / diag / ab_iter / sweep / ensemble …）
benchmarks/                      # 基准与多 seed 结果（本地数据，不入库）
```

## 2. 模块职责与依赖方向

```
cli ─→ simulation ─→ {activity, contacts, disease, interventions}
  │         │
  │         ├─→ render（图元更新）        └─→ state / config
  └─→ config
profiles：完全独立（只依赖标准库）
```

- 依赖单向、无环；`state`/`config` 是叶子（不 import 其它业务模块）。
- `update(sim, frame)` 里所有可变状态都挂在 `sim` 上，阶段函数统一签名 `f(sim, ...)`，
  便于单测直接构造 `SimState` 的瘦身副本。
- 连续量（家庭/场所位置、年龄表）在 `build_state()` 里一次性构造；逐帧只写
  `status`/`timer`/`pos`/`place` 等热字段。
- 注释与 docstring 约定沿用模块 docstring + 四类行内标记（RNG/性能/调用方/不变式），
  没有单独的流程文档。

## 3. 数据流（一帧）

```
update(sim, frame)
  ├─ 时钟/日界：clock_min、day_index；日界时 _daily_step（归档、报告抽样、行为乘子）
  ├─ 主机计数门控：n_E/n_A/n_H/n_active_inf/_n_restricted/... （来自上一帧统计向量）
  ├─ advance_mobility(sim, frame, clock_min, day_index)   # 出行/到岗/随机游走
  │    └─ v4：MIX_ON 时 (status==3)&symp_home 进入 homebound（症状居家）
  ├─ progression：免疫衰减 → 潜伏期→发病/无症状 → 自愈 → 隔离倒计时
  │               → 社区死亡/康复 → 住院/出院/床位 → v3 检测与追踪
  │               └─ v4：隔离/追踪概率按个体 compliance 缩放（只改阈值）
  ├─ transmission：接触对枚举 → pair 级概率 → 感染写回 → 变异/再感染计数
  │    └─ v4：感染者侧 ×virus_scale；pair 级 ×(同户系数, 年龄同配 φ)；新感染者抽新 scale
  └─ stats.finalize：counts/搬运一次 .cpu() → 标题/曲线/散点 → 写回 counts_prev 等
```

RNG 顺序契约（重构后仍然遵守，但**不再要求跨实现逐位一致**）：所有
`torch.rand/randn` 抽取的**次数与形状**由 `CONFIG` 开关组合决定；新增随机数必须
由功能开关包住，否则同 seed 的轨迹会整体偏移。S1 优化只改变"如何算"，不改变
"抽几次、按什么形状抽"。v4 沿用同一规则：`virus_scale`/`compliance`/`symp_home`
的抽取都在 `HET_ON`/`MIX_ON` 内，关闭档案（toy、v2_noiv）零额外随机数。

## 3b. v4 状态字段与观测层

- `state.py` 的 v4 块（在 v3 状态之后）：`HET_ON`、`MIX_ON`、`sigma_infectious`、
  `virus_scale[N]`、`compliance[N]`、`_secondary[N]`（仅 `SIM_TRACK_SECONDARY=1`）、
  `age_group[N]`、`mix_flat[16]`、`hh_mult`、`symp_home[N]`（MIX_ON 时）与
  `age_group/mix_flat/symp_home=None`（关闭时）。
- `tools/ensemble.py` **不改模型**：父进程为每个 seed 起独立子进程
  （`tools/_loader.load_module`，env 注入档案/人口/seed/渲染关闭），子进程逐帧
  `update` + `cuda synchronize`，每 50 帧记录 counts 序列，结束时打印
  `@@ENSEMBLE@@{json}`；父进程汇总 p10/p50/p90、可选 CSV/PNG 分位带。
  `--track-secondary` 时子进程额外统计逐感染者二代计数（分 `completed` /
  `all` 两个口径，前者才可用来度量超传播过散布）。


## 4. 兼容层（为什么分析工具不用改）

`tools/` 里的脚本用 `spec_from_file_location` 直接加载
`simple_disease_simulation.py`（模块 `__package__` 为空，因此入口自己把仓库根插入
`sys.path`），再按名字读取模块级属性（`status`、`fig`、`title_obj`、`update` …）。
薄入口通过模块级 `__getattr__` 把未知名字转发给唯一的 `_SIM`
（`SimState` 实例），标量走 lambda 实时读取，避免陈旧副本；`update(frame)` /
`toggle(event)` 是真实包装函数，签名与返回值不变。共享加载逻辑在 `tools/_loader.py`。

## 5. 渲染路径与节流

三条路径成本不同，报告时必须分清：

| 路径 | 调用 | 说明 |
|---|---|---|
| 全量重绘 | `bench.py`：每帧 `fig.canvas.draw()` | 最坏口径（133–232 ms @10k） |
| 交互 blit | `FuncAnimation(blit=True)`：缓存背景 → `restore_region` + `draw_artist` + 每 axes 一次 `blit` | 真实 GUI 口径（22 ms @10k，第四轮把主散点改成 `marker='s'` 后） |
| 节流 | `--render-every K` / `--async-render[=K]`：模型每帧推进，只每 K 帧重绘 | 10k：K=1 30.6 ms、K=2 20.9 ms（47.9 fps）、K=4/8 更低 |

实现要点：`state.render_every` 由 `SIM_RENDER_EVERY`、`--render-every` 或
`--async-render`（默认 K=2）设置；`cli._frame()` 在非重绘帧返回 `[]`
（blit=True 下即"本帧不画任何 artist"），模型状态仍逐帧推进，因此标题/曲线数据
不丢失。散点抽样上限 `CONFIG["RENDER_MAX_POINTS"]`（默认 10000，可用
`SIM_RENDER_POINTS` 覆盖）决定 PathCollection 的绘制成本；主散点用 2 px 小方块
后 10k 点约 7–9 ms/draw，剩余大头是三行标题文本（约 15 ms，每帧都会变）。

## 6. 常用命令

```powershell
# 冒烟：跑通并出图（Agg 下不弹窗）
python simple_disease_simulation.py --profile toy --frames 30
python simple_disease_simulation.py --profile covid19 --async-render

# 档案派生 / R0 / 代际时间自洽
python tools/profile_check.py --profile covid19

# 观测层：多 seed 分位带（不改模型；--track-secondary 打开逐感染者二代统计）
python tools/ensemble.py --profile covid19 --population 10000 --frames 3000 --seeds 1-12 `
    --json benchmarks/ensemble_covid19_10k_12.json --png benchmarks/ensemble_covid19_10k_12.png
python tools/ensemble.py --profile covid19 --population 10000 --frames 3000 --seeds 1-4 `
    --track-secondary --json benchmarks/ensemble_secondary_10k.json
# 只重画分位带图（复用已有 JSON，不重跑模型）
python tools/ensemble.py --png-from-json benchmarks/ensemble_covid19_10k_12.json `
    --png benchmarks/ensemble_covid19_10k_12.png

# 性能：单世界 A/B（抗机器漂移）、分段归因、决定性诊断与多进程吞吐
python tools/ab_iter.py --profile covid19 --population 10000 --frames 200 --rounds 8
python tools/phase_probe.py simple_disease_simulation.py covid19 10000 200
python tools/diag.py covid19 10000 42
python tools/bench.py --profile covid19 --frames 400 --population 10000 --json benchmarks/bench_after_10000.json
python tools/sweep.py --profile covid19 --population 10000 --frames 600 --jobs-list 1,2,4,8

# 渲染抽查（交互 blit 口径）
python tools/render_prof.py covid19 10000 40 2
```

## 7. 设计草案：单进程 B-world 向量化（未实现）

目标：在不引入新依赖的前提下，让单进程也能把 GPU 利用率从 16–44% 推到接近饱和。
思路是把"世界"变成张量的一个维度：

- 状态张量形状 `(B, N)`（B 个独立世界，各自一份 `SimState` 的展平视图），
  随机数按 `(B, N)` 一次抽取；B 个世界共享同一份档案与超参，只有 seed 不同。
- 接触对枚举按世界维度分块（`grid` 路径天然可批：`(B, n_inf)` 的索引平铺），
  一次 `cdist`/邻格展开处理所有世界，输出 `(B, P)` 的 pair 张量。
- 统计与标题按世界取 `counts` 矩阵 `(B, 8)`，一次 `.cpu()`；渲染仍只画 world 0
  （或按帧轮换），因此渲染成本不随 B 增长。
- 语义等价性：每个世界内部与当前单世界实现**逐位一致**（同一随机数消耗顺序），
  只是把 B 次 kernel 合成 1 次——因此可以用"同 seed 单世界 vs B 世界第 k 份"
  的行级对照来验证。
- 预期收益：kernel 数量不变但每个 kernel 的工作量 ×B，launch 开销被摊薄；
  代价是显存 ×B（1M 单世界约 0.37 GB，B=4 约 1.5 GB，仍在 8 GB 内）。

未实现的原因：实现量大（所有阶段函数都要接受批量维度），且当前 `tools/sweep.py`
的进程级并行已能把 GPU 推到 89–96%（见 SCALING.md §6.4），边际收益主要在
"少进程、低内存"场景。若后续要做，建议先在 `contacts.py` + `activity.py` 上做
B=2 的原型验证语义等价性，再逐步推广。
