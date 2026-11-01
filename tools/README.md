# tools/ —— 测量、门禁与观测工具

所有工具都**不修改模型、不写仓库源码**（只写 `benchmarks/` 下的 JSON 日志），
统一在仓库根目录运行、无头（`MPLBACKEND=Agg`）。测量类工具**不要并行跑**
（互相争 GPU 会让 util/耗时失真），跑之前先 `nvidia-smi` 确认没有别的重负载。

> 改代码/改性能/加机制前该跑什么，见 [../docs/CONTRIBUTING.md](../docs/CONTRIBUTING.md) §1 与 §5。

## 速查表

| 工具 | 回答什么问题 | 典型命令 |
| --- | --- | --- |
| `headless_check.py` | 结构/机制契约是否成立（38 项断言，**官方硬门禁**） | `python tools/headless_check.py --profile toy --frames 300` |
| `profile_check.py` | 档案数值是否自洽、经验 R0 是否合理 | `python tools/profile_check.py --profile covid19 --empirical` |
| `golden_check.py` | 与历史快照比对（**诊断用，非门禁**） | `python tools/golden_check.py --contract` |
| `bench.py` | 单世界延迟/绘制/显存（`update` 含 sync，`draw` 为最坏口径） | `python tools/bench.py --profile covid19 --population 100000 --json benchmarks/b.json` |
| `ab_iter.py` | **A/B 改动是否真的更快**（两进程交替，抗漂移） | `python tools/ab_iter.py --before _base/simple_disease_simulation.py --profile covid19 --json benchmarks/ab.json` |
| `diag.py` | 瓶颈是 CPU 派发还是 GPU 算力 | `python tools/diag.py covid19 10000 42` |
| `phase_probe.py` | 一帧时间花在 mobility / 接触对 / 内联段各多少 | `python tools/phase_probe.py simple_disease_simulation.py covid19 10000 200` |
| `render_prof.py` | 交互 blit 口径的绘制耗时与节流摊薄 | `python tools/render_prof.py covid19 10000 40 2` |
| `sweep.py` | 多进程吞吐/加速比/util/功耗/显存矩阵 | `python tools/sweep.py --jobs-list 1,2,4,8 --population 10000 --frames 600` |
| `ensemble.py` | 多 seed 分位带与超传播过散布（观测层，不改模型） | `python tools/ensemble.py --population 10000 --frames 3000 --seeds 1-12` |
| `lut_check.py` | 传染力曲线 LUT 与解析式的数值误差 | `python tools/lut_check.py` |
| `gpu_util.py` | 纯 update 循环下单进程能压到多少 GPU 利用率/功耗 | `python tools/gpu_util.py 10000 1200` |
| `_ab.py` | **旧版**同进程 A/B（已被 `ab_iter.py` 取代，勿用于正式结论） | `python tools/_ab.py toy 300` |

## 各工具的判读要点与已知偏差

### `headless_check.py`（门禁）

- 无头加载入口（`matplotlib.use('Agg')`、替换 `FuncAnimation`、`plt.show` 置空），
  经 `spec_from_file_location` 导入后跑 `frames` 帧并逐帧断言。
- **官方口径**：`toy` 300、`covid19` 300/500、`covid19_v2_noiv` 400，每次都必须
  "失败断言 0 项 / 共 38 项"；退出码 0=PASS、1=FAIL、2=参数/加载错误。
- 陷阱：张量属性缺失时部分检查会**静默 SKIP**（退出码仍是 0），所以不要只看退出码；
  也不要修改断言列表/名字/消息（用户侧契约）。
- `--json` 会落盘完整快照（counts/标题/轴限/计时），适合做跨版本对照证据。

### `golden_check.py`（历史快照，非门禁）

- `--dump` 记录张量指纹、RNG 状态、CONFIG 摘要、逐帧 counts/标题；`--compare` 比对；
  `--contract` 只检查模块级契约名字与 artist 数量是否齐全。
- 自第二轮性能优化起**不再逐位相等**（浮点运算顺序改变），因此比较失败只意味着
  "行为变了，去查是否预期"，不是"代码坏了"。
- 复刻快照时必须严格遵守探针协议：`update(0)` 一次 → `range(1, frames)`，
  与 harness 完全一致；否则整条轨迹偏移。`.bak` 不能被 `spec_from_file_location`
  直接加载，需要复制成 `.py`（如 `_orig_sim.py`）。

### `bench.py`（单世界延迟）

- 与 harness 口径不同：`update` 计时**含 `torch.cuda.synchronize()`**，
  `draw` 是**每帧 `canvas.draw()` 全量重绘**（真实 GUI 用 `FuncAnimation(blit=True)`，
  便宜得多；harness 用 `restore_region`+逐 artist blit）。
- 绝对值跨时段/跨机器**不可比**（笔记本热漂移可达 ±30%）。只用于同会话趋势；
  要证明"变快了多少"用 `ab_iter.py`。

### `ab_iter.py`（A/B 门禁）

- **两进程**逐轮交替（同进程会因 `sys.modules` 让两侧共用同一份包），丢弃前几轮预热，
  输出 before/after 中位数与**配对中位比**；低于 ~3% 视为噪声。
- `--before` 指向一个历史快照入口（形如 `_base/simple_disease_simulation.py`，
  其目录内需有 `disease_sim/` 与 `diseases/`）。
- 渲染开关：吞吐/纯计算对比时用 `--no-draw` 或设置 `SIM_RENDER_EVERY=0`。

### `diag.py`（瓶颈定位）

- 输出每帧 wall、CPU 入队时间、主机同步次数、kernel 数/帧与 top aten 算子。
- **CUPTI/CUDA profiler 报的 GPU 时间会被放大（可 > wall）**，只能看相对占比与计数；
  判断"CPU 派发受限"的依据是"CPU 入队时间 ≈ wall"，而不是 GPU util 低。

### `phase_probe.py`（分段归因）

- 在每段前后 `synchronize()`，因此会把上一次入队的工作算进当前段：**只能同版本内
  相对比较**。warm 后必须清零计数器（否则 warm 的调用会把段耗算进去）。

### `sweep.py`（多进程吞吐）

- 父进程为每个 `K` 启 K 个独立子进程（各自加载完整模型），等全部就绪后开始计时，
  并行读 `nvidia-smi` 采样 util/功耗/显存。
- 输出的是**吞吐**（帧/秒 与加速比），不是单世界延迟；K=1 基线也会随机器状态漂移，
  报告里要写"同会话内相对比 + 绝对帧率"。

### `ensemble.py`（观测层）

- 每个 seed 一个子进程（GPU 上下文隔离），汇总 p10/p50/p90，可出 CSV/PNG。
- **二代统计两套口径**：`all`（所有曾感染者）与 `completed`（传染期已结束者）。
  默认看 `completed`：仍在 E/I/A 的感染者尚未产生完整二代，右删失会压缩方差，
  把 overdispersion/负二项 k 估歪。
- `--png-from-json` 只重画图，不重跑模型；图内标签用英文（DejaVu 无 CJK 字形）。

### `render_prof.py`（绘制）

- 复刻 `FuncAnimation(blit=True)` 的绘制协议（`restore_region` + `draw_artist` + `blit`），
  给出 draw 耗时与 `--render-every K` 的摊薄结果。与 `bench.py` 的全量口径不可混谈。

### `profile_check.py`（档案）

- 解析路径：检查派生参数与 `PROFILE_INFO.derived` 自洽（R0、代际、倍增期）；
  `--empirical` 路径：从模拟轨迹反推增长率并折算 R0，容差 30%。
- 它**不感知 v3/v4 机制**（只做宏观量自洽），所以 PASS 不等于机制正确。

### `lut_check.py` / `gpu_util.py` / `_ab.py`

- `lut_check.py`：LUT 版 `_curve_eval` 与解析版在采样网格上的最大误差（当前 < 1.2e-07）。
- `gpu_util.py`：纯 update 循环 + 外部采样，用于说明"单进程吃不满 GPU"是 launch 受限。
- `_ab.py`：项目早期同进程 A/B，`sys.modules` 冲突导致**只能一侧是单文件实现**，
  已被 `ab_iter.py` 取代；保留以便重现历史数字。

## 产物位置

所有 JSON/CSV/PNG 都写在 `benchmarks/`：`bench_*.json`、`ab_*.json`、`sweep_*.json`、
`ensemble_*.{json,csv,png}`、`diag_*.json`、`phase_*.json`、`golden_*.json`（历史快照）。
命名约定：`<类型>_<主题>_<profile>_<population>.json`，文档里引用文件名而不是口头数字。
