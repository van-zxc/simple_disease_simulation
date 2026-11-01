# tools/ —— 测量与观测脚本

所有工具都**不修改模型、不写仓库源码**（只写 `benchmarks/` 下的 JSON/PNG），
统一在仓库根目录运行、无头（`MPLBACKEND=Agg`）。测量类工具**不要并行跑**
（互相争 GPU 会让 util/耗时失真），跑之前先 `nvidia-smi` 确认没有别的重负载。

> 这里的脚本用来看"跑多快、瓶颈在哪、多 seed 时结果分布如何"，不是质量门禁：
> 本仓库是研究型 toy model，不设自动化测试套件（2026-11 起移除）。

## 速查表

| 工具 | 回答什么问题 | 典型命令 |
| --- | --- | --- |
| `profile_check.py` | 档案数值是否自洽、经验 R0 是否合理 | `python tools/profile_check.py --profile covid19` |
| `bench.py` | 单世界延迟/绘制/显存（`update` 含 sync，`draw` 为最坏口径） | `python tools/bench.py --profile covid19 --population 100000 --json benchmarks/b.json` |
| `ab_iter.py` | **A/B 改动是否真的更快**（两进程交替，抗漂移） | `python tools/ab_iter.py --before _base/simple_disease_simulation.py --profile covid19 --json benchmarks/ab.json` |
| `diag.py` | 瓶颈是 CPU 派发还是 GPU 算力 | `python tools/diag.py covid19 10000 42` |
| `phase_probe.py` | 一帧时间花在 mobility / 接触对 / 内联段各多少 | `python tools/phase_probe.py simple_disease_simulation.py covid19 10000 200` |
| `render_prof.py` | 交互 blit 口径的绘制耗时与节流摊薄 | `python tools/render_prof.py covid19 10000 40 2` |
| `sweep.py` | 多进程吞吐/加速比/util/功耗/显存矩阵 | `python tools/sweep.py --jobs-list 1,2,4,8 --population 10000 --frames 600` |
| `ensemble.py` | 多 seed 分位带与超传播过散布（观测层，不改模型） | `python tools/ensemble.py --population 10000 --frames 3000 --seeds 1-12` |
| `gpu_util.py` | 纯 update 循环下单进程能压到多少 GPU 利用率/功耗 | `python tools/gpu_util.py 10000 1200` |

`_loader.py` 是共享加载器（Agg 打桩 + `spec_from_file_location` 加载入口），
被其它脚本 `import _loader`；单独运行没有输出。

## 各工具的判读要点与已知偏差

### `bench.py`（单世界延迟）

- `update` 计时**含 `torch.cuda.synchronize()`**；`draw` 是**每帧 `canvas.draw()`
  全量重绘**（真实 GUI 用 `FuncAnimation(blit=True)`，便宜得多）。
- 绝对值跨时段/跨机器**不可比**（笔记本热漂移可达 ±30%）。只用于同会话趋势；
  要证明"变快了多少"用 `ab_iter.py`。

### `ab_iter.py`（A/B 对照）

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

### `gpu_util.py`

- 纯 update 循环 + 外部采样，用于说明"单进程吃不满 GPU"是 launch 受限。

## 产物位置

所有 JSON/CSV/PNG 都写在 `benchmarks/`：`bench_*.json`、`ab_*.json`、`sweep_*.json`、
`ensemble_*.{json,csv,png}`、`diag_*.json`、`phase_*.json`、`golden_*.json`（历史快照）。
命名约定：`<类型>_<主题>_<profile>_<population>.json`，文档里引用文件名而不是口头数字。
`benchmarks/` 被 `.gitignore` 忽略（可反复生成，不入库）。
