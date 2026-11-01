# simple_disease_simulation

基于 PyTorch 的**逐个体（agent-based）传染病模拟**：SEIR 框架上扩展出
无症状（A）与住院（H）状态，用疾病档案（JSON schema v1–v4）切换自然史、v3 社会响应
（检测 / 追踪 / 隔离 / 行为反馈 / 报告延迟）与 v4 个体异质性（超传播 / 依从性差异 /
年龄同配混合）。观察界面是**本地 Web 仪表盘**：地图 + 关键指标读数块 + 5 张自动缩放曲线，
浏览器打开即可，不需要装任何图形库。

## 背景

最早是为回答知乎问题[《对于致死率几乎百分百的病毒和细菌，那些在感染下幸存下来的人是因为基因很牛逼还是因为运气好？》](https://www.zhihu.com/question/1987928986068723299)写的小玩具，后来逐步扩展成现在这个可切换疾病档案的 agent-based 模拟器。

## 运行方式

依赖 Python >= 3.10、`torch`（有 CUDA 会自动用 GPU）、`numpy` 与 `aiohttp`（仪表盘的本地服务）：

```powershell
pip install torch numpy aiohttp
```

```powershell
python simple_disease_simulation.py                                        # 默认 covid19 档案（v4，机制全开）
python simple_disease_simulation.py --profile toy                          # 旧 v1 路径，快速看效果
python simple_disease_simulation.py --profile covid19 --population 100000 --seed 7
python simple_disease_simulation.py --initial-seeds 200                   # 固定 200 个初始感染者（不随人口缩放）
python simple_disease_simulation.py --frames 3000                          # 等价 SIM_FRAMES=3000
```

启动后终端会打印 `Dashboard: http://127.0.0.1:8765/` 并自动打开浏览器；**浏览器连上后模型才开始跑**
（不想等就加 `--no-wait`）。界面右下角 **LOCKDOWN 封控 / RESUME 解封** 按钮切换封控；
左上角显示 `Day / 时钟 / t / 封控`，下方三行是活跃感染拆分、人群池、医疗与免疫读数。

| 参数 | 环境变量 | 含义 | 默认 |
| --- | --- | --- | --- |
| `--profile NAME` | `DISEASE_PROFILE` | 疾病档案（可选值见 [diseases/README.md](diseases/README.md) §1） | `covid19` |
| `--population N` | `DISEASE_POPULATION` | 人口规模（自动按 √r 缩放面积、按 r 缩放场所数） | 配置内 `N_AGENTS` |
| `--seed N` | `DISEASE_SEED` | 随机种子 | 42 |
| `--frames N` | `SIM_FRAMES` | 运行帧数（1 帧 = 6 分钟，240 帧/天） | 配置内 `FRAMES` |
| `--initial-seeds N` | `DISEASE_INITIAL_SEEDS` | **初始感染人数的绝对值**（不随人口缩放，`0` = 零初始感染；对应配置项 `INITIAL_SEEDS_ABSOLUTE`） | 档案 `population.initial_seeds` × 人口/10000 |
| `--render-every K` | `SIM_RENDER_EVERY` | 每 K 帧推送一次画面（模型仍逐帧推进） | `1` |
| `--async-render [K]` | — | 异步推送快捷方式（等价 `--render-every K`，默认 2） | — |
| `--port N` | `DISEASE_VIEW_PORT` | 仪表盘端口（被占用会自动 +1） | `8765` |
| `--host ADDR` | `DISEASE_VIEW_HOST` | 监听地址（默认回环，不会触发防火墙弹窗） | `127.0.0.1` |
| `--no-browser` | — | 只起服务，不自动开浏览器 | — |
| `--no-wait` | — | 不等浏览器连接，立刻开跑 | — |
| `--max-fps F` | — | 推送帧率上限（0 = 不限） | `60` |
| `--headless` | `DISEASE_VIEW=headless` | 不启服务、只跑模型（脚本/工具用） | — |
| — | `SIM_RENDER_POINTS` | 地图散点抽样上限（限制大人口下的传输量） | `10000` |
| — | `CONTACT_MODE` | `auto` / `dense` / `grid` 接触对算法 | `auto` |

内置档案：`covid19`（默认）、`covid19_homogeneous`（同质对照）、`covid19_v2_noiv`
（关闭社会响应）、`toy`（v1 旧路径）。想改疾病参数做自己的实验，见
[diseases/README.md](diseases/README.md) §9。

## 自检

```powershell
python simple_disease_simulation.py --profile toy --frames 30       # 冒烟：能跑通
python tools/profile_check.py --profile covid19                     # 档案派生 / R0 / 代际时间自洽
python tools/ensemble.py --profile covid19 --population 10000 --frames 3000 --seeds 1-12
```

性能量级（covid19，400 帧，`--no-pack`，2026-11-01 本机单次测量）：单帧 `update` 中位
**6.7 / 10.9 / 14.6 ms**，1% low **63.5 / 42.0 / 41.3 fps**（10k / 100k / 1M；笔记本热
状态会让绝对值漂移 ±30%，只看同会话趋势）。视图负载 `pack` 约 **0.3 ms/帧、89 KB/帧**
（10k 与 1M 同量级，因为点云上限就是抽样后的 10000 点）。口径与全部实验见
[SCALING.md](SCALING.md) §6。

## 文档地图

| 想知道 | 看 |
| --- | --- |
| 每个机制怎么实现（状态机 / 自然史 / 医疗 / v3 社会响应 / v4 异质性） | [MECHANISMS.md](MECHANISMS.md) |
| 参数取值、量级、人口缩放、性能与吞吐实验 | [SCALING.md](SCALING.md) |
| 目录结构、模块职责、一帧数据流、兼容层原理 | [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) |
| 哪些地方不像真实世界（以及为什么不打算拟合） | [REALISM.md](REALISM.md) |
| 档案字段、派生公式、校验规则、新增一种疾病、改参数 | [diseases/README.md](diseases/README.md) |
| 测量 / 观测工具怎么用、结果怎么读 | [tools/README.md](tools/README.md) |
