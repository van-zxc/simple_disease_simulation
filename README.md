# simple_disease_simulation

基于 PyTorch + matplotlib 的**逐个体（agent-based）传染病模拟**：SEIR 框架上扩展出
无症状（A）与住院（H）状态，用疾病档案（JSON schema v1–v4）切换自然史、v3 社会响应
（检测 / 追踪 / 隔离 / 行为反馈 / 报告延迟）与 v4 个体异质性（超传播 / 依从性差异 /
年龄同配混合），并在 matplotlib 动画里实时观察空间分布、人群构成曲线、病毒参数演化
与再感染率。

## 背景

最早是为回答知乎问题[《对于致死率几乎百分百的病毒和细菌，那些在感染下幸存下来的人是因为基因很牛逼还是因为运气好？》](https://www.zhihu.com/question/1987928986068723299)写的小玩具，后来逐步扩展成现在这个可切换疾病档案的 agent-based 模拟器。

## 运行方式

依赖 Python >= 3.10 与 `torch`（有 CUDA 会自动用 GPU）、`matplotlib`、`numpy`：

```powershell
pip install torch matplotlib numpy
```

```powershell
python simple_disease_simulation.py                                        # 默认 covid19 档案（v4，机制全开）
python simple_disease_simulation.py --profile toy                          # 旧 v1 路径，快速出图
python simple_disease_simulation.py --profile covid19 --population 100000 --seed 7
python simple_disease_simulation.py --initial-seeds 200                   # 固定 200 个初始感染者（不随人口缩放）
python simple_disease_simulation.py --frames 3000                          # 等价 SIM_FRAMES=3000
```

窗口内点击 **LOCKDOWN** 按钮切换封控；左上标题实时显示计数、医疗压力、免疫/再感染率、
时钟与场所构成。大人口下建议用 `--async-render`（模型逐帧推进、每 2 帧重绘一次）或
`--render-every 4` 降绘制开销。

| 参数 | 环境变量 | 含义 | 默认 |
| --- | --- | --- | --- |
| `--profile NAME` | `DISEASE_PROFILE` | 疾病档案（可选值见 [diseases/README.md](diseases/README.md) §1） | `covid19` |
| `--population N` | `DISEASE_POPULATION` | 人口规模（自动按 √r 缩放面积、按 r 缩放场所数） | 配置内 `N_AGENTS` |
| `--seed N` | `DISEASE_SEED` | 随机种子 | 42 |
| `--frames N` | `SIM_FRAMES` | 运行帧数（1 帧 = 6 分钟，240 帧/天） | 配置内 `FRAMES` |
| `--initial-seeds N` | `DISEASE_INITIAL_SEEDS` | **初始感染人数的绝对值**（不随人口缩放，`0` = 零初始感染；对应配置项 `INITIAL_SEEDS_ABSOLUTE`） | 档案 `population.initial_seeds` × 人口/10000 |
| `--render-every K` | `SIM_RENDER_EVERY` | 每 K 帧重绘一次（模型仍逐帧推进） | `1` |
| `--async-render [K]` | — | 异步渲染快捷方式（等价 `--render-every K`，默认 2） | — |
| — | `SIM_RENDER_POINTS` | 散点抽样上限（限制大人口下的绘制成本） | `10000` |
| — | `CONTACT_MODE` | `auto` / `dense` / `grid` 接触对算法 | `auto` |

内置档案：`covid19`（默认）、`covid19_homogeneous`（同质对照）、`covid19_v2_noiv`
（关闭社会响应）、`toy`（v1 旧路径）。想改疾病参数做自己的实验，见
[diseases/README.md](diseases/README.md) §9。

## 自检

```powershell
python simple_disease_simulation.py --profile toy --frames 30       # 冒烟：能跑通、能出图
python tools/profile_check.py --profile covid19                     # 档案派生 / R0 / 代际时间自洽
python tools/ensemble.py --profile covid19 --population 10000 --frames 3000 --seeds 1-12
```

性能量级（covid19，400 帧，`--no-draw`）：单帧 `update` 中位 **8.2 / 13.2 / 19.6 ms**，
1% low **81.9 / 55.8 / 32.7 fps**（10k / 100k / 1M）；口径与全部实验见
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
