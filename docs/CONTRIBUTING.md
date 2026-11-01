# 贡献与维护指南（人 + AI 通用）

本文件回答三件事：**代码怎么放、注释怎么写、改动怎么验**。它是
[ARCHITECTURE.md](ARCHITECTURE.md)（结构/数据流）与 [../MECHANISMS.md](../MECHANISMS.md)
（模型机制）之外的**过程约定**，新人和 AI 助手建议按 "§1 → §2 → §3" 顺序读一遍。

> 项目定位提醒：这是一个**机制探索型玩具模型**，不是预测工具。允许"不再逐位复现"，
> 但不允许悄悄改变语义：任何影响状态迁移、概率、曲线、年龄表、床位规则或可视化
> 契约的改动，都必须在提交说明与文档里讲清楚，并给出验证证据。

---

## 1. 最短验证路径

环境：Windows + PowerShell，Python 3.12，torch 2.5.1+cu124，matplotlib 3.9。
所有命令在仓库根目录执行；无头机器记得 `$env:MPLBACKEND='Agg'`、`$env:PYTHONIOENCODING='utf-8'`。

| 改动类型 | 必跑 | 耗时 |
| --- | --- | --- |
| 只改注释 / 文档 | 不需要跑模型；涉及引用编号时先 grep 确认章节存在 | 秒级 |
| 改档案数值（`diseases/*.json`） | `python tools/profile_check.py --profile <name>` + `python tests/run_all.py --quick` | ~40 s |
| 改包内代码（`disease_sim/`） | `python tools/headless_check.py --profile toy --frames 300` + `python tests/run_all.py --quick` | ~1 min |
| 改机制 / 状态机 / 传播 | `python tests/run_all.py --full`（7 阶段，含 harness 三档） | ~4 min |
| 改性能 | `tools/ab_iter.py` 配对 A/B（见 §5）+ `--full` | ~6 min |
| 改 GUI / 渲染 | `tools/render_prof.py` + 真实窗口点一次按钮 | ~1 min |

`tests/run_all.py --full` 的 7 个阶段：统计/结构 → profile 派生 → 接触对 parity →
v4 异质性 → harness toy / covid19 / covid19_v2_noiv。

---

## 2. 代码放在哪里

依赖方向**单向、无环**：

```
cli ─► simulation ─► {activity, contacts, disease, interventions, render, stats(内联在 simulation)} ─► {state, config}
population/build 在 state.py；profiles/ 完全独立，不 import 模型
入口 simple_disease_simulation.py ─► disease_sim/__init__.py（装配）└─► cli（动画）
tools/ 与 tests/ 只通过"加载入口"或"import 包"单向依赖，包内绝不 import tools/tests
```

放置原则：

| 你要加的东西 | 放哪 | 注意 |
| --- | --- | --- |
| 新的可调参数 | `config.py::_DEFAULTS`（命名前缀分组）+ 档案 schema 默认块 | 档案写入优先于默认值；`CONFIG` 字典身份不能变（见 `reset_config`） |
| 新的个体状态字段 | `state.py::build_state()` | **开关关闭时必须零成本、不消耗随机数** |
| 每帧逻辑 | `simulation.py::update()` 的对应阶段（A0/A/B/C/D/E/F） | 写清阶段与 RNG 顺序；不要跨阶段偷读上一帧的计数器 |
| 纯函数/算法 | `contacts.py` / `disease.py` / `activity.py` 等无状态模块 | 便于单测；不要悄悄持有全局态 |
| 新的可视化元素 | `render.py`（创建）+ `simulation.py` F 段（更新） | **15 artist 契约**不能破（harness 检查数量与 `.axes`） |
| 一次性分析脚本 | `tools/`，可长期保留则写成带 `--help` 的工具 | 临时脚本用完删除，别留仓库 |

---

## 3. 注释与 docstring 规范

全仓库统一，按"**读者 30 秒内能否知道为什么这样写**"来取舍。

### 3.1 语言与长度

- 正文用**中文**，标识符/命令用英文；行宽尽量 ≤ 100 字符。
- 模块 docstring ≤ 12 行：**负责什么 / 不负责什么 / 对外入口 / 必须保持的不变式**。
- 函数 docstring 1–6 行：目的 → 关键参数/返回 → 不变式。签名已说明的参数不要复述。
- 只有"非平凡"函数需要 docstring；三行 getter 写了反而是噪音。

### 3.2 行内注释只写 why

代码说 *what*，注释说 *why*。优先使用四个标记（便于 grep 与 AI 检索）：

| 标记 | 用于 | 例子 |
| --- | --- | --- |
| `# RNG 契约：` | 随机抽样次数/形状/顺序不能动的原因 | `# RNG 契约：still 段 randn 必须是 (N,2)，改分块会让玩具档计数漂移` |
| `# 性能：` | 为什么用主机门控/原地写/合并系数/不加 `.item()` | `# 性能：用上一帧 counts_prev 避免每帧 bincount+同步` |
| `# harness 契约：` | 被 `tools/headless_check.py` 读取的名字/形状/文本 | `# harness 契约：title_obj 必须含 "Inf:" 与 "t="` |
| `# 不变式：` | 恒等式、向量布局、计数器组成 | `# 不变式：sum(counts)==N；stat[0:8] 为 counts` |

### 3.3 要删的注释

- 复述代码的废话（`i += 1  # 自增`）；
- 编年史（"第二轮优化时我们改成了……"）——保留一句指向 `SCALING.md` 的指针即可；
- 被注释掉的旧代码（直接删，git 里有历史）；
- 空洞 TODO / FIXME（要么写清触发条件与负责人，要么删掉）。

### 3.4 魔数必须解释

`1.5`（传染力次幂）、`2.5`（医疗崩溃压力）、`0.05`、`0.12`（原地游走幅度）、
`65536`（诊断容器上限）、`30_000_000`（dense 接触对阈值）……每个都要写"是什么、
为什么是这个量级、改了会影响什么"。

### 3.5 引用文档章节

注释里写"见 `SCALING.md §6.2d`"之前，先 `Select-String` 确认该章节真的存在；
文档改版时要顺手更新引用（`tests/` 与 `tools/` 里同类引用一并 grep）。

### 3.6 给 AI 助手的导航顺序

1. `docs/ARCHITECTURE.md` §1–§3（目录、依赖、一帧数据流）；
2. 目标模块的**模块 docstring**；
3. `simulation.py::update()` 顶部的阶段表注释；
4. 具体段落的 `# RNG 契约 / # 性能 / # 不变式` 注释；
5. 改动后按 §1 表格跑验证。

---

## 4. 新增一个机制（checklist）

以 v4 的 `heterogeneity` 为范例，完整顺序：

1. **参数**：`config.py::_DEFAULTS` 加键（前缀分组、默认"关闭"）；
   `profiles/schema.py` 加 `*_DEFAULTS` 块与取值范围常量。
2. **校验**：`profiles/validate.py` 加 `_validate_<block>()`（未知键一律拒绝、区间检查、
   schema 版本门槛），并接进 `validate_profile()`。
3. **应用**：`profiles/apply.py` 加 `_apply_<block>()` 写 CONFIG，`PROFILE_INFO` 暴露子字典，
   `format_summary` 加一行人类可读摘要。注意：**即使 `"enabled": false`，块内数值也会写进
   CONFIG**，所以"关闭对照档"必须用中性值（本项目真实踩过这个坑）。
4. **状态**：`state.py::build_state()` 加字段；**所有随机抽取放在开关内**（否则 toy/v2 的
   轨迹会变，"关闭路径零行为变化"就失效了）。
5. **挂点**：`simulation.py` 对应阶段；写清与现有随机数的先后顺序；若影响传播概率，
   优先在 pair 级乘子处合成而不是新增 kernel（见 `# 性能：` 注释）。
6. **档案**：只在 `diseases/covid19.json` 默认开启；`toy`（v1）、`covid19_v2_noiv`（v3）
   保持关闭；需要对照时新建 `diseases/<name>.json`（**文件名 = profile_id**）。
7. **测试**：分布层（公式/量化区间）+ 行为层（对照档差异）+ 守恒/artist；
   需要时给 `tests/run_all.py` 加 stage（含超时常量）。
8. **文档**：`MECHANISMS.md` 新增机制节、`SCALING.md` 成本/缩放节、`README.md` 档案表与
   轮次表、`diseases/README.md` 字段与校验规则、必要时 `REALISM.md` 局限性。
9. **门禁**：`tests/run_all.py --full` 全绿；**不得修改 `tools/headless_check.py` 的断言**
   （它是用户侧契约）。

---

## 5. 性能改动规范

1. **先归因，再动手**：`tools/diag.py`（CPU 派发 vs GPU 算力）、`tools/phase_probe.py`
   （分段）、`tools/render_prof.py`（绘制）。单次 `tools/bench.py` 的绝对值跨时段不可比
   （笔记本热漂移可达 ±30%），**只能同会话对比**。
2. **A/B 协议**：`python tools/ab_iter.py --before <快照入口> --after simple_disease_simulation.py
   --profile covid19 --population 10000 --frames 200 --rounds 8 --json benchmarks/ab_xxx.json`。
   - 两进程交替（同进程会因 `sys.modules` 让两侧共用一份包）；
   - 丢弃前 2 轮预热，取**配对中位比**；**低于 3% 的差异视为噪声**。
3. 每步一个 JSON 落 `benchmarks/`，命名 `ab_<主题>_<profile>_<pop>.json`，
   文档里引用文件名而不是口头数字。
4. 不允许用"改变语义/随机数顺序"换性能；渲染、吞吐（多进程）、单世界延迟三种口径
   不要混着报（见 `SCALING.md` §6.4–§6.5）。
5. 优化后必须回到 §1 的"改机制"级别验证（`--full`），因为掩码写/索引化最容易悄悄改行为。

---

## 6. 文档维护映射（改代码时该动哪份文档）

| 你改了什么 | 必须更新 |
| --- | --- |
| 新机制 / 状态机 / 概率语义 | `MECHANISMS.md`（新增一节）+ `SCALING.md`（成本影响）+ `README.md` 机制表 |
| 新档案字段 / 校验规则 | `diseases/README.md`（字段表 + 示例 + 范围）+ `tests/test_profiles.py` |
| 新模块 / 依赖方向变化 | `docs/ARCHITECTURE.md` §1–§2 + `README.md` 结构树 |
| 新性能数字 | `SCALING.md` 对应 §6.x + `README.md` 轮次表 + `benchmarks/` JSON |
| 新工具 / 新测试 | `README.md` 结构树 + `tools/README.md`（工具速查表）+ `tests/README.md` / 工具自身 docstring |
| 验收口径变化 | `README.md` 验证段 + `MECHANISMS.md §11.1` + `SCALING.md` 口径说明 |
| 注释/文档规范本身 | 本文件 §3（改完同步检查是否与代码实践一致） |

---

## 7. 本仓库真实踩过的坑（改动前先看）

- **harness 会静默 SKIP**：`tools/headless_check.py` 在张量属性缺失时走 `isinstance`
  前置判断而跳过部分检查。汇总行现在会明确写 `SKIP n 项`（例如默认 300 帧有 7 项
  SKIP），**但 SKIP 不计入退出码**——改契约名字后不要只看 PASS，要读
  "失败断言 0 项 / 共 38 项 / SKIP n 项"，并确认 SKIP 数是基线值而不是变多。
- **`spec_from_file_location` 加载**：模块 `__package__` 为空、`sys.argv` 属于工具
  （不是模拟），所以入口里的 CLI 解析必须带"`sys.argv[0]` 是否本文件"的守卫。
- **不要用 PowerShell 重写 UTF-8 源文件**（`Get-Content | Set-Content` 会把中文变乱码、
  甚至破坏三引号）。用编辑工具改，或明确以 UTF-8 读写。
- **RNG 形状依赖**：`torch.randn((n,2))` 这类形状本身是契约；改分块/索引方式若改变了
  抽样形状，轨迹会变（允许，但要说明并重定统计区间）。
- **CUDA 上的 `torch.binomial`**：`count` 与 `prob` 必须同 dtype（本项目用 float32）。
- **`apply` 即使 `enabled: false` 也会写块内值**：对照档必须全中性值。
- **CUPTI 会放大 GPU 时间**（报出的 GPU 时间可 > wall），只能看相对占比与 kernel 计数；
  判断是否 CPU 受限看"CPU 入队时间 ≈ wall"。
- **`tools/bench.py` 的 draw 是全量 `canvas.draw()`**，比 harness 的
  `restore_region + 逐 artist blit` 和真实 GUI 的 `FuncAnimation(blit=True)` 都更贵。
- **历史快照不是门禁**：`tools/golden_check.py` 自第二轮优化后只用于诊断；不要拿它的
  失败当"代码坏了"的唯一判据。另外它的 `--dump` 现在**契约不完整时会拒绝写盘（rc=1）**，
  `compare` 也会把契约缺失判为分歧——旧快照契约过期属于预期。
- **一键回归是严格的**：`tests/run_all.py` 的每个阶段都要求"命中 PASS 标记且 `k==k`"，
  退出码 0 但阶段没跑会被判 FAIL；harness 阶段还要求"失败断言 0 项"，SKIP 单独计数。

---

## 8. 提交建议（本项目默认不自动 commit）

建议按"一个主题一个提交"拆分，便于回滚与 review：

1. 结构/入口/兼容层；
2. 每次性能优化一轮（附 `benchmarks/ab_*.json`）；
3. 每个机制（schema/validate/apply + state + simulation + tests + 档案）；
4. 工具与测试；
5. 文档。

提交前自查：

- [ ] `python tests/run_all.py --full` 全绿（7 阶段）；
- [ ] `python tools/headless_check.py --profile toy --frames 300` → 0 失败 / 共 38 项
  （SKIP 数应与基线一致：toy 7、covid19 6，见汇总行）；
- [ ] 新增/修改的参数在 `diseases/README.md` 有字段说明；
- [ ] 注释里的文档引用编号真实存在；
- [ ] 没有把临时脚本、日志、实验 JSON 混进代码目录。
