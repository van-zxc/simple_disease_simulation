"""运行期可变状态容器：``build_state()`` 一次性创建全部张量与标量，逐帧被 simulation 原地更新。

负责：人口/家庭/场所/身份、个体疾病状态与病毒参数、v2 计时、v3 社会响应、v4 异质性与
      接触混合、出行状态与渲染节流（分组见下方注释）。
不负责：逐帧逻辑（simulation/activity/interventions）、存档读档、绘图对象（render）。
对外入口：``build_state()`` → ``SimState``；入口模块 ``__getattr__`` 把未知名转发到唯一
      ``_SIM``，所以这里字段名/形状是调用方契约（工具按属性名读取）。
不变式：
  - 字段只在这里创建一次，运行期只原地改热字段，不换张量；长度均为 N（场所表除外）。
  - RNG 契约：结构用 numpy rng、疫情/逐帧行为用 torch 流；抽签顺序与形状被 golden 基线固化。
  - 开关（NH/IV/TESTING/TRACING/BEHAVIOR/HET/MIX）同时决定字段是否有意义与是否抽随机数；
    关闭时对应字段保持占位值，不能当"零填充的真实数据"读。
"""

import os

import numpy as np
import torch

from .config import CONFIG, SEED, device
from .profiles.schema import AGE_BAND_TO_GROUP


class SimState:
    """属性包：字段由 ``build_state`` 动态挂到实例上（无类级声明）。

    所有可变状态都挂在实例上，update(sim, frame) 的各阶段函数因此只传一个 sim 参数；
    入口模块与工具按属性名读取，不要在别处往实例上补字段。
    """

    def __repr__(self) -> str:  # pragma: no cover - 仅调试用，统计已挂属性个数
        return f"SimState({len(self.__dict__)} attributes)"


def _clip_world(pos: np.ndarray, margin: float, w: float, h: float) -> np.ndarray:
    """把 (N,2) 坐标按轴收回世界内：x∈[margin, w-margin]、y∈[margin, h-margin]。

    【2026-11 审计修复】原先 x/y 都按 ``W`` 裁剪；``CONFIG["WIDTH"]`` 与 ``CONFIG["HEIGHT"]``
    是两个独立键，非方形世界下 y 会越界（默认两者均为 100，所以此前不可见）。W==H 时
    本函数与原 ``np.clip(..., W)`` 逐元素等价，不改变既有轨迹。
    """
    return np.column_stack((
        np.clip(pos[:, 0], margin, w - margin),
        np.clip(pos[:, 1], margin, h - margin),
    ))


# ==================== 城市结构：城区 / 邻里 / 场所选址 ====================
# 这些函数只决定"谁在哪"，不决定"多少人"：人口/密度由 _apply_population_scale 负责。
# 现实城市的三条结构规则在这里落地（详见 REALISM.md §3）：
#   1) 多中心且规模重尾：区数随人口亚线性增长，各区的就业规模服从 Zipf；
#   2) 职住分离：住在哪决定通勤去哪（距离衰减的引力模型），不是全国均匀抽签；
#   3) 规模不变性：场馆内的落点散布 σ_v 由"占用人数 / 场馆密度"反推，人数 ∝ 面积，
#      所以换个布局不会悄悄改掉单帧接触数（这是 R0 标定的前提）。
# 所有抽样都用 numpy rng，且顺序固定（同 seed ⇒ 同布局）。


def _stratified_centers(rng: np.random.Generator, w: float, h: float, count: int,
                        jitter: float, margin: float) -> np.ndarray:
    """分层抖动网格：count 个中心近似铺满世界，各自在网格单元内随机抖动。

    为什么不用纯均匀随机：会成团并留下大片空洞；为什么不用规则栅格：看起来太"人造"。
    城区中心用它（城市在地域上分散），公园也用它（公园填城区之间的开阔地）。
    """
    side = max(1, int(np.ceil(np.sqrt(count))))
    cell_w = (w - 2.0 * margin) / side
    cell_h = (h - 2.0 * margin) / side
    gx, gy = np.meshgrid(np.arange(side, dtype=float), np.arange(side, dtype=float))
    cells = np.column_stack(((gx.ravel() + 0.5) * cell_w + margin,
                             (gy.ravel() + 0.5) * cell_h + margin))
    pick = rng.permutation(cells.shape[0])[:count]
    offsets = rng.normal(0.0, 1.0, (count, 2)) * np.array([cell_w, cell_h]) * float(jitter)
    return _clip_world(cells[pick] + offsets, margin, w, h)


def _zipf_weights(rng: np.random.Generator, count: int, alpha: float) -> np.ndarray:
    """规模重尾权重 w_k ∝ 1/rank^α，再随机分配到各中心（规模与空间位置无关）。"""
    ranks = np.arange(1, count + 1, dtype=float)
    weights = 1.0 / ranks ** max(0.0, float(alpha))
    weights /= weights.sum()
    return rng.permutation(weights)


def _allocate_counts(total: int, weights: np.ndarray) -> np.ndarray:
    """把 total 个场所按权重分到各区（最大余数法；总数严格等于 total）。"""
    raw = weights * total
    counts = np.floor(raw).astype(np.int64)
    order = np.argsort(-(raw - counts), kind="stable")
    i = 0
    while counts.sum() < total and i < 4 * total + counts.size:
        counts[order[i % counts.size]] += 1
        i += 1
    while counts.sum() > total:
        j = int(np.argmax(counts))
        if counts[j] <= 0:
            break
        counts[j] -= 1
    return counts


def _build_districts(rng: np.random.Generator, sim) -> tuple[np.ndarray, np.ndarray,
                                                             np.ndarray, np.ndarray]:
    """构造城区：分层抖动网格的多中心 + 就业 Zipf 权重 + 住宅权重 + 容纳半径。"""
    margin = sim.W * 0.1
    count = int(CONFIG["N_DISTRICTS"])
    pos = _stratified_centers(rng, sim.W, sim.H, count,
                              CONFIG["DISTRICT_GRID_JITTER"], margin)
    jobs_w = _zipf_weights(rng, count, CONFIG["DISTRICT_EMPLOYMENT_ALPHA"])
    res_w = (CONFIG["DISTRICT_RESIDENTIAL_BASE"]
             + CONFIG["DISTRICT_RESIDENTIAL_SPREAD"] * jobs_w)
    res_w = res_w / res_w.sum()
    # 容纳半径：住宅权重 × 基准密度下需要的面积（√(v·A/π)），决定邻里与场所的散开范围
    radius = np.sqrt(res_w * sim.W * sim.H / np.pi)
    sim.district_pos = torch.tensor(pos, dtype=torch.float32, device=device)
    sim.district_jobs_w = torch.tensor(jobs_w, dtype=torch.float32, device=device)
    sim.district_res_w = torch.tensor(res_w, dtype=torch.float32, device=device)
    sim.district_radius = torch.tensor(radius, dtype=torch.float32, device=device)
    return pos, jobs_w, res_w, radius


def _build_neighborhoods(rng: np.random.Generator, sim, district_pos: np.ndarray,
                         district_res_w: np.ndarray,
                         district_radius: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """构造居住邻里：在世界内近似均匀铺开，每个邻里归属最近的城区。

    为什么不在城区内部聚成小团：人均居住面积是"密度→接触率→R0"标定的前提，
    把邻里全部塞进城区圆盘会让居住密度成倍上升（实测 10k 下 home 段接触数从 16.7 涨到
    31.6）。所以邻里仍是全城铺开（与旧布局的 12 个居住簇同口径），只是加上"归属哪个城区"
    的信息，供通勤引力使用；城区规模差异体现在"住哪儿去哪个中心上班"。
    """
    count = int(CONFIG["N_NEIGHBORHOODS"])
    margin = sim.W * 0.1
    center = _clip_world(np.column_stack((
        rng.uniform(margin, sim.W - margin, count),
        rng.uniform(margin, sim.H - margin, count))), 2.0, sim.W, sim.H)
    nb_district = np.argmin(
        np.linalg.norm(center[:, None, :] - district_pos[None, :, :], axis=2), axis=1)
    sim.neighborhood_center = torch.tensor(center, dtype=torch.float32, device=device)
    sim.nb_district = torch.tensor(nb_district, dtype=torch.long, device=device)
    return center, nb_district


def _venues_around_districts(rng: np.random.Generator, sim, count: int,
                             weights: np.ndarray, district_pos: np.ndarray,
                             district_radius: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """按权重把 count 个场所分到各区，区内用黄金角螺旋（蓝噪声）铺开。

    为什么不是"高斯撒在区中心附近"：场馆一多（大人口档位）就会全堆到区中心，
    区内的行人云互相重叠，接触数随人口单调上升（实测 1M 的 work 段比 10k 高 46%）。
    螺旋让"单位面积内的场馆数"与基准一致，接触率才对人口规模不敏感。
    """
    counts = _allocate_counts(count, weights)
    district_of = np.repeat(np.arange(weights.size), counts)
    pos = np.empty((count, 2), dtype=float)
    _golden = np.pi * (3.0 - np.sqrt(5.0))   # ≈2.39996 rad，黄金角
    _start = 0
    for _k, _n in enumerate(counts):
        if _n == 0:
            continue
        _idx = np.arange(_n, dtype=float)
        # 区内半径按 √(i/n) 递增：等价于圆盘内面积均匀（Vogel 螺旋），最外圈留 10% 余量
        _r = (district_radius[_k] * 0.9 * np.sqrt((_idx + 0.5) / _n)
              if _n > 1 else np.zeros(1))
        _theta = _idx * _golden + rng.uniform(0.0, 2.0 * np.pi)
        pos[_start:_start + _n, 0] = district_pos[_k, 0] + np.cos(_theta) * _r
        pos[_start:_start + _n, 1] = district_pos[_k, 1] + np.sin(_theta) * _r
        pos[_start:_start + _n] += rng.normal(
            0.0, 0.05 * float(district_radius[_k]), (_n, 2))
        _start += _n
    return _clip_world(pos, 1.5, sim.W, sim.H), district_of


def _venue_sigma(occupancy: np.ndarray, density: float) -> np.ndarray:
    """场馆内落点散布 σ_v = √(n_v / (2π·ρ))：人数 ∝ 面积 ⇒ 单帧接触数不随人口放大而变。"""
    return np.sqrt(np.maximum(occupancy, 1.0) / (2.0 * np.pi * max(float(density), 1e-9)))


def build_state() -> SimState:
    """构造整局仿真的初始状态：人口/家庭/场所、初始免疫与种子、以及各开关层的占位字段。

    返回：已完整初始化的 ``SimState``（同一进程只构造一次，之后逐帧原地更新）。
    不变式：返回时 counts_prev/_n_* 反映"第 0 帧之前"的状态；所有 torch 张量都在 device 上
    且长度 N；NH/v3/v4 关闭时对应字段是占位值（0/False/None），逻辑分支由开关位决定。
    """
    sim = SimState()
    # ==================== 初始化：人口、家庭结构与场所 ====================
    # ---- 人口与世界尺寸：时钟尺度 ----
    sim.W, sim.H = CONFIG["WIDTH"], CONFIG["HEIGHT"]
    sim.N = CONFIG["N_AGENTS"]
    sim.MIN_PER_FRAME = 1440.0 / CONFIG["DAY_FRAMES"]  # 1440=一天分钟数；仅日程用，不做物理换算

    # 结构初始化用 numpy 随机流（家庭/场所），疫情与逐帧行为用 torch 随机流。
    # 结构只由 SEED 决定；torch 流的抽样顺序见下方 v4/初始免疫/日程抖动各段的 RNG 契约注释。
    rng = np.random.default_rng(int(SEED))

    # ---- 城市骨架：城区（多中心）+ 容纳半径 ----
    # margin=10% 世界边长：让中心离边界留出余量，后面 venue 抖动/clamp 时少被推回。
    margin = sim.W * 0.1
    district_pos, district_jobs_w, district_res_w, district_radius = _build_districts(rng, sim)

    # ---- 家庭规模与人口结构（向量化，支持大规模人口）----
    # 不变式：sum(sizes)==sim.N 且 household_id 长度 == sim.N（所有按人张量都依赖这个长度）。
    hh_weights = np.asarray(CONFIG["HOUSEHOLD_SIZE_WEIGHTS"], dtype=float)
    hh_weights = hh_weights / hh_weights.sum()
    # +1：把权重下标 0..k-1 映射成户规模 1..k
    drawn_sizes = rng.choice(len(hh_weights), size=sim.N, p=hh_weights) + 1
    # 按总人口截断：保留能完整容纳的家庭，最后用余数补齐
    _cum_sizes = np.cumsum(drawn_sizes)
    _n_full = int(np.searchsorted(_cum_sizes, sim.N, side="right"))
    sizes = drawn_sizes[:_n_full].astype(np.int64)
    if sizes.sum() < sim.N:
        sizes = np.append(sizes, sim.N - sizes.sum())
    # 按户规模降序排列，使同规模家庭连续，便于批量生成家庭构成
    _order = np.argsort(-sizes, kind="stable")
    sizes = sizes[_order]
    n_households = int(sizes.size)
    household_id_np = np.repeat(np.arange(n_households, dtype=np.int64), sizes)

    # 家庭构成模式：kinds 0=儿童，1=成人，2=老人；概率经校准，总体约 18% 儿童 / 16% 老人。
    # 表按户规模分档（>5 人的户按 5 人档处理，见下方 min(_s,5)），每行 (概率, 成员 kinds 模式)。
    # 魔数：18%/16% 是人口结构的校准目标（儿童/老人占比），改表会改变年龄分布，
    # 进而改变 MECHANISMS.md §4 年龄别严重性的加权结果（ICU/死亡风险）。
    _HOUSEHOLD_PATTERNS = {
        1: [(0.40, [1]), (0.60, [2])],
        2: [(0.30, [1, 1]), (0.30, [1, 0]), (0.30, [1, 2]), (0.10, [2, 2])],
        3: [(0.35, [1, 1, 0]), (0.10, [1, 0, 0]), (0.31, [1, 1, 1]),
            (0.15, [1, 1, 2]), (0.05, [1, 2, 2]), (0.04, [2, 2, 2])],
        4: [(0.30, [1, 1, 0, 0]), (0.05, [1, 0, 0, 0]), (0.10, [1, 1, 1, 0]),
            (0.10, [1, 1, 2, 2]), (0.08, [1, 1, 1, 2]), (0.32, [1, 1, 1, 1]),
            (0.05, [2, 2, 2, 2])],
        5: [(0.30, [1, 1, 0, 0, 0]), (0.10, [1, 1, 1, 0, 0]),
            (0.05, [1, 0, 0, 0, 0]), (0.10, [1, 1, 1, 2, 2]),
            (0.05, [1, 1, 2, 2, 2]), (0.10, [1, 1, 1, 1, 0]),
            (0.30, [1, 1, 1, 1, 1])],
    }

    kinds = np.zeros(sim.N, dtype=np.int8)
    _boundaries = np.cumsum(sizes)
    _neg_sizes = -sizes
    for _s in np.unique(sizes)[::-1]:
        _s = int(_s)
        _h0 = int(np.searchsorted(_neg_sizes, -_s))
        _h1 = int(np.searchsorted(_neg_sizes, -_s, side="right"))
        _cnt = _h1 - _h0
        _m0 = int(_boundaries[_h0 - 1]) if _h0 > 0 else 0
        _table = _HOUSEHOLD_PATTERNS[min(_s, 5)]
        _probs = np.array([p for p, _ in _table], dtype=float)
        _probs /= _probs.sum()
        _comp = rng.choice(len(_table), size=_cnt, p=_probs)
        _pat = np.zeros((_cnt, _s), dtype=np.int8)
        for _k, (_, _pattern) in enumerate(_table):
            _sel = _comp == _k
            if _sel.any():
                _pat[_sel, :len(_pattern)] = _pattern
        kinds[_m0:_m0 + _cnt * _s] = _pat.reshape(-1)

    # ---- 身份：年龄 / 就业 / 就学 ----
    # 年龄由家庭 kinds 决定区间：儿童 0-17、成人 18-64、老人 65-89（18/65/90 是人口分档边界，
    # 与 AGE_BANDS/age_band 的分段一致）。
    age_np = np.empty(sim.N, dtype=np.int64)
    _age_child = kinds == 0
    _age_adult = kinds == 1
    _age_elder = kinds == 2
    age_np[_age_child] = rng.integers(0, 18, int(_age_child.sum()))
    age_np[_age_adult] = rng.integers(18, 65, int(_age_adult.sum()))
    age_np[_age_elder] = rng.integers(65, 90, int(_age_elder.sum()))

    _adult_np = (age_np >= 18) & (age_np < 65)
    _child_np = age_np < 18
    # 抽样顺序固定：先就业、后就学（两次 rng.random(N)；换顺序会改同 seed 的人口结构）
    employed_np = _adult_np & (rng.random(sim.N) < CONFIG["ADULT_EMPLOYMENT"])
    enrolled_np = (_child_np & (age_np >= CONFIG["SCHOOL_MIN_AGE"]) &
                   (rng.random(sim.N) < CONFIG["CHILD_SCHOOL_ENROLL"]))

    # ---- 家庭位置：先按人口权重选邻里，再在邻里中心附近撒家庭锚点 ----
    # 聚集的意义：同户位置相近、户与户成簇，接触结构才有"社区"而不是均匀撒点。
    # 邻里是"居住聚落"（人口 ≈ NEIGHBORHOOD_POP）；HOME_CLUSTER_STD 是物理常量（世界单位），
    # 不随人口缩放：1M 人口下邻里仍是"小区"大小，只是一块地方有更多邻里，而不是糊成一团。
    nb_center, nb_district = _build_neighborhoods(
        rng, sim, district_pos, district_res_w, district_radius)
    cluster_of_house = rng.integers(0, nb_center.shape[0], n_households)
    home_anchor_np = nb_center[cluster_of_house] + rng.normal(
        0.0, CONFIG["HOME_CLUSTER_STD"], (n_households, 2))
    # 锚点先收回世界内（留 2.0 边距），避免与随后只留 1.5 边距的 home_pos 产生大偏差
    home_anchor_np = _clip_world(home_anchor_np, 2.0, sim.W, sim.H)
    home_np = home_anchor_np[household_id_np] + rng.normal(
        0.0, CONFIG["HOME_JITTER_STD"], (sim.N, 2))
    home_np = _clip_world(home_np, 1.5, sim.W, sim.H)
    sim.home_pos = torch.tensor(home_np, dtype=torch.float32, device=device)
    # home_anchor 按人展开成 (N,2)：同户 N 人共享同一个锚点（出发/返家的聚散基点）
    sim.home_anchor = torch.tensor(home_anchor_np[household_id_np], dtype=torch.float32, device=device)
    sim.pos = sim.home_pos.clone()

    # 调用方契约：household_id/agent_age/employed/student/home_pos/home_anchor 被
    #   入口与分析工具读取；改类型/形状/语义会让历史对照失配。
    sim.household_id = torch.tensor(household_id_np, dtype=torch.long, device=device)
    sim.agent_age = torch.tensor(age_np, dtype=torch.long, device=device)
    # 粗年龄组（0=儿童 1=成人 2=老人）：v4 混合矩阵按它聚合；MIX_ON 时再映射到 4 组 age_group
    age_group = torch.tensor(
        np.where(age_np < 18, 0, np.where(age_np >= 65, 2, 1)),
        dtype=torch.long, device=device)
    sim.employed = torch.tensor(employed_np, dtype=torch.bool, device=device)
    sim.student = torch.tensor(enrolled_np, dtype=torch.bool, device=device)

    # ---- 场所：工作/学校按区配置，商店贴邻里，公园填区际空隙 ----
    # 布局规则（现实城市）：工作场所集中在各城区的就业中心附近；学校跟着学生住址所在区；
    # 商店贴着居住区（步行可达）；公园是开阔地，铺在城区之间的空档里。
    n_work = CONFIG["N_WORKPLACES"]
    n_school = CONFIG["N_SCHOOLS"]
    sim.n_shop = CONFIG["N_SHOPS"]
    sim.n_park = CONFIG["N_PARKS"]
    sim.work_venue_np, work_district = _venues_around_districts(
        rng, sim, n_work, district_jobs_w, district_pos, district_radius)
    sim.school_venue_np, school_district = _venues_around_districts(
        rng, sim, n_school, district_res_w, district_pos, district_radius)
    # 商店挂到随机邻里，落在邻里中心 ±1.5×HOME_CLUSTER_STD（社区商业，不是全城均匀撒点）
    shop_nb = rng.integers(0, nb_center.shape[0], sim.n_shop)
    sim.shop_venue_np = _clip_world(
        nb_center[shop_nb] + rng.normal(0.0, 1.0, (sim.n_shop, 2))
        * (1.5 * CONFIG["HOME_CLUSTER_STD"]), 1.5, sim.W, sim.H)
    sim.park_venue_np = _stratified_centers(
        rng, sim.W, sim.H, sim.n_park, CONFIG["DISTRICT_GRID_JITTER"], margin)
    sim.park_venue_np = _clip_world(sim.park_venue_np, 1.5, sim.W, sim.H)
    sim.work_venue = torch.tensor(sim.work_venue_np, dtype=torch.float32, device=device)
    sim.school_venue = torch.tensor(sim.school_venue_np, dtype=torch.float32, device=device)
    sim.shop_venue = torch.tensor(sim.shop_venue_np, dtype=torch.float32, device=device)
    sim.park_venue = torch.tensor(sim.park_venue_np, dtype=torch.float32, device=device)

    # ---- 本区就近的商业/绿地（外出"就近"用）----
    # 每个邻里记住最近的商店/公园，逐人索引到自家邻里：不用 O(N×场所) 的最近邻搜索，
    # 大人口下也只有 (邻里数 × 场所数) 的距离矩阵。
    _shop_near = np.argmin(np.linalg.norm(
        nb_center[:, None, :] - sim.shop_venue_np[None, :, :], axis=2), axis=1)
    _park_near = np.argmin(np.linalg.norm(
        nb_center[:, None, :] - sim.park_venue_np[None, :, :], axis=2), axis=1)
    _nb_of_agent = cluster_of_house[household_id_np]
    sim.local_shop_idx = torch.tensor(
        _shop_near[_nb_of_agent], dtype=torch.long, device=device)
    sim.local_park_idx = torch.tensor(
        _park_near[_nb_of_agent], dtype=torch.long, device=device)

    # ---- 职住分配：通勤距离衰减的引力模型 + 区内均衡 ----
    # 引力模型：P(工作区 k | 居住区 h) ∝ w_k · exp(-d(h,k) / (COMMUTE_DECAY·W))。
    # 现实依据：城市内出行服从引力模型（距离衰减），且就业规模重尾（Zipf）。
    emp_idx = np.where(employed_np)[0]
    stu_idx = np.where(enrolled_np)[0]
    home_district_of_agent = nb_district[cluster_of_house[household_id_np]]
    commute_len = max(float(CONFIG["COMMUTE_DECAY"]) * sim.W, 1.0)
    district_dist = np.linalg.norm(district_pos[:, None, :] - district_pos[None, :, :], axis=2)

    # 不变式：无对应场所者索引为 -1（activity/contacts 用该哨兵判断"今天不去"）。
    work_idx_np = np.full(sim.N, -1, dtype=np.int64)
    school_idx_np = np.full(sim.N, -1, dtype=np.int64)

    # 工作区抽样：先按引力概率建 (M,M) 行 cdf，再对每个就业者用一次 U(0,1) 做逆变换
    _log_p = np.log(np.maximum(district_jobs_w, 1e-12))[None, :] - district_dist / commute_len
    _has_work = np.bincount(work_district, minlength=district_jobs_w.size) > 0
    _p = np.exp(_log_p - _log_p.max(axis=1, keepdims=True))
    _p[:, ~_has_work] = 0.0                      # 没有工作场所的区不可选
    _p /= _p.sum(axis=1, keepdims=True)
    _cdf = np.cumsum(_p, axis=1)
    _u = rng.random(emp_idx.size)
    work_district_choice = (_cdf[home_district_of_agent[emp_idx]] < _u[:, None]).sum(axis=1)
    # 区内按"单位规模"分配：单位岗位规模服从对数正态（大多数小单位 + 少量大单位），
    # 员工按容量成比例落位。容量只改变"谁在哪家单位"，不改变场馆内的人均密度
    # （σ_v 由实际占用人数反推），所以重尾不会破坏接触率标定。
    _job_cap = rng.lognormal(mean=0.0, sigma=float(CONFIG["WORKPLACE_SIZE_SIGMA"]), size=n_work)
    for _k in range(district_jobs_w.size):
        _sel = emp_idx[work_district_choice == _k]
        if _sel.size == 0:
            continue
        _venues_k = np.where(work_district == _k)[0]
        _cap = _job_cap[_venues_k]
        _cap_cdf = np.cumsum(_cap / _cap.sum())
        _u_local = rng.random(_sel.size)
        work_idx_np[_sel] = _venues_k[
            np.minimum(np.searchsorted(_cap_cdf, _u_local, side="right"),
                       _venues_k.size - 1)]

    # 上学：本区学校优先；本区没有学校时按距离找最近的有学校的区（确定性，不抽签）
    _has_school = np.bincount(school_district, minlength=district_jobs_w.size) > 0
    _school_of_district = np.argmin(np.where(_has_school[None, :], district_dist, np.inf), axis=1)
    school_district_choice = _school_of_district[home_district_of_agent[stu_idx]]
    for _k in range(district_jobs_w.size):
        _sel = stu_idx[school_district_choice == _k]
        if _sel.size == 0:
            continue
        _venues_k = np.where(school_district == _k)[0]
        school_idx_np[_sel] = _venues_k[rng.permutation(_sel.size) % _venues_k.size]

    # 场馆占用人数 → 停留散布 σ_v（人数 ∝ 面积 ⇒ 单帧接触数对人口规模近似不变）
    work_occ = np.bincount(work_idx_np[emp_idx], minlength=n_work).astype(float)
    school_occ = np.bincount(school_idx_np[stu_idx], minlength=n_school).astype(float)
    work_sigma = _venue_sigma(work_occ, CONFIG["VENUE_DENSITY"]["work"])
    school_sigma = _venue_sigma(school_occ, CONFIG["VENUE_DENSITY"]["school"])
    work_pos_np = np.zeros((sim.N, 2))
    # 个人在场所内的落点按 σ_v 散布：既不是挤在场馆中心点上，也不随世界放大而散掉
    # （旧实现按 VENUE_SPREAD×√area 散布，大人口下会散到接触半径之外，接触率被悄悄改掉）。
    # 非就业/非在册者的 work_pos/school_pos 只是被 clip 到 1.5 的占位坐标（-1 索引永不使用）。
    work_pos_np[emp_idx] = sim.work_venue_np[work_idx_np[emp_idx]] + rng.normal(
        0.0, 1.0, (emp_idx.size, 2)) * work_sigma[work_idx_np[emp_idx], None]
    school_pos_np = np.zeros((sim.N, 2))
    school_pos_np[stu_idx] = sim.school_venue_np[school_idx_np[stu_idx]] + rng.normal(
        0.0, 1.0, (stu_idx.size, 2)) * school_sigma[school_idx_np[stu_idx], None]
    work_pos_np = _clip_world(work_pos_np, 1.5, sim.W, sim.H)
    school_pos_np = _clip_world(school_pos_np, 1.5, sim.W, sim.H)
    # 调用方契约：work_pos/school_pos/work_venue_idx/school_venue_idx/shop_venue/park_venue
    #   在 CONTRACT_TENSORS 里；-1 哨兵与固定坐标范围被基线指纹固化。
    sim.work_pos = torch.tensor(work_pos_np, dtype=torch.float32, device=device)
    sim.school_pos = torch.tensor(school_pos_np, dtype=torch.float32, device=device)
    sim.work_venue_idx = torch.tensor(work_idx_np, dtype=torch.long, device=device)
    sim.school_venue_idx = torch.tensor(school_idx_np, dtype=torch.long, device=device)

    # 逐人停留半径（供 activity 的停留段按 place 取用；家仍固定 0.6）
    work_spread_np = np.zeros(sim.N, dtype=np.float32)
    work_spread_np[emp_idx] = work_sigma[work_idx_np[emp_idx]]
    school_spread_np = np.zeros(sim.N, dtype=np.float32)
    school_spread_np[stu_idx] = school_sigma[school_idx_np[stu_idx]]
    sim.spread_home = torch.full((sim.N,), 0.6, device=device)
    sim.spread_work = torch.tensor(work_spread_np, device=device)
    sim.spread_school = torch.tensor(school_spread_np, device=device)

    # 场馆占用规模（渲染点尺寸用；商店/公园按"同时在场期望人数"估：外出概率 × 时长占比）
    _errand_share = (CONFIG["ERRAND_PROB_WEEKDAY"]
                     * CONFIG["ERRAND_DURATION_MIN"] / CONFIG["DAY_FRAMES"])
    _shop_occ = max(1.0, sim.N * _errand_share * 0.65 / max(1, sim.n_shop))
    _park_occ = max(1.0, sim.N * _errand_share * 0.35 / max(1, sim.n_park))
    sim.venue_occupancy = {
        "work": work_occ,
        "school": school_occ,
        "shop": np.full(sim.n_shop, _shop_occ),
        "park": np.full(sim.n_park, _park_occ),
    }

    # ---- 个体状态：status 0..7（语义见 MECHANISMS.md §1 状态码表）----
    # 初始全 0=S；1(初始免疫)/2(E) 稍后由初始免疫与种子写入。统计按 status 取值，勿改状态码。
    sim.status = torch.full((sim.N,), 0, dtype=torch.long, device=device)

    # ---- 病毒参数：每人的 beta(传染力)/gamma(康复率)/mu(死亡率)/sigma(变异率)/gen(代数) ----
    # 初始全 0，只有被感染/种子个体才写入有效值；E/I/A 个体参与传播时才读。
    sim.virus_beta = torch.zeros(sim.N, device=device)
    sim.virus_gamma = torch.zeros(sim.N, device=device)
    sim.virus_mu = torch.zeros(sim.N, device=device)
    sim.virus_sigma = torch.zeros(sim.N, device=device)
    sim.virus_gen = torch.zeros(sim.N, dtype=torch.long, device=device)

    # ---- 潜伏期计时：timer 每帧对 status==2 加 1，与 max_incubation 比较判断转症 ----
    # max_incubation 是每个人的潜伏期上限（感染时抽一次，之后只读）。
    sim.timer = torch.zeros(sim.N, device=device)
    sim.max_incubation = torch.zeros(sim.N, device=device)

    # 是否曾被感染（再感染口径：免疫消退后再次感染仍算再感染）
    sim.ever_infected = torch.zeros(sim.N, dtype=torch.bool, device=device)

    # ---- 自然史/医疗机制（schema v2；enable=False 时全部为占位符，不影响旧路径）----
    sim.FPD = float(CONFIG["DAY_FRAMES"])
    sim.NH_ENABLED = bool(CONFIG["NATURAL_HISTORY_ENABLED"])
    # 年龄分档边界 [10,20,...,80] → age_band 0..8（9 档，每档 10 岁，对应 AGE_BANDS）；
    # 下方的 IFR/住院率/易感性相对表都按 age_band 索引。
    _age_bounds = torch.tensor([10, 20, 30, 40, 50, 60, 70, 80],
                               dtype=torch.long, device=device)
    sim.age_band = torch.bucketize(sim.agent_age, _age_bounds)  # 0..8，对应 AGE_BANDS

    # ---- v2 异质性乘子：年龄别相对风险（按实际人口归一化到均值 1，只改个体分布）----
    # 归一化的意义：住院率/易感性/IFR 的总体期望仍等于档案值，年龄只决定"谁的风险高"。
    if sim.NH_ENABLED:
        # 年龄别相对表在初始化时按实际人口年龄分布归一化（均值 1）：
        # 住院率/易感性/IFR 的总体均值仍等于档案值，只改变个体差异
        _age_ifr_rel = torch.tensor([float(v) for v in CONFIG["AGE_IFR_RELATIVE"]],
                                    dtype=torch.float32, device=device)
        sim.severity_mult = _age_ifr_rel[sim.age_band]
        # clamp(min=1e-9) 只防"人口全落在 0 权重档"时除零，正常档案下不触发
        sim.severity_mult = sim.severity_mult / sim.severity_mult.mean().clamp(min=1e-9)
        _age_hosp_rel = torch.tensor([float(v) for v in CONFIG["AGE_HOSPITALIZATION_RELATIVE"]],
                                     dtype=torch.float32, device=device)
        sim.hosp_mult = _age_hosp_rel[sim.age_band]
        sim.hosp_mult = sim.hosp_mult / sim.hosp_mult.mean().clamp(min=1e-9)
        _age_suscept_rel = torch.tensor([float(v) for v in CONFIG["AGE_SUSCEPTIBILITY"]],
                                        dtype=torch.float32, device=device)
        sim.suscept_mult = _age_suscept_rel[sim.age_band]
        sim.suscept_mult = sim.suscept_mult / sim.suscept_mult.mean().clamp(min=1e-9)
    else:
        # 关闭 NH 时全置 1：E 段仍按乘子形式乘上去，但取值恒为 1，旧路径数值不变；
        # severity_mult/hosp_mult 只被 NH 分支读取，这里纯粹占位。
        sim.severity_mult = torch.ones(sim.N, device=device)
        sim.hosp_mult = torch.ones(sim.N, device=device)
        sim.suscept_mult = torch.ones(sim.N, device=device)

    # ---- v2 计时与标记（仅 NH_ENABLED 时由 simulation 的 D 段维护）----
    # 哨兵语义（F 段主机门控按这三段读，别改成 0 基计数）：-1=未启动、>0=倒计时中、==0=本帧到期；
    #   iso_timer 在到期帧置 isolated=True；新感染者会被统一复位回 -1（见 simulation E 段）。
    # 调用方契约：inf_elapsed/symp_elapsed/a_timer/iso_timer/isolated/admission_attempted/
    #   untreated/beds_total 都在 CONTRACT_TENSORS/CONTRACT_SCALARS 里，初值被 golden 基线固化。
    sim.inf_elapsed = torch.zeros(sim.N, device=device)        # 感染后帧数（E/I/A）
    sim.symp_elapsed = torch.zeros(sim.N, device=device)       # 发病后帧数
    sim.a_timer = torch.zeros(sim.N, device=device)            # 无症状剩余帧数
    sim.iso_timer = torch.full((sim.N,), -1.0, device=device)  # 隔离倒计时；-1 表示不隔离
    sim.isolated = torch.zeros(sim.N, dtype=torch.bool, device=device)
    # 住院抽签只尝试一次（置真后不再抽）：床位是否可用不改变随机数消耗，只改变结局
    sim.admission_attempted = torch.zeros(sim.N, dtype=torch.bool, device=device)
    sim.untreated = torch.zeros(sim.N, dtype=torch.bool, device=device)  # 需住院但无床位
    # 床位总数：BEDS_PER_1000 是"每千人床位数"（covid19 档案 3 → 1 万人 30 床），
    #   round 取整、至少 1 张；NH 关闭时无医疗系统记账，置 0（标题不显示 Hosp/Unmet 段）。
    sim.beds_total = max(1, int(round(CONFIG["BEDS_PER_1000"] * sim.N / 1000.0))) if sim.NH_ENABLED else 0

    # ---- 社会响应层状态（schema v3；关闭时全部为占位符，不消耗随机数）----
    # 子开关 = "档案想开" 且 "总开关开"：避免档案只开 TESTING 时仍去跑 v3 随机抽样。
    sim.IV_ENABLED = bool(CONFIG["INTERVENTIONS_ENABLED"])
    sim.TESTING_ON = bool(CONFIG["TESTING_ENABLED"]) and sim.IV_ENABLED
    sim.TRACING_ON = bool(CONFIG["TRACING_ENABLED"]) and sim.IV_ENABLED
    sim.BEHAVIOR_ON = bool(CONFIG["BEHAVIOR_ENABLED"]) and sim.IV_ENABLED
    sim.REPORTING_ON = bool(CONFIG["REPORTING_ENABLED"]) and sim.IV_ENABLED

    # 计时器哨兵同 v2：-1=未启动、>0=倒计时中、==0=到期（F 段的 _n_test_pending/_n_trace_pending 读它）。
    # 调用方契约：wants_test/test_timer/tested/trace_timer/quarantined/quarantine_timer 在
    #   CONTRACT_TENSORS 里，初值 -1/False 被 golden 基线固化；改变初值等于改轨迹。
    sim.wants_test = torch.zeros(sim.N, dtype=torch.bool, device=device)   # 发病时抽到的检测意愿
    sim.test_timer = torch.full((sim.N,), -1.0, device=device)             # 检测排队剩余帧数；-1=未预约
    sim.tested = torch.zeros(sim.N, dtype=torch.bool, device=device)       # 已出结果（不再重复检测）
    sim.trace_timer = torch.full((sim.N,), -1.0, device=device)            # 阳性后追踪倒计时；-1=未触发
    sim.quarantined = torch.zeros(sim.N, dtype=torch.bool, device=device)  # 密接隔离中
    sim.quarantine_timer = torch.full((sim.N,), -1.0, device=device)       # 隔离剩余帧数

    # 每日检测上限：TEST_CAPACITY_PER_100K_PER_DAY 是"每 10 万人每天可检测数"，按人口换算取整。
    # 下面这些是主机侧 Python 标量（不是张量）：每天只更新一次、只被标题/工具读，
    #   放进 GPU 反而每帧多一次显存→主机同步，因此全部留在主机侧。
    sim.test_capacity_day = (int(round(CONFIG["TEST_CAPACITY_PER_100K_PER_DAY"] * sim.N / 100000.0))
                         if sim.TESTING_ON else 0)
    sim.tests_today = 0
    sim.tests_total = 0
    sim.quarantine_total = 0        # 累计被隔离的密接人数（事件计数）
    sim.onset_today = 0             # 今日新增有症状发病数
    sim.onset_total = 0             # 累计有症状发病数
    sim.reported_today = 0
    sim.reported_total = 0
    sim.behavior_contact_mult = 1.0  # 行为反馈产生的接触乘子（作用于休闲接触与出行概率）
    sim.onset_by_day = []            # 每日有症状发病数（索引=日）
    sim.reported_by_day = []         # 每日新增报告数（索引=日）
    sim.report_events = []           # 每日报告事件数（行为反馈窗口用）
    # _last_stat_day 用于 _daily_step 去重（同一日只结算一次日界事件），初值 -1 表示尚未结算
    sim._last_stat_day = -1

    # ---- 个体异质性 / 接触混合（schema v4；关闭时不消耗任何随机数，旧路径可复现）----
    # RNG 契约：本段的抽样顺序固定为 virus_scale.randn → compliance.randn → symp_home.rand，
    #   且全部在 HET_ON/MIX_ON 内；关闭档案必须做到"一次都不抽"（docs/ARCHITECTURE.md §3）。
    sim.HET_ON = bool(CONFIG["HETEROGENEITY_ENABLED"])
    sim.MIX_ON = bool(CONFIG["MIXING_ENABLED"])
    sim.sigma_infectious = float(CONFIG["INFECTIOUSNESS_SIGMA"])
    # 传染力乘子：对数正态（均值 1），体现超传播/子代过散布
    sim.virus_scale = torch.ones(sim.N, device=device)
    if sim.HET_ON and sim.sigma_infectious > 0.0:
        # -0.5*σ² 是对数正态的均值修正项（E[exp(σZ-σ²/2)]=1）；
        # clamp[0.05,20] 截断极端尾部（见 MECHANISMS.md §12）。
        sim.virus_scale = torch.exp(
            torch.randn(sim.N, device=device) * sim.sigma_infectious
            - 0.5 * sim.sigma_infectious ** 2).clamp_(0.05, 20.0)
        # 归一化到均值 1，保证总体 R0 期望不因抽样噪声漂移
        sim.virus_scale = sim.virus_scale / sim.virus_scale.mean().clamp(min=1e-6)
    # 依从性：个体固定属性（乘在隔离/密接隔离/行为反馈的概率上）；关闭 HET 时恒 1=无个体差异
    sim.compliance = torch.ones(sim.N, device=device)
    if sim.HET_ON and float(CONFIG["COMPLIANCE_SD"]) > 0.0:
        sim.compliance = (float(CONFIG["COMPLIANCE_MEAN"])
                          + float(CONFIG["COMPLIANCE_SD"])
                          * torch.randn(sim.N, device=device)).clamp_(0.0, 1.0)
    # 诊断：SIM_TRACK_SECONDARY=1 时累计每个感染者的二代感染数（不参与模型逻辑）
    _track_secondary = os.environ.get("SIM_TRACK_SECONDARY", "") not in ("", "0")
    sim._secondary = (torch.zeros(sim.N, dtype=torch.float32, device=device)
                      if _track_secondary else None)

    if sim.MIX_ON:
        # 4 组粗年龄 + 同配矩阵 φ(g,h)=1-s+s·1{g==h}/n_g（E[φ]=1 → 总体 R0 期望不变）
        _group_map = torch.tensor(AGE_BAND_TO_GROUP, dtype=torch.long, device=device)
        sim.age_group = _group_map[sim.age_band]
        # share 是各组人口占比，clamp(min=0.02) 防小群组的 1/n_g 爆炸（4 组下只影响极端档案）
        _share = (torch.bincount(sim.age_group, minlength=4).to(torch.float32)
                  / float(max(1, sim.N))).clamp(min=0.02)
        _s = float(CONFIG["AGE_ASSORTATIVITY"])
        # 4x4 展平为 16：行主序 index = 源组*4 + 目标组，对角线（i*4+i=i*5）放同配强化项
        sim.mix_flat = torch.full((16,), 1.0 - _s, device=device)
        sim.mix_flat[torch.arange(4, device=device) * 5] = 1.0 - _s + _s / _share
        sim.hh_mult = float(CONFIG["HOUSEHOLD_CONTACT_MULT"])
        # 有症状者自发减少外出（在检测/隔离之外的行为反应）；activity 按 counts_prev[3] 门控读取
        sim.symp_home = (torch.rand(sim.N, device=device)
                         < float(CONFIG["SYMPTOM_STAY_HOME_FRACTION"]))
    else:
        # 关闭混合时这些字段为 None/1.0：activity/传播段用 is None 或 MIX_ON 分支判断，
        #   不能把 None 当"没有邻居"的空矩阵读。
        sim.age_group = None
        sim.mix_flat = None
        sim.hh_mult = 1.0
        sim.symp_home = None


    # ==================== 免疫系统 ====================
    # 免疫力水平 [0,1]：感染/康复时按 IMMUNITY_GAIN_PER_INFECTION 增益、逐帧按比率衰减；
    #   上限 MAX_IMMUNITY(=0.98) 只在 A 段增益侧 clamp，这里从 0 开始。
    sim.immunity_level = torch.zeros(sim.N, device=device)      # 免疫力水平 [0, 1]

    # 初始免疫（比例/水平由疾病档案给出）。
    # RNG 契约：这是本函数里较早的一次 torch 抽样（前面只有 v4 的 randn/rand，且受开关保护）；
    #   torch 流在本函数内的顺序是：v4 属性 → 初始免疫 → 日程抖动 rand×4 —— 插入新抽样必须放最后，
    #   否则同 seed 的人口结构/初始免疫会整体错位。
    initial_immune = torch.rand(sim.N, device=device) < CONFIG["INITIAL_IMMUNE_FRACTION"]
    sim.status[initial_immune] = 1
    sim.immunity_level[initial_immune] = CONFIG["INITIAL_IMMUNE_LEVEL"]

    # ==================== 初始感染源 ====================
    # 从编号靠前的若干工作场所的就业者中选种子：07:00 感染，上午在岗时进入发病期，
    # 形成可解释的工作场所暴发；人口放大时自动扩展到多个工作场所。
    # 本块不含随机抽样（纯排序/切片）：种子集只由结构（employment/venue）与 INITIAL_SEEDS 决定。
    _employees = torch.where(sim.employed)[0]
    _per_venue = max(1, int(_employees.numel()) // max(1, n_work))
    # 向上取整：保证 INITIAL_SEEDS 个种子在每场馆人数少时也能分散到足够多的场馆
    _n_seed_venues = min(n_work, max(1, (CONFIG["INITIAL_SEEDS"] + _per_venue - 1) // _per_venue))
    _seed_pool = torch.where((sim.work_venue_idx >= 0) & (sim.work_venue_idx < _n_seed_venues))[0]
    seeds = _seed_pool[:CONFIG["INITIAL_SEEDS"]]
    sim.status[seeds] = 2
    sim.virus_beta[seeds] = CONFIG["BASE_TRANSMIT"]
    sim.virus_gamma[seeds] = CONFIG["BASE_RECOVERY"]
    sim.virus_mu[seeds] = CONFIG["BASE_DEATH"]
    sim.virus_sigma[seeds] = CONFIG["MUTATION_RATE"]
    sim.virus_gen[seeds] = 1
    sim.max_incubation[seeds] = CONFIG["INCUBATION_MU"]
    sim.ever_infected[seeds] = True

    sim.lockdown_active = False

    # 统计（工具读 strain_stats；F 段每帧 append 新感染/再感染计数，长度=帧数）
    sim.strain_stats = {'new_infections': [], 'reinfections': []}

    # ==================== 活动/出行状态 ====================
    # place: 0=家 1=工作场所 2=学校 3=商店/公园（休闲）；traveling 期间 place 仍是"上一个地点"，
    #   渲染用 pos/dest 插值、统计用 place 构成时要把在途者排除（F 段的 still_m）。
    sim.place = torch.zeros(sim.N, dtype=torch.long, device=device)
    sim.traveling = torch.zeros(sim.N, dtype=torch.bool, device=device)
    sim.dest = sim.home_pos.clone()
    sim.trip_left = torch.zeros(sim.N, dtype=torch.long, device=device)
    sim.trip_total = torch.zeros(sim.N, dtype=torch.long, device=device)
    sim.trip_place = torch.zeros(sim.N, dtype=torch.long, device=device)
    sim.trip_count = torch.zeros(sim.N, dtype=torch.long, device=device)   # 每人累计出行次数

    # 个体日程抖动（分钟，每人固定；让作息不完全同步）：幅度依次为 ±20 / ±30 / ±10 / ±15 分钟。
    # RNG 契约：这里是连续 4 次 torch.rand(N)，顺序固定为上班→下班→上学→放学，别换序。
    sim.work_start_min = CONFIG["WORK_START_MIN"] + (torch.rand(sim.N, device=device) - 0.5) * 40.0
    sim.work_end_min = CONFIG["WORK_END_MIN"] + (torch.rand(sim.N, device=device) - 0.5) * 60.0
    sim.school_start_min = CONFIG["SCHOOL_START_MIN"] + (torch.rand(sim.N, device=device) - 0.5) * 20.0
    sim.school_end_min = CONFIG["SCHOOL_END_MIN"] + (torch.rand(sim.N, device=device) - 0.5) * 30.0

    # 每日重采样的外出计划（resample_daily_plan 在日界重置；这里只给初值/占位）
    sim.errand_flag = torch.zeros(sim.N, dtype=torch.bool, device=device)
    sim.errand_start_min = torch.zeros(sim.N, device=device)
    sim.errand_kind = torch.zeros(sim.N, dtype=torch.long, device=device)   # 0=商店 1=公园
    sim.errand_pos = sim.home_pos.clone()
    # 休闲停留半径的初值（日界 resample_daily_plan 会按当天目的商店/公园重写；这里给个
    # 与商店口径一致的占位，保证首帧日界之前读不到未定义属性）
    sim.errand_spread = torch.full((sim.N,), float(CONFIG["VENUE_SPREAD"]["shop"] * 0.5),
                                   device=device)
    sim.mobility_stats = {'trip_starts': [], 'trip_durations': []}
    # -1 哨兵：首帧 day_index=0 必然 != _last_day，从而触发第一次日界重采样
    sim._last_day = -1
    # 出行诊断：上车次数在 GPU 上按日累加，到达时长只在每日第一帧采样一次（避免每帧同步）
    sim._trip_starts_acc = torch.zeros((), dtype=torch.long, device=device)
    sim._trip_sample_stride = int(CONFIG["DAY_FRAMES"])
    # 每帧复用的活动缓冲（避免每帧新建 long 张量）
    sim._buf_desired = torch.zeros(sim.N, dtype=torch.long, device=device)

    # ---- 帧首快照与主机门控计数（F 段末尾写回，下一帧 update 开头读；不是"实时计数"）----
    # 帧首状态快照：状态在两次 update 之间不变，因此上一帧末的计数就是本帧首的计数。
    # 用它替代 update 内部的 (status==3).sum().item() / dead.any() 等隐式同步点。
    # 不变式：sum(counts_prev)==N、长度 8（status 0..7）；初始值必须与上面的初始 status 一致。
    _counts = torch.bincount(sim.status, minlength=8).cpu().numpy()
    sim.counts_prev = _counts
    sim._n_dead = int(_counts[4])
    # 受限人数（隔离/密接隔离/住院/无床位等待）之和：为 0 时 mobility 整块跳过。
    # 初始时 quarantine/isolated/untreated 都为空，所以它等于住院数 counts[7]。
    sim._n_restricted = int(_counts[7])
    sim._n_untreated = 0
    # v3 子系统的待办计数（上一帧末统计）：为 0 时整块跳过检测/追踪/隔离处理。
    # 初始无任何检测/追踪/隔离，故全 0；之后每帧由 F 段按计时器哨兵语义重算。
    sim._n_test_pending = 0
    sim._n_trace_pending = 0
    sim._n_quarantine_active = 0
    # 渲染节流：>0 时每帧更新散点（交互模式默认 1）；0 表示不更新散点
    # （吞吐模式/批量实验用 SIM_RENDER_EVERY=0 省掉两次显存→主机传输）
    sim.render_every = int(os.environ.get("SIM_RENDER_EVERY", "1") or 1)
    # ---- 缓存常量张量：每帧都会被读，预先建好以省 kernel/主机开销 ----
    # 位置 clamp 的上下界：存成 (2,) 张量后一次 clamp 即可，避免对 x/y 分别 clamp 共 4 个 kernel
    sim._pos_min = torch.tensor([1.0, 1.0], device=device)
    sim._pos_max = torch.tensor([float(sim.W) - 1.0, float(sim.H) - 1.0], device=device)
    # 状态→是否具传染性（E/I/A=1，其他 0）的 LUT：一次 gather 取代 5 个比较/或 kernel。
    # 不变式：只点亮 2/3/6；H(7) 不参与社区传播，D/R/S/初始免疫都不是传染源。
    sim._inf_lut = torch.zeros(8, dtype=torch.bool, device=device)
    sim._inf_lut[2] = True
    sim._inf_lut[3] = True
    sim._inf_lut[6] = True
    # 至此所有字段创建完毕：运行期只在既有字段上原地更新，不再新增/替换张量。
    return sim
