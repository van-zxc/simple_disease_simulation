"""活动/出行模型：日界重采样日程 + 逐帧推进位置。

负责：日程（是否/何时出门、去哪类场所）、在途推进与到达、停留段的随机游走；不负责
状态机/传播判定（simulation.py）与检测/隔离的开关语义（interventions.py）。

入口：``resample_daily_plan`` 在日界由 ``advance_mobility`` 调用，后者每帧调用一次。

不变式：两处 ``randn`` 必须保持整幅 ``(N, 2)`` 精确形状（外出散布、停留抖动）——只对
"当前外出/停留者"抽 ``(n, 2)`` 会改变随机数消耗序列，破坏 golden/toy 回归；``desired``
优先级 homebound > 工作/学校 > 外出 > 家（后写覆盖先写）；掩码写一律 where/masked_fill_，
布尔索引会隐式 nonzero 并强制一次主机同步。
"""

import torch

from .config import CONFIG, PLACE_HOME, PLACE_LEISURE, PLACE_SCHOOL, PLACE_WORK, device

# 性能：诊断样本（平均出行时长/每日出行次数）按天聚合后封顶 65536 条，
# 否则长跑下 list 无界增长（旧实现每帧 .cpu().tolist() 泄漏主机内存）；
# 该容器只服务诊断，不参与模型语义。
_MAX_TRIP_SAMPLES = 65536


def _trip_frames(distance):
    """距离 → 出行帧数：ceil(distance / TRIP_SPEED)，再 clamp 到 [MIN, MAX] 帧。

    不变式：返回 long 张量，行程帧数即 trip_total，trip_left 每帧减 1，因此恰好
    trip_total 帧后到达（推进用 1/trip_left 的 lerp，末帧精确落在 dest）；距离为 0
    时 ceil 得 0，由 TRIP_MIN_FRAMES 抬起，保证 trip_left 一定能在有限帧内归零。
    """
    return torch.clamp(
        torch.ceil(distance / CONFIG["TRIP_SPEED"]),
        min=CONFIG["TRIP_MIN_FRAMES"], max=CONFIG["TRIP_MAX_FRAMES"]
    ).long()


def _spread_lut(sim):
    """停留半径 LUT：按 place 码 0..3 索引（家/工作/学校/休闲四类，顺序固定）。

    家固定 0.6；其余为 VENUE_SPREAD[kind] * 0.5（半场馆尺度，人不会飘出场馆太远）。
    不变式：CONFIG 要等档案应用 + 人口缩放后才定型，所以只能在首次使用时构建；缓存
    挂在 ``sim`` 上而不是模块级——不同 sim 可能对应不同档案/人口，不能跨 sim 复用。
    """
    lut = getattr(sim, "_spread_lut", None)
    if lut is None:
        spread = CONFIG["VENUE_SPREAD"]
        lut = torch.tensor(
            [0.6, spread["work"] * 0.5, spread["school"] * 0.5, spread["shop"] * 0.5],
            dtype=torch.float32, device=device)
        sim._spread_lut = lut
    return lut


def _place_spread(place_idx, lut=None):
    """兼容旧接口：按场所类别张量返回逐人游走半径（内部改为 LUT 查表）。

    当前包内已无调用方（只有 MECHANISMS.md 还按旧名描述），保留给外部脚本；
    注意入参必须是张量（要取 ``place_idx.device``），传 Python int 会抛 AttributeError。
    """
    if lut is None:
        lut = _spread_lut_for(place_idx.device)
    return lut[place_idx]


def _spread_lut_for(dev):
    """无缓存版 ``_spread_lut``：按 device 现场构建，只给上面的兼容接口使用。

    与 ``_spread_lut`` 的 0.6 / 0.5 系数是同一套口径，改动时两处必须同步。
    """
    spread = CONFIG["VENUE_SPREAD"]
    return torch.tensor(
        [0.6, spread["work"] * 0.5, spread["school"] * 0.5, spread["shop"] * 0.5],
        dtype=torch.float32, device=dev)


def resample_daily_plan(sim, day_index):
    """重采样一天的出行计划：是否出门、何时出门、去商店还是公园（日界调用一次）。

    RNG 契约（顺序固定，改动会破坏 golden/toy 回归）：rand(N) 外出抽签 → rand(N) 出门
    时刻 → rand(N) 公园/商店二选一 → 两次 randint 选场馆 → randn((N,2)) 落点散布。
    外出概率 = ERRAND_PROB_* × 行为乘子；HET_ON 时按个体依从性插值（mult=1 退化为原概率）。
    """
    weekday = (day_index % 7) not in CONFIG["WEEKEND_DAYS"]
    prob = CONFIG["ERRAND_PROB_WEEKDAY"] if weekday else CONFIG["ERRAND_PROB_WEEKEND"]
    if sim.BEHAVIOR_ON:
        if sim.HET_ON:
            # 依从性异质性：行为信号对高依从个体效果更强（mult=1 时退化为原概率）
            prob_t = prob * (1.0 - sim.compliance
                             * (1.0 - float(sim.behavior_contact_mult)))
        else:
            prob_t = prob * sim.behavior_contact_mult
    else:
        prob_t = prob
    sim.errand_flag = torch.rand(sim.N, device=device) < prob_t

    # 出门时刻窗口 [ERRAND_START_MIN, ERRAND_END_MIN - ERRAND_DURATION_MIN]：上界先减
    # 时长，保证"出门 + 停留"整段落在日间窗口内
    lo = torch.full((sim.N,), float(CONFIG["ERRAND_START_MIN"]), device=device)
    hi = torch.full((sim.N,), float(CONFIG["ERRAND_END_MIN"] - CONFIG["ERRAND_DURATION_MIN"]),
                    device=device)
    if weekday:
        # 就业者/学生只在下班/放学后出门；hi 此刻对全幅同值，取 hi[0] 当主机侧标量上界
        # （省一次整幅比较；hi 后面还会被 clamp，语义不变）
        lo = torch.where(sim.employed, torch.clamp(sim.work_end_min + 30.0, max=float(hi[0])), lo)
        lo = torch.where(sim.student, torch.clamp(sim.school_end_min + 30.0, max=float(hi[0])), lo)
    # 不变式：窗口必须非退化（hi > lo），否则 rand*(hi-lo) 恒 0、所有人同一时刻出门
    hi = torch.clamp(hi, min=lo + 1.0)
    sim.errand_start_min = lo + torch.rand(sim.N, device=device) * (hi - lo)

    sim.errand_kind = (torch.rand(sim.N, device=device) < 0.35).long()  # 1 = 公园；0.35 是公园:商店的经验比例，非校准参数
    shop_pick = torch.randint(0, sim.n_shop, (sim.N,), device=device)
    park_pick = torch.randint(0, sim.n_park, (sim.N,), device=device)
    sim.errand_pos = torch.where(sim.errand_kind.unsqueeze(1).bool(),
                                 sim.park_venue[park_pick], sim.shop_venue[shop_pick])
    # 外出落点在场馆尺度内随机散布，避免大人口下商店/公园被压缩成"单点"；
    # RNG 契约：本函数第 5 次抽取，形状必须是整幅 (N, 2)
    errand_spread = torch.where(sim.errand_kind.bool(),
                                torch.tensor(CONFIG["VENUE_SPREAD"]["park"] * 0.5, device=device),
                                torch.tensor(CONFIG["VENUE_SPREAD"]["shop"] * 0.5, device=device))
    sim.errand_pos = sim.errand_pos + torch.randn((sim.N, 2), device=device) * errand_spread.unsqueeze(1)


def advance_mobility(sim, frame, clock_min, day_index, alive=None):
    """按"家庭—场所—日程"推进一帧位置：日界重采样 → 改道 → 新行程 → 在途 → 停留。

    每帧调用一次；``alive`` 由 update() 传入（同帧内本函数不改状态，省一次 status!=4）。
    日界（``day_index != sim._last_day``）先归档上一日出行统计、再重采样日程，因此日程
    的随机数消耗发生在日界那一帧。封控且 LOCKDOWN_SHELTER=False 时原地冻结并直接返回
    （旧行为：此时 homebound/desired 一概不算）。
    """
    if day_index != sim._last_day:
        # 日界：把上一日的出行统计一次性归档（诊断用，不参与模型语义）
        if sim._last_day >= 0:
            sim.mobility_stats["trip_starts"].append(int(sim._trip_starts_acc.item()))
            sim._trip_starts_acc.zero_()
        resample_daily_plan(sim, day_index)
        sim._last_day = day_index

    # 存活掩码由 update() 传入（同一帧内 mobility 不会改状态，避免重复计算 status != 4）
    if alive is None:
        alive = sim.status != 4
    # 帧首死亡数来自上一帧末的统计（状态在帧间不变），无需 dead.any() 的隐式同步
    if sim._n_dead:
        sim.traveling.logical_and_(alive)
        sim.trip_left.masked_fill_(~alive, 0)

    # 配置为旧行为时：封控期间原地冻结（homebound 等下面再算，避免无用计算）
    if sim.lockdown_active and not CONFIG["LOCKDOWN_SHELTER"]:
        return

    # 自然史行为约束：隔离者、住院者、无床位等待者、密接隔离者都居家。
    # 用上一帧末的受限人数做门控：为 0 时整块跳过（常见于 toy / 疫情早期）
    # v4 混合结构额外让一部分有症状者自发居家（在检测/隔离之外的行为反应）
    _symp_stay = None
    if sim.MIX_ON and sim.symp_home is not None and int(sim.counts_prev[3]):
        # counts_prev[3] = 上一帧末有症状人数；为 0 时 (status==3)&symp_home 必为空，
        # 跳过整幅比较
        _symp_stay = (sim.status == 3) & sim.symp_home
    homebound = None
    if sim._n_restricted or _symp_stay is not None:
        homebound = alive & (sim.isolated | sim.quarantined | (sim.status == 7) | sim.untreated)
        if _symp_stay is not None:
            homebound = homebound | _symp_stay

    weekday = (day_index % 7) not in CONFIG["WEEKEND_DAYS"]
    desired = sim._buf_desired.zero_()   # 预分配缓冲，原地清零，避免每帧新建 long 张量
    if not (sim.lockdown_active and CONFIG["LOCKDOWN_SHELTER"]):
        if weekday:
            work_ok = (sim.employed & (clock_min >= sim.work_start_min) &
                       (clock_min < sim.work_end_min))
            school_ok = (sim.student & (clock_min >= sim.school_start_min) &
                         (clock_min < sim.school_end_min))
            desired.masked_fill_(work_ok, PLACE_WORK)
            desired.masked_fill_(school_ok, PLACE_SCHOOL)
        # errand_ok 额外要求 desired 仍为"家"：工作/学校优先级更高，出门只补空档
        errand_ok = (sim.errand_flag & (desired == PLACE_HOME) &
                     (clock_min >= sim.errand_start_min) &
                     (clock_min < sim.errand_start_min + CONFIG["ERRAND_DURATION_MIN"]))
        desired.masked_fill_(errand_ok, PLACE_LEISURE)
    # 不变式：homebound 是优先级最高的一项，最后覆盖 → homebound > 工作/学校 > 外出 > 家
    if homebound is not None:
        desired.masked_fill_(homebound, PLACE_HOME)

    # 目标坐标：家 / 工作 / 学校 / 休闲（where 链，避免布尔掩码写触发 nonzero）
    target = sim.home_pos
    target = torch.where((desired == PLACE_WORK).unsqueeze(1), sim.work_pos, target)
    target = torch.where((desired == PLACE_SCHOOL).unsqueeze(1), sim.school_pos, target)
    target = torch.where((desired == PLACE_LEISURE).unsqueeze(1), sim.errand_pos, target)

    # 封控：已在途的非回家行程改道回家。位置已经变了，旧 trip_total/trip_left 对应旧
    # 目标，必须按到家的新距离重算并重置计数，否则会瞬间"到达"或走出世界。
    # 这里允许 nonzero：只在封控切换后的在途者上执行，不是每帧热路径。
    if sim.lockdown_active and CONFIG["LOCKDOWN_SHELTER"]:
        ridx = (alive & sim.traveling & (sim.trip_place != PLACE_HOME)).nonzero().squeeze(1)
        if ridx.numel():
            trip_len = _trip_frames(torch.norm(sim.home_pos[ridx] - sim.pos[ridx], dim=1))
            sim.dest[ridx] = sim.home_pos[ridx]
            sim.trip_total[ridx] = trip_len
            sim.trip_left[ridx] = trip_len
            sim.trip_place[ridx] = PLACE_HOME

    # 自然史行为约束：隔离/住院/等待床位者若在途，改道回家。
    # 性能：用整幅 where（home_trip 对全幅算一次 norm），换掉 nonzero + 4 次 index_put，
    # 避免每帧一次主机同步——这正是本模块"掩码写优先"的原因。
    if homebound is not None:
        redir = homebound & sim.traveling & (sim.trip_place != PLACE_HOME)
        home_trip = _trip_frames(torch.norm(sim.home_pos - sim.pos, dim=1))
        sim.dest = torch.where(redir.unsqueeze(1), sim.home_pos, sim.dest)
        sim.trip_total = torch.where(redir, home_trip, sim.trip_total)
        sim.trip_left = torch.where(redir, home_trip, sim.trip_left)
        sim.trip_place = sim.trip_place.masked_fill(redir, PLACE_HOME)

    # 发起新行程：期望地点与当前地点不同，且不在途中（全部为确定性写入）。
    # trip_place 记录目的地类别，到达时写回 place；trip_total/trip_left 都置为同一行程帧数
    start = alive & (~sim.traveling) & (desired != sim.place)
    trip_len = _trip_frames(torch.norm(target - sim.pos, dim=1))
    sim.dest = torch.where(start.unsqueeze(1), target, sim.dest)
    sim.trip_place = torch.where(start, desired, sim.trip_place)
    sim.trip_total = torch.where(start, trip_len, sim.trip_total)
    sim.trip_left = torch.where(start, trip_len, sim.trip_left)
    sim.traveling.logical_or_(start)
    sim.trip_count.add_(start)          # long += bool 可直接原地累加，无需显式 dtype 转换
    sim._trip_starts_acc.add_(start.sum())

    # 旅途推进：每帧走剩余距离的 1/trip_left，最后一帧精确到达
    moving = alive & sim.traveling
    left = sim.trip_left.clamp(min=1)
    moved = torch.lerp(sim.pos, sim.dest, (1.0 / left).unsqueeze(1))
    sim.pos = torch.where(moving.unsqueeze(1), moved, sim.pos)
    sim.trip_left.add_(moving, alpha=-1)
    arrived = moving & (sim.trip_left <= 0)
    # 性能：诊断样本按天采样（stride = DAY_FRAMES）而不是每帧 tolist，且总量封顶；
    # 采样只影响统计，不影响模型语义
    if frame % sim._trip_sample_stride == 0:
        durations = sim.mobility_stats["trip_durations"]
        room = _MAX_TRIP_SAMPLES - len(durations)
        if room > 0:
            sample = sim.trip_total[arrived]
            durations.extend(sample[:room].cpu().tolist())
    # 到达帧：位置精确落到 dest，并把 trip_place 写进 place（place 只在到达时更新）
    sim.pos = torch.where(arrived.unsqueeze(1), sim.dest, sim.pos)
    sim.traveling.logical_and_(~arrived)
    sim.place = torch.where(arrived, sim.trip_place, sim.place)

    # 停留：轻微随机游走，并限制在目的场所半径内（避免长期漂移）
    # 整幅计算：不再对 stidx 做 nonzero + 3 次 gather + 2 次 index_put。
    # Windows/WDDM 下每次 indexing kernel 的 launch 开销远大于 10k 规模的算术本身，
    # 且 nonzero 还会引入一次主机同步。
    still = alive & (~sim.traveling) & (desired == sim.place)
    # RNG 契约：必须是整幅 (N, 2)；0.12 是每帧的游走步长（世界坐标单位）
    jitter = torch.randn((sim.N, 2), device=device) * 0.12
    walked = sim.pos + jitter
    rad = torch.norm(walked - target, dim=1)
    allowed = _spread_lut(sim)[sim.place]
    over = rad > allowed
    # 越界者沿"目标→新位置"方向投影回半径圆上（保留方向、不重置到中心，避免视觉跳变）
    fixed = (target
             + (walked - target) / rad.clamp(min=1e-6).unsqueeze(1) * allowed.unsqueeze(1))
    sim.pos = torch.where(
        (still & over).unsqueeze(1), fixed,
        torch.where(still.unsqueeze(1), walked, sim.pos))

    # 位置 clamp：一次 elementwise clamp（x/y 上下界用 (2,) 常量张量广播），替代 4 个 kernel
    sim.pos.clamp_(sim._pos_min, sim._pos_max)
