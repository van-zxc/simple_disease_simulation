"""逐帧仿真步：病程推进、传播判定与统计渲染（对外唯一入口 ``update``）。

负责：一帧内的阶段编排（阶段表见 ``update`` 顶部）、病程/死亡康复、接触对传播、
      统计聚合与 artist 更新；不负责初始状态（``state``）、日程出行（``activity``）、
      接触对枚举（``contacts``）、传染力曲线（``disease``）、v3 干预（``interventions``）、
      绘图对象与样式（``render``）。
对外入口：``update(sim, frame)``，返回需重绘的 artist 列表（FuncAnimation 消费）。
必须保持的不变式：
  - RNG 契约：随机抽取的次数/形状/顺序由 CONFIG 开关组合决定（docs/ARCHITECTURE.md §3）；
    新增随机数必须由开关包住，关闭档案下不能多抽，否则同 seed 轨迹整体偏移。
  - 不变式：sum(counts)==N；stat 为 24 维布局（见 F 段），标题/曲线都由它取数。
  - 可变态全挂 ``sim`` 上；本模块唯一跨帧状态是 ``_MUTATION_CONSTS`` 常量缓存。
"""

import numpy as np
import torch

from .activity import advance_mobility
from .config import CONFIG, PLACE_HOME, PLACE_LEISURE, PLACE_SCHOOL, PLACE_WORK, device
from .contacts import _contact_pairs_dense, _contact_pairs_grid
from .disease import _curve_asym, _curve_eval, _curve_symp
from .interventions import (
    _advance_quarantine,
    _daily_step,
    _process_tests,
    _resolve_traces,
)
from .render import COLOR_LUT

# 空索引常量：主机侧整数门控判定"本帧没有该类个体"时用它占位，
# 免去对 0 元素再做一次 nonzero，且 (0,) 的 long 索引可直接参与后续 index_select。
_EMPTY_INDEX = torch.empty(0, dtype=torch.long, device=device)

# 变异参数常量缓存：只依赖档案应用后的 CONFIG（运行期不再变），首次使用时构建。
# 三行依次对应 beta/gamma/mu；gamma 与 mu 的变异幅度是 beta 的 0.3 倍（见 _mutation_consts）。
_MUTATION_CONSTS = {}


def _mutation_consts():
    """构建并缓存变异所需的常量张量：行缩放 (3,1) + 下界/上界 (3,1)。

    三行固定为 beta/gamma/mu；gamma 与 mu 的变异步长取 beta 的 0.3 倍，
    与旧版"三套独立缩放"的数值等价（折算进同一张 (3,1) 张量）。
    首次调用时构建、整运行复用：CONFIG 在档案应用后就是只读的。
    """
    consts = _MUTATION_CONSTS.get("v")
    if consts is None:
        scale = float(CONFIG["MUTATION_SCALE"])
        bounds = CONFIG["MUTATION_BOUNDS"]
        # 0.3：gamma/mu 变异步长相对 beta 的比例（旧版三行各自 scale 的等价折算）
        rows = torch.tensor([[scale], [scale * 0.3], [scale * 0.3]],
                            dtype=torch.float32, device=device)
        lo = torch.tensor([bounds["beta"][0], bounds["gamma"][0], bounds["mu"][0]],
                          dtype=torch.float32, device=device)
        hi = torch.tensor([bounds["beta"][1], bounds["gamma"][1], bounds["mu"][1]],
                          dtype=torch.float32, device=device)
        # 下界/上界先 unsqueeze 成列向量，供 (3, num_new) 整块 clamp_ 广播
        consts = (rows, lo.unsqueeze(1), hi.unsqueeze(1))
        _MUTATION_CONSTS["v"] = consts
    return consts


def update(sim, frame):
    # 一帧阶段表（顺序即语义：先动状态还是先读快照会改变轨迹与 counts 口径）：
    #   A0 日界事件：v3 每日排程/报告/行为反馈 + 密接隔离倒计时（仅 IV_ENABLED）
    #   A  压力与免疫：用上一帧 counts_prev 判定医疗崩溃、免疫衰减/消退回 S
    #   B  活动出行：advance_mobility（日程、封控返家、homebound 门控）
    #   C  帧首快照：symp_prev / rec_prev / incu_mask（本帧新转者不参与本帧抽签与传播）
    #   D  病程：NATURAL_HISTORY_ENABLED 走自然史分支（含 v3 检测/追踪），否则旧路径
    #   E  传播：接触对 → 稀疏概率 → 每目标至多一次感染 → 变异/免疫/计数器写回
    #   F  统计与渲染：GPU 聚合 24 维 stat → 单次 .cpu() → 标题/曲线/散点
    #   G  显存：每 50 帧检查 CUDA 缓存碎片
    # 不变式：A 段只读上一帧末写回的 counts_prev（状态在两次 update 之间不变），
    #         因此 A 段不能依赖本帧 D 段的结果；status 只在 D/E 段写。
    # 详见 docs/ARCHITECTURE.md §3（一帧数据流）与 docs/CONTRIBUTING.md §3.6。

    # 行为时钟（第 0 帧 = 07:00；仅用于日程/标题，疾病速率仍按帧计）
    clock_min = (CONFIG["CLOCK_START_MIN"] + frame * sim.MIN_PER_FRAME) % 1440.0
    day_index = int((CONFIG["CLOCK_START_MIN"] + frame * sim.MIN_PER_FRAME) // 1440)

    # A0. 社会响应日界事件（检测排程/报告/行为反馈）与密接隔离倒计时
    #     RNG 契约：整个 v3 层只在 IV_ENABLED 时执行——关闭时连一次随机数都不抽，
    #     保证 toy / v2 档案（同 seed）轨迹与引入 v3 前逐位一致。
    #     门控读 _n_quarantine_active（上一帧 F 段写回）：没有隔离者时连倒计时都不启动。
    if sim.IV_ENABLED:
        _daily_step(sim, day_index)
        if sim._n_quarantine_active:
            _advance_quarantine(sim, )

    # A. 医疗系统压力（用帧首症状数；本帧的病程与传播都基于该状态）
    #     帧首计数取自上一帧末的统计快照，省掉一次 (status==3).sum().item() 同步。
    num_symptomatic = int(sim.counts_prev[3])
    is_collapsed = num_symptomatic >= CONFIG["MEDICAL_THRESHOLD"]
    # 2.5：旧版"医疗崩溃压力"乘子（死亡 hazard×2.5、康复率÷2.5，见 REALISM.md 的医疗崩溃条目）。
    #     自然史路径不使用该乘子：F 段会按床位/无床人数重新定义 COLLAPSE。
    medical_pressure = 2.5 if is_collapsed else 1.0
    # 帧首各类人数（来自上一帧末快照）：用来整块跳过"空集合"分支。
    # 性能：Windows/WDDM 下每次 kernel launch 约十几微秒，少量元素也照付；
    #       用主机侧整数门控比在 GPU 上空跑一堆 kernel 便宜，且不引入额外同步。
    n_E = int(sim.counts_prev[2])
    n_A = int(sim.counts_prev[6])
    n_H = int(sim.counts_prev[7])
    n_active_inf = n_E + num_symptomatic + n_A

    # 免疫力衰减与消失；衰减到阈值以下后重新变为易感
    # 性能：死亡个体的免疫值不再可见（统计按存活掩码、再感染只对存活目标），
    #       因此整幅 sub_/clamp_ 与"只对存活者做 where"等价，但少 2 个 kernel。
    alive_mask = sim.status != 4
    # IMMUNITY_DECAY_RATE=0.0005/帧：约 2000 帧（≈8 天）把满免疫磨到 0；上限只在增益侧 clamp
    sim.immunity_level.sub_(CONFIG["IMMUNITY_DECAY_RATE"]).clamp_(min=0.0)
    # status∈{1,5} 已隐含"存活"，不需要再与 alive_mask 相与
    waning = ((sim.status == 1) | (sim.status == 5)) & (
        sim.immunity_level <= CONFIG["IMMUNITY_WANING_THRESHOLD"])
    sim.status.masked_fill_(waning, 0)

    # B. 活动/出行（家庭—场所—日程驱动；含封控返家）
    #    advance_mobility 用 alive_mask 清在途死者，并用上一帧写回的
    #    _n_restricted/_n_dead 做主机门控；它只改 pos/place/traveling/dest，不动 status。
    advance_mobility(sim, frame, clock_min, day_index, alive_mask)

    # C. 帧首快照（D/E 段唯一的只读基线；必须在任何 status 写入之前取）
    # 记录本帧开始时的症状者：新转症状者本帧不参与死亡/康复抽签
    symp_prev = (sim.status == 3)
    # 记录帧首已康复者：本帧新康复/自愈者不进入本帧传播目标（否则同帧可再感染）
    rec_prev = (sim.status == 5)
    incu_mask = (sim.status == 2)

    # D. 病程（schema v2 自然史分支；关闭时走下方旧路径。两条路径互斥）
    if CONFIG["NATURAL_HISTORY_ENABLED"]:
        # 计时：感染后帧数（E/I/A）与发病后帧数（I）
        # 性能：掩码加 1 改成整幅加（bool→float 只有 0/1），避免布尔索引内部 nonzero；
        #       门控 n_active_inf（帧首无 E/I/A）时整块跳过。
        if n_active_inf:
            inf_now = sim._inf_lut[sim.status]
            sim.inf_elapsed.add_(inf_now)      # float += bool 由 PyTorch 类型提升，免一次 .to()
            sim.symp_elapsed.add_(symp_prev)

        # 潜伏期结束：按比例分流为无症状/有症状；有症状者抽是否隔离、是否愿检测
        # 性能：先取索引——nonzero 是本块唯一需要主机同步的步骤，后续 gather 全复用索引
        incu_idx = (incu_mask.nonzero().squeeze(1) if n_E
                    else _EMPTY_INDEX)
        if incu_idx.numel():
            sim.timer.add_(incu_mask)
            to_onset = sim.timer[incu_idx] >= sim.max_incubation[incu_idx]
            onset_idx = incu_idx[to_onset]
            if onset_idx.numel():
                # RNG 契约：本块抽取顺序固定为 无症状分流 → 隔离 → 检测意愿 →（下方）自愈，
                # 每步长度都等于本帧到期人数；换顺序/合并会改变同 seed 轨迹。
                asym_draw = (torch.rand(onset_idx.numel(), device=device)
                             < CONFIG["ASYMPTOMATIC_FRACTION"])
                asym_idx = onset_idx[asym_draw]
                sym_idx = onset_idx[~asym_draw]
                if asym_idx.numel():
                    sim.status[asym_idx] = 6
                    sim.a_timer[asym_idx] = CONFIG["ASYMPTOMATIC_DAYS_FRAMES"]
                if sym_idx.numel():
                    sim.status[sym_idx] = 3
                    sim.symp_elapsed[sym_idx] = 0.0
                    # 计数用 int(numel)：只在本帧真有发病者时才付出一次主机同步
                    sim.onset_today += int(sym_idx.numel())
                    sim.onset_total += int(sym_idx.numel())
                    if sim.HET_ON:
                        # v4 依从性异质性：隔离决定按个体 compliance 缩放
                        _iso_p = CONFIG["ISOLATION_PROBABILITY"] * sim.compliance[sym_idx]
                    else:
                        _iso_p = CONFIG["ISOLATION_PROBABILITY"]
                    # RNG 契约：即使 _iso_p=0 也照抽一次——抽取形状只由人数决定、与参数值无关。
                    iso_draw = (torch.rand(sym_idx.numel(), device=device) < _iso_p)
                    # 注意 delay=0 时写 0：下方"隔离倒计时"块会在本帧就把 isolated 置真
                    sim.iso_timer[sym_idx[iso_draw]] = CONFIG["ISOLATION_DELAY_FRAMES"]
                    if sim.TESTING_ON:
                        sim.wants_test[sym_idx] = (
                            torch.rand(sym_idx.numel(), device=device)
                            < CONFIG["TEST_PROBABILITY_SYMPTOMATIC"])
            # 自愈（旧语义：上尾判定 rand > 1-rate；rate=0 时整块跳过，连 rand 都不抽）
            # RNG 契约：不要"优化"成 rand < rate —— 概率相同但用掉的随机数不同，同 seed 轨迹会变。
            if CONFIG["INCUBATION_SELF_RECOVERY"] > 0:
                self_recov = (~to_onset) & (
                    torch.rand(incu_idx.numel(), device=device)
                    > (1.0 - CONFIG["INCUBATION_SELF_RECOVERY"]))
                rec_idx = incu_idx[self_recov]
                if rec_idx.numel():
                    sim.status[rec_idx] = 5
                    # 免疫增益用 index_add_ + 整幅 clamp_：比"索引读-加-写"少 3 个 kernel。
                    # 病毒参数在离开 E/I/A 后不可观测（传染力/统计都按感染者掩码读取，
                    # 再感染时由 transmission 全量覆写），故不再逐症状清零。
                    sim.immunity_level.index_add_(
                        0, rec_idx,
                        torch.full((rec_idx.numel(),),
                                   CONFIG["IMMUNITY_GAIN_PER_INFECTION"],
                                   dtype=sim.immunity_level.dtype, device=device))
                    sim.immunity_level.clamp_(0, CONFIG["MAX_IMMUNITY"])

        # 无症状者按 a_timer 固定病程康复（确定性写入、无随机数 → where/masked_fill 免 nonzero）
        # 门控 n_A or n_E：本帧新转无症状者（上一帧还是 E）也要在当帧扣一次计时，
        # 否则其无症状病程会多一帧、与旧模型口径不一致。
        if n_A or n_E:
            a_mask = (sim.status == 6)
            sim.a_timer.add_(a_mask, alpha=-1)   # float -= bool（alpha=-1 免 .to()）
            rec = a_mask & (sim.a_timer <= 0.0)
            sim.status.masked_fill_(rec, 5)
            # 免疫增益 = 原地布尔式加 + 整幅 clamp（3 个 kernel；原 where+clamp 多建一个张量）
            sim.immunity_level.add_(rec, alpha=CONFIG["IMMUNITY_GAIN_PER_INFECTION"])
            sim.immunity_level.clamp_(0, CONFIG["MAX_IMMUNITY"])

        # 隔离倒计时：发病后延迟若干帧转为居家（mobility 的 homebound 判定读 isolated）
        # 性能/不变式：m_symp_now 只算一次，下面的社区结局块复用它（省 2 个比较 kernel）；
        #   这里必须用"本帧现状"而不是 symp_prev —— delay=0 时新发病者要同帧立即隔离。
        #   门控含 IV_ENABLED：检测触发的隔离不依赖 ISOLATION_PROBABILITY。
        m_symp_now = sim.status == 3
        if (num_symptomatic or n_E) and (
                CONFIG["ISOLATION_PROBABILITY"] > 0 or sim.IV_ENABLED):
            m_symp_open = m_symp_now & (~sim.isolated)
            iso_pending = m_symp_open & (sim.iso_timer > 0)
            sim.iso_timer.add_(iso_pending, alpha=-1)
            newly_iso = m_symp_open & (sim.iso_timer == 0)
            sim.isolated.logical_or_(newly_iso)

        # D.1 社区结局（用帧首状态；本帧新转者不抽签）
        # 1) 未住院症状者：社区死亡/康复；需住院但无床位者死亡风险放大
        # 不变式：comm_mask ⊆ symp_prev，因此 cidx 最多 num_symptomatic 个，
        #         这里拿 num_symptomatic 当判空门控可以省一次 bool(any())/nonzero。
        comm_mask = symp_prev & m_symp_now
        cidx = (comm_mask.nonzero().squeeze(1) if num_symptomatic
                else _EMPTY_INDEX)
        if cidx.numel():
            # 未住院者：普通社区病程 vs 需要住院但无床位（按住院疗程的死亡风险放大）
            death_hazard = torch.where(
                sim.untreated[cidx],
                CONFIG["UNTREATED_DEATH_HAZARD"], CONFIG["COMMUNITY_DEATH_HAZARD"])
            # severity_mult 是年龄别相对风险按人口均值归一化（均值 1），只改个体分布不改总体 IFR
            death_hazard = death_hazard * sim.severity_mult[cidx]
            # RNG 契约：每人固定抽两次、先死亡后康复；即使 hazard=0 也照抽（形状=人数），
            #            不要为"没人会死"加短路，否则后续随机流错位。
            is_dead = torch.rand(cidx.numel(), device=device) < death_hazard
            is_rec = (~is_dead) & (
                torch.rand(cidx.numel(), device=device) < CONFIG["COMMUNITY_RECOVERY_RATE"])
            dead_idx = cidx[is_dead]
            rec_idx = cidx[is_rec]
            if dead_idx.numel():
                sim.status[dead_idx] = 4
                sim.isolated[dead_idx] = False
                sim.untreated[dead_idx] = False
                # 同帧死亡者立即结束在途标记：否则统计/断言会看到"死者在途"
                sim.traveling[dead_idx] = False
                sim.trip_left[dead_idx] = 0
                # 死者不再是"密接隔离中"：否则 stat[18]/stat[22] 把死者算进隔离数，
                # 且 _n_quarantine_active 门控一直为真（2026-11 审计修复）
                sim.quarantined[dead_idx] = False
                sim.quarantine_timer[dead_idx] = -1.0
            if rec_idx.numel():
                sim.status[rec_idx] = 5
                sim.immunity_level.index_add_(
                    0, rec_idx,
                    torch.full((rec_idx.numel(),),
                               CONFIG["IMMUNITY_GAIN_PER_INFECTION"],
                               dtype=sim.immunity_level.dtype, device=device))
                sim.immunity_level.clamp_(0, CONFIG["MAX_IMMUNITY"])
                sim.isolated[rec_idx] = False
                sim.untreated[rec_idx] = False

        # D.2 住院者：病房死亡/康复（按年龄严重性缩放死亡风险）
        # ward_dead / ward_rec 用主机 int 记账（本帧病房死亡数 / 康复数），供 D.3/D.4 算空床：
        # 两者都释放床位、都必须在空床公式里扣除，否则同帧释放的床本帧不可用。
        ward_dead = 0
        ward_rec = 0
        hidx = ((sim.status == 7).nonzero().squeeze(1) if n_H else _EMPTY_INDEX)
        if hidx.numel():
            death_hazard = CONFIG["WARD_DEATH_HAZARD"] * sim.severity_mult[hidx]
            # RNG 契约：每人两次 rand、先死亡后康复（与 D.1 同款），形状=hidx 人数
            is_dead = torch.rand(hidx.numel(), device=device) < death_hazard
            is_rec = (~is_dead) & (
                torch.rand(hidx.numel(), device=device) < CONFIG["WARD_RECOVERY_RATE"])
            dead_idx = hidx[is_dead]
            rec_idx = hidx[is_rec]
            if dead_idx.numel():
                ward_dead = int(dead_idx.numel())
                sim.status[dead_idx] = 4
                sim.untreated[dead_idx] = False
                sim.quarantined[dead_idx] = False
                sim.quarantine_timer[dead_idx] = -1.0
            if rec_idx.numel():
                ward_rec = int(rec_idx.numel())
                sim.status[rec_idx] = 5
                sim.immunity_level.index_add_(
                    0, rec_idx,
                    torch.full((rec_idx.numel(),),
                               CONFIG["IMMUNITY_GAIN_PER_INFECTION"],
                               dtype=sim.immunity_level.dtype, device=device))
                sim.immunity_level.clamp_(0, CONFIG["MAX_IMMUNITY"])

        # D.3 住院入场：发病满延迟后按年龄别住院率抽签；床位满则无床位等待
        # 不变式：admission_attempted 一旦置真就不再抽签（每人一生只尝试一次入院），
        #         因此本块随机数消耗 = 本帧新进入"可入院"窗口的人数，与床位无关。
        admitted = 0
        if num_symptomatic or n_E:
            admit_ready = ((sim.status == 3) & (~sim.admission_attempted)
                           & (sim.symp_elapsed >= CONFIG["HOSPITAL_DELAY_FRAMES"]))
            aidx = admit_ready.nonzero().squeeze(1)
        else:
            aidx = _EMPTY_INDEX
        if aidx.numel():
            sim.admission_attempted[aidx] = True
            hosp_p = (CONFIG["HOSPITALIZATION_RATE"] * sim.hosp_mult[aidx]).clamp(0.0, 1.0)
            cand = aidx[torch.rand(aidx.numel(), device=device) < hosp_p]
            admitted = 0
            if cand.numel():
                # 空床数用主机侧记账（帧首住院数 - 本帧病房死亡 - 本帧病房康复）计算，
                # 免一次 .sum().item() 同步；漏掉康复会让同帧释放的床白白空着。
                free = max(0, sim.beds_total - (n_H - ward_dead - ward_rec))
                # 直接切片即"先到先得"：索引升序 = 自然序，不引入随机数
                take = cand[:free] if free > 0 else cand[:0]
                admitted = int(take.numel())
                if admitted:
                    sim.status[take] = 7
                    sim.isolated[take] = False
                    sim.untreated[take] = False
                rest = cand[admitted:]
                if rest.numel():
                    # 无床位者标记 untreated：仍可活动，但活动模块的 homebound 会强制其居家
                    sim.untreated[rest] = True

        # D.4 有空床时按等待时长补收（先到先得：symp_elapsed 越大 = 等待越久）
        if sim._n_untreated:
            waiting = sim.untreated & (sim.status == 3)
            widx = waiting.nonzero().squeeze(1)
        else:
            widx = _EMPTY_INDEX
        if widx.numel():
            # 空床要扣掉 D.3 刚收治的 admitted；free<=0 时连 argsort 都不跑
            free = sim.beds_total - (n_H - ward_dead - ward_rec + admitted)
            if free > 0:
                order = torch.argsort(sim.symp_elapsed[widx], descending=True)
                take = widx[order[:free]]
                sim.status[take] = 7
                sim.isolated[take] = False
                sim.untreated[take] = False

        # D.5 社会响应层：检测结果与追踪隔离到期
        #    门控用上一帧 F 段末尾写回的主机侧待办计数（默认 0，见 F 段 stat 布局）：
        #    test 用"计时中"、trace 用"计时中"，为 0 时整块跳过——否则每帧都要跑 kernel。
        if sim.IV_ENABLED:
            if sim._n_test_pending:
                _process_tests(sim, )
            if sim._n_trace_pending:
                _resolve_traces(sim, )

    # D-legacy.1 潜伏期计时/转症（旧路径：NATURAL_HISTORY_ENABLED 关闭时；与 NH 分支互斥）
    #   旧路径没有症状分流/隔离/检测/床位，只有 timer+max_incubation 与潜伏期自愈。
    if not CONFIG["NATURAL_HISTORY_ENABLED"]:
        incu_idx = incu_mask.nonzero().squeeze(1)
        if incu_idx.numel():
            sim.timer.add_(incu_mask)
            to_symp = sim.timer[incu_idx] >= sim.max_incubation[incu_idx]
            sim.status[incu_idx[to_symp]] = 3

            # 自愈判定（沿用旧版上尾判定 rand > 1-rate；rate=0 时永不发生）
            # RNG 契约：别改成 rand < rate——概率相同但用掉的随机数不同，轨迹会漂。
            self_recov = (~to_symp) & (
                torch.rand(incu_idx.numel(), device=device)
                > (1.0 - CONFIG["INCUBATION_SELF_RECOVERY"]))
            recovered_idx = incu_idx[self_recov]
            sim.status[recovered_idx] = 5

            # 康复时增强免疫力（index_add_ + 整幅 clamp_；病毒参数离开 E/I/A 后不可观测）
            # 不判空：空索引的 index_add_ 只是 no-op kernel，省一次 numel() 主机同步。
            sim.immunity_level.index_add_(
                0, recovered_idx,
                torch.full((recovered_idx.numel(),),
                           CONFIG["IMMUNITY_GAIN_PER_INFECTION"],
                           dtype=sim.immunity_level.dtype, device=device))
            sim.immunity_level.clamp_(0, CONFIG["MAX_IMMUNITY"])

    # D-legacy.2 社区结局（旧全局乘子路径；只用帧首就已是症状者，本帧新转者不抽签）
    #   旧口径下死亡/康复概率由病毒 mu/gamma × medical_pressure（2.5 的历史含义见 A 段）直接决定，
    #   不区分住院/无床位/年龄严重性——那些是 NH 路径才引入的机制。
    symp_mask = symp_prev
    # 病死和康复判定（帧首症状数即 counts_prev[3]，可直接当判空条件）
    # 0.05：旧路径把 virus_mu 折算成"帧死亡风险"的固定系数（NH 路径改用 COMMUNITY_DEATH_HAZARD）；
    #       r_p = gamma / pressure 同属旧口径，两个公式都只为复现旧档案数值，勿与 NH 常数互换。
    if (not CONFIG["NATURAL_HISTORY_ENABLED"]) and num_symptomatic:
        idx = symp_mask.nonzero().squeeze(1)
        d_p = sim.virus_mu[idx] * medical_pressure * 0.05
        r_p = sim.virus_gamma[idx] / medical_pressure

        # RNG 契约：整块只抽一次 rolls(num_symptomatic)，下尾判死、上尾判康复（两事件互斥）；
        #   拆成两次 rand 会改变随机流与"同帧既死又康"的口径，legacy 基线会失配。
        rolls = torch.rand(num_symptomatic, device=device)
        
        # 死亡判定（下尾，与康复共用 rolls，见上）
        is_dead = rolls < d_p
        dead_idx = idx[is_dead]
        sim.status[dead_idx] = 4
        # 同帧死亡者立即停止出行（避免瞬间死亡后仍标记为在途）
        sim.traveling[dead_idx] = False
        sim.trip_left[dead_idx] = 0
        # 与 NH 路径同款清理：死者不再占据隔离计数/计时
        sim.quarantined[dead_idx] = False
        sim.quarantine_timer[dead_idx] = -1.0
        
        # 康复判定（上尾 rolls > 1-r_p，且排除已死者）
        is_rec = (~is_dead) & (rolls > (1 - r_p))
        rec_idx = idx[is_rec]
        sim.status[rec_idx] = 5

        # 康复时增强免疫力（index_add_ + 整幅 clamp_；病毒参数离开 E/I/A 后不可观测）
        if rec_idx.numel():
            sim.immunity_level.index_add_(
                0, rec_idx,
                torch.full((rec_idx.numel(),),
                           CONFIG["IMMUNITY_GAIN_PER_INFECTION"],
                           dtype=sim.immunity_level.dtype, device=device))
            sim.immunity_level.clamp_(0, CONFIG["MAX_IMMUNITY"])

    # E. 传播（帧首无感染者时不可能产生新感染 → 整段跳过，省下十几个 kernel）
    # RNG 契约：本段随机数消耗 = pair 级一次 rand(pair 数) + 每个新感染者的变异/潜伏期抽样；
    #   两者都随疫情状态变化（不是常数），优化只能改"怎么算"，不能改抽取顺序与形状。
    num_new = 0
    # 再感染计数先留在 GPU 上，随 F 段统计一起单次拷回（省一次 .item() 同步）
    reinf_sum = torch.zeros((), dtype=torch.long, device=device)
    if n_active_inf:
        # 状态→具传染性 用 LUT 查表（一次 gather 取代 3–5 个比较/或 kernel）
        # 不变式：LUT 只点亮 2/3/6（H 住院者不参与社区传播），故下面三分支覆盖全部感染者。
        inf_mask = sim._inf_lut[sim.status]
        # 目标 = 本帧易感者 + 帧首已康复者；本帧新康复/自愈者不参与本帧传播。
        #   只认帧首快照 rec_prev：否则"本帧刚康复"者会被当成可再感染目标，免疫语义漂移。
        target_mask = (sim.status == 0) | rec_prev

        # 传染过程：先取一次索引（nonzero 是唯一需要主机同步的步骤），后续全部按索引 gather
        inf_indices = inf_mask.nonzero().squeeze(1)
        n_inf = inf_indices.numel()
        target_indices = target_mask.nonzero().squeeze(1)
        n_tar = target_indices.numel()
    else:
        n_inf = 0
        n_tar = 0
    if n_inf and n_tar:
        inf_pos = sim.pos.index_select(0, inf_indices)
        tar_pos = sim.pos.index_select(0, target_indices)

        # 接触对：auto 模式小规模走 dense（与旧版逐对语义一致），大规模走 GPU 空间网格
        #   性能：dense 是 (n_inf, n_tar) 广播距离矩阵，规模小时更省；grid 按空间桶只枚举近邻，
        #   避免 O(n_inf×n_tar) 的临时矩阵。CONTACT_MODE=auto 以 CONTACT_DENSE_LIMIT 乘积为界。
        mode = CONFIG["CONTACT_MODE"]
        if mode == "auto":
            mode = ("dense" if n_inf * n_tar <= int(CONFIG["CONTACT_DENSE_LIMIT"])
                    else "grid")
        if mode == "grid":
            pair_src, pair_tar = _contact_pairs_grid(sim, inf_indices, target_indices,
                                                     inf_pos, tar_pos)
        else:
            pair_src, pair_tar = _contact_pairs_dense(inf_pos, tar_pos)

        # 仅对实际接触对计算传播概率（稀疏化，避免 inf×tar 的临时矩阵）
        if pair_src.numel():
            # 0.15 = 封控/居家令期间的接触传播折减（与 LOCKDOWN_SHELTER 的出行改写共同作用）
            lockdown_mult = 0.15 if sim.lockdown_active else 1.0
            inf_status = sim.status.index_select(0, inf_indices)
            if CONFIG["NATURAL_HISTORY_ENABLED"]:
                # 时变传染力：E/I 按相对症状出现天数查症状曲线，A 按感染后天数查无症状曲线
                #   E 的 (inf_elapsed-max_incubation)/FPD 通常为负 = 症状曲线左尾（潜伏早期）
                #   隔离乘子对家庭内/外接触统一乘（简化，未按场所区分）
                inf_factor = torch.ones(n_inf, device=device)
                _e = (inf_status == 2)
                _rel = ((sim.inf_elapsed[inf_indices[_e]]
                         - sim.max_incubation[inf_indices[_e]]) / sim.FPD)
                inf_factor[_e] = _curve_eval(_rel, _curve_symp())
                _i = (inf_status == 3)
                inf_factor[_i] = _curve_eval(
                    sim.symp_elapsed[inf_indices[_i]] / sim.FPD, _curve_symp())
                _a = (inf_status == 6)
                inf_factor[_a] = _curve_eval(
                    sim.inf_elapsed[inf_indices[_a]] / sim.FPD, _curve_asym())
                inf_factor[sim.isolated[inf_indices]] *= CONFIG["ISOLATION_TRANSMISSION_MULT"]
                if sim.TRACING_ON:
                    inf_factor[sim.quarantined[inf_indices]] *= CONFIG["QUARANTINE_TRANSMISSION_MULT"]
                status_mult = inf_factor
            else:
                # 旧路径：只有 E 有潜伏期折减，I/A 一律 1.0（无时变曲线、无隔离乘子）
                status_mult = torch.where(
                    inf_status == 2,
                    CONFIG["INCUBATING_TRANSMISSION_MULT"], 1.0)

            # 1.5 次幂：病毒 beta→传染力的换算指数（CONTRIBUTING §3.4 点名的魔数）；
            #   beta 逐代变异，幂律放大高 beta 尾部，让"超级病毒"碾压均值。
            beta_factor = torch.pow(sim.virus_beta[inf_indices], 1.5)
            target_immunity = sim.immunity_level[target_indices]
            # 免疫抵抗 = (1-immunity)^3：IMMUNITY_EFFECTIVENESS=3 让高免疫者非线性地更难被感染
            immunity_resistance = torch.pow(1 - target_immunity, CONFIG["IMMUNITY_EFFECTIVENESS"])
            # 目标全局索引只取一次：后续易感性/休闲/同配判定都复用它（值不变，纯去重）
            tar_glob = target_indices[pair_tar]

            if CONFIG["NATURAL_HISTORY_ENABLED"]:
                # 目标年龄易感性（均值已归一化为 1，不改变总体 R0）
                if sim.BEHAVIOR_ON:
                    leisure_pair = ((sim.place[inf_indices[pair_src]] == PLACE_LEISURE)
                                    & (sim.place[tar_glob] == PLACE_LEISURE))
                    pair_contact_mult = torch.where(
                        leisure_pair, float(sim.behavior_contact_mult), 1.0)
                else:
                    pair_contact_mult = 1.0
                # 系数折叠：感染者侧 β^1.5·status_mult·基础概率·封控（HET 时再 ×virus_scale）
                # 先合成 n_inf 长向量，pair 级只做 4 次 gather 乘法：
                #   p_pair = coef[src] × 免疫抵抗[tar] × 易感[tar] × 行为乘子
                # 乘子物理含义：status_mult=病程时变传染力×隔离折减；virus_scale=个体超传播；
                #   immunity_resistance=目标免疫史；suscept_mult=年龄易感；pair_contact_mult=行为
                coef = beta_factor * status_mult * (CONFIG["TRANSMIT_BASE_PROB"] * lockdown_mult)
                if sim.HET_ON:
                    # v4 超传播：每个感染者的传染力乘子（对数正态，均值 1，clamp[0.05,20]）
                    coef = coef * sim.virus_scale[inf_indices]
                p_pair = (coef[pair_src]
                          * immunity_resistance[pair_tar]
                          * sim.suscept_mult[tar_glob]
                          * pair_contact_mult)
                if sim.MIX_ON:
                    # v4 混合结构：家庭接触乘子 × 粗年龄同配矩阵（E[φ]=1，总体 R0 期望不变）
                    # 用 src*4+tar 的一维索引取矩阵元素，避免一次 4×4 的二维 gather
                    src_glob = inf_indices[pair_src]
                    same_hh = sim.household_id[src_glob] == sim.household_id[tar_glob]
                    pair_extra = torch.where(same_hh, sim.hh_mult, 1.0)
                    pair_extra = pair_extra * sim.mix_flat[
                        sim.age_group[src_glob] * 4 + sim.age_group[tar_glob]]
                    p_pair = p_pair * pair_extra
            else:
                # 旧路径：同一套系数折叠，但没有年龄易感/行为/混合结构
                coef = beta_factor * status_mult * (CONFIG["TRANSMIT_BASE_PROB"] * lockdown_mult)
                if sim.HET_ON:
                    coef = coef * sim.virus_scale[inf_indices]
                p_pair = (coef[pair_src]
                          * immunity_resistance[pair_tar])
            # RNG 契约：每个接触对抽一次 rand(pair 数)，与 p_pair 逐元素对应（形状必须等于 pair 数）
            ok = torch.rand(pair_src.numel(), device=device) < p_pair
            ok_src = pair_src[ok]
            ok_tar = pair_tar[ok]

            # 每个目标最多感染一次；父本取"成功源中索引最小"，与旧 argmax 语义一致
            #   性能：scatter_reduce amax/amin 一次并行归约完成选源，避免按目标循环或主机侧判选
            src_choice = torch.full((n_tar,), n_inf, dtype=torch.long, device=device)
            if ok_src.numel():
                src_choice.scatter_reduce_(0, ok_tar, ok_src, reduce="amin")
                was_inf = src_choice < n_inf

                sources = inf_indices[src_choice[was_inf]]
                targets = target_indices[was_inf]

                # 记录再感染：按"是否曾被感染"判定，免疫消退后再感染仍计入
                is_reinfection = sim.ever_infected[targets]
                num_new = targets.numel()
                reinf_sum = is_reinfection.sum()

                if sim.HET_ON and sim.sigma_infectious > 0.0:
                    # v4：新感染者的传染力乘子重新抽取（对数正态，均值 1）
                    # RNG 契约：只有 v4 档案才抽；关闭时零额外随机数（docs/ARCHITECTURE.md §3）
                    sim.virus_scale[targets] = torch.exp(
                        torch.rand(num_new, device=device) * sim.sigma_infectious
                        - 0.5 * sim.sigma_infectious ** 2).clamp_(0.05, 20.0)
                if sim._secondary is not None:
                    # 诊断用：累计每个感染者产生的二代感染数（不参与模型逻辑）
                    sim._secondary.index_add_(
                        0, sources,
                        torch.ones(num_new, dtype=sim._secondary.dtype, device=device))

                # 新感染者统一置 E，并把所有逐人计时/标记复位。
                #   哨兵语义：iso_timer/test_timer/trace_timer 的 -1=未启动、>0=倒计时中、0=已到期；
                #   F 段的主机侧门控计数（_n_test_pending/_n_trace_pending）就按这三段语义读。
                sim.status[targets] = 2
                sim.ever_infected[targets] = True
                if CONFIG["NATURAL_HISTORY_ENABLED"]:
                    sim.inf_elapsed[targets] = 0.0
                    sim.symp_elapsed[targets] = 0.0
                    sim.a_timer[targets] = 0.0
                    sim.iso_timer[targets] = -1.0
                    sim.isolated[targets] = False
                    sim.admission_attempted[targets] = False
                    sim.untreated[targets] = False
                    sim.wants_test[targets] = False
                    sim.test_timer[targets] = -1.0
                    sim.tested[targets] = False
                    sim.trace_timer[targets] = -1.0

                # beta/gamma/mu 三组变异合并为一次 (3, num_new) 张量运算：
                # 原先三套"rand→clone→rand→掩码加→clamp→写回"共约 39 个 kernel，现约 17 个。
                # 等价性：mutate=(rand<sigma)、delta=(rand-0.5)*scale、边界 clamp 都对三行分别成立，
                #   行顺序固定 beta/gamma/mu 且与 vir_new[0/1/2] 写回一一对应——改行序就是改语义。
                # RNG 契约：形状 (3, num_new)；与旧版三组独立抽样的随机数消耗顺序不同，当前
                #   基线按此形状固化，再改会让 golden 轨迹失配（S1 已接受的等价替换，需整档校验）。
                _delta_rows, _bnd_lo, _bnd_hi = _mutation_consts()
                vir_src = torch.stack((sim.virus_beta[sources],
                                       sim.virus_gamma[sources],
                                       sim.virus_mu[sources]))
                mutate = (torch.rand((3, num_new), device=device)
                          < sim.virus_sigma[sources])
                delta = (torch.rand((3, num_new), device=device) - 0.5) * _delta_rows
                vir_new = vir_src + mutate * delta
                vir_new.clamp_(_bnd_lo, _bnd_hi)
                sim.virus_beta[targets] = vir_new[0]
                sim.virus_gamma[targets] = vir_new[1]
                sim.virus_mu[targets] = vir_new[2]

                # sigma 为常数遗传（当前设计不做变异），直接拷贝父代
                sim.virus_sigma[targets] = sim.virus_sigma[sources]
                sim.virus_gen[targets] = sim.virus_gen[sources] + 1

                # 潜伏期 N(INCUBATION_MU, INCUBATION_SD) 并截掉负/过短尾部；每人一次抽样（形状=num_new）
                sim.max_incubation[targets] = torch.normal(
                    CONFIG["INCUBATION_MU"], CONFIG["INCUBATION_SD"],
                    (num_new,), device=device).clamp(min=CONFIG["INCUBATION_MIN"])
                sim.timer[targets] = 0

    # F. 统计（在 GPU 上聚合、单次拷回；渲染 artist 更新按 render_every 节流）
    # 不变式：stat 固定 24 维，v 的下标在下面整块与标题/曲线里被复用，增删字段必须同步改解包：
    #   [0:8]  counts_g        status 0..7 计数（状态码见 MECHANISMS.md §1）
    #   [8:12] place 构成      home/work/school/leisure 中"存活且非在途"的人数
    #   [12:15] vir_mean       beta/mu/gen 在 E/I/A 上的均值（分母 n_inf）
    #   [15]   imm_mean        存活者免疫均值
    #   [16]   cnt_travel      在途人数
    #   [17]   untreated  [18] quarantined  [19] isolated（均只数存活者）
    #   [20]   _n_test_pending     test_timer>0 = 仍在队列中的检测票
    #   [21]   _n_trace_pending    trace_timer>0 = 追踪计时中（下一帧递减，减到 0 即展开）
    #   [22]   _n_quarantine_active quarantine_timer>0 = 密接隔离中
    #   [23]   reinf_sum       本帧再感染数
    #   末尾把 [20][21][22] 写回 sim._n_*，作为下一帧 D.5 的主机侧门控（默认 0 → 整块跳过）。
    counts_g = torch.bincount(sim.status, minlength=8)
    if sim.render_every > 0:
        # 抽样渲染：位置/颜色两次传输（与状态一一对应，无法并进标量批）
        if sim.render_idx is None:
            sim.scat.set_offsets(sim.pos.cpu().numpy())
            sim.scat.set_facecolor(COLOR_LUT[sim.status.to(torch.uint8).cpu().numpy()])
        else:
            # 性能：大人口下只画固定抽样点 RENDER_MAX_POINTS 个，画面结构稳定且每帧传输量恒定
            sim.scat.set_offsets(sim.pos[sim.render_idx].cpu().numpy())
            sim.scat.set_facecolor(
                COLOR_LUT[sim.status[sim.render_idx].to(torch.uint8).cpu().numpy()])

    # 平均值/派生计数先在 GPU 上算好，和 counts 拼成一个向量一次拷回。
    # 性能：不用布尔索引取子数组（arr[mask] 内部走 aten::nonzero，每帧多次隐式同步），
    #   统一用"掩码乘加"；零散标量也不逐个 stat[i]= 写（每次是一次 select_+copy_），
    #   而是 cat 成一个向量后只 .cpu() 一次。
    alive_mask = sim.status != 4
    nh_on = CONFIG["NATURAL_HISTORY_ENABLED"]
    # 不变式：感染口径 = 2/3/6（NH）或 2/3（legacy），与 _inf_lut 点亮的行一致
    infected_mask = sim._inf_lut[sim.status]
    if nh_on:
        n_inf = counts_g[2] + counts_g[3] + counts_g[6]
    else:
        n_inf = counts_g[2] + counts_g[3]
    still_m = alive_mask & (~sim.traveling)
    # clamp(min=1)：分母保 1，避免零感染/全死帧产生 0/0 的 NaN 污染渲染曲线
    n_alive = (sim.N - counts_g[4]).clamp(min=1)
    n_inf = n_inf.clamp(min=1)
    imm_mean = (sim.immunity_level * alive_mask).sum() / n_alive
    vir = torch.stack((sim.virus_beta, sim.virus_mu,
                       sim.virus_gen.to(sim.virus_beta.dtype)))
    vir_mean = (vir * infected_mask).sum(1) / n_inf
    # 活动构成 + 门控计数：一次 stack + 一次按行求和（行序即 stat[16..22] 的顺序）
    flags = torch.stack((
        alive_mask & sim.traveling,
        sim.untreated,
        sim.quarantined,
        alive_mask & sim.isolated,
        sim.test_timer > 0.0,
        sim.trace_timer > 0.0,
        sim.quarantine_timer > 0.0,
    )).sum(1)
    stat = torch.cat((
        # 在途者的 place 不代表"此刻身处何处"，用 still_m 做权重把这些行排除在构成之外
        counts_g.to(torch.float32),
        torch.bincount(sim.place, weights=still_m.to(torch.float32), minlength=4),
        vir_mean,
        imm_mean.reshape(1),
        flags.to(torch.float32),
        reinf_sum.to(torch.float32).reshape(1),
    ))
    v = stat.cpu().numpy()          # ← 本帧唯一一次标量同步
    counts = v[0:8]
    num_symp = int(v[3])
    if nh_on:
        # 口径：Inf = E+I+A+H；legacy 没有 H/A 语义，只数 E+I
        total_infected = int(v[2] + v[3] + v[6] + v[7])
        hospitalized = int(v[7])
        asymptomatic = int(v[6])
    else:
        total_infected = int(v[2] + v[3])
        hospitalized = 0
        asymptomatic = 0
    # v 已是 numpy：下面 int()/float() 都是纯主机操作，不再触发 GPU 同步
    cnt_home = int(v[8])
    cnt_work = int(v[9])
    cnt_school = int(v[10])
    cnt_leisure = int(v[11])
    avg_beta = float(v[12])
    avg_mu = float(v[13])
    avg_gen = float(v[14])
    avg_immunity = float(v[15])
    cnt_travel = int(v[16])
    untreated_count = int(v[17])
    quarantined_count = int(v[18])
    isolated_count = int(v[19])
    # 三个门控计数只在 v3/NH 下有值；其余档案恒 0，D.5 因此整块跳过（不跑 kernel、不抽随机数）
    sim._n_test_pending = int(v[20])
    sim._n_trace_pending = int(v[21])
    sim._n_quarantine_active = int(v[22])
    reinfection_count = int(v[23])
    if nh_on:
        # 医疗崩溃（NH 口径）：有未收治者，或住院占用达到床位上限；标题 Med/颜色都读它
        is_collapsed = (untreated_count > 0) or (sim.beds_total > 0 and hospitalized >= sim.beds_total)

    # 逐帧记录传播/再感染数：虽放在 strain_stats 里，实际是与 strain 历史等宽的逐帧序列
    sim.strain_stats['new_infections'].append(num_new)
    sim.strain_stats['reinfections'].append(reinfection_count)
    # 帧末写回：下一帧 A 段/activity 直接读这些主机侧副本，避免再次 bincount/掩码求和。
    #   注意 counts_prev 是"本帧结束后"的状态计数，不是帧首——A/B 段拿它当帧首快照用。
    sim.counts_prev = counts
    sim._n_dead = int(counts[4])
    sim._n_untreated = untreated_count
    # 受限人数 = untreated(等床) + quarantined(密接隔离) + isolated(自然隔离) + hospitalized，
    #   与 activity 的 homebound 判定同集合；为 0 时 mobility 整块跳过 homebound 重算
    sim._n_restricted = untreated_count + quarantined_count + isolated_count + hospitalized

    # virus 曲线窗口 500 帧（与下面 set_data 的 [-500:] 对应），入列的都是原值
    sim.virus_history['beta'].append(avg_beta)
    sim.virus_history['mu'].append(avg_mu)
    sim.virus_history['gen'].append(avg_gen)
    sim.virus_history['immunity'].append(avg_immunity)

    # 再感染率 = 最近 50 帧的 "再感染数 / 新感染数"；
    #   只显示最后 500 点，且只对末尾 500+49+2 个样本做前缀和——长跑（上万帧）时
    #   避免每帧对全历史做 O(T) 的 cumsum。
    #   魔数：500=显示窗口；49=比率窗口 50 帧 - 1（denom 取 [idx-49, idx] 共 50 个样本）；
    #         2=给 cumsum 首元素 0 哨兵和索引上界留的余量；100.0=百分数换算。
    _W = 500
    _new_hist = sim.strain_stats['new_infections']
    _rei_hist = sim.strain_stats['reinfections']
    _start = max(0, len(_new_hist) - (_W + 49 + 2))
    new_counts = np.asarray(_new_hist[_start:], dtype=np.float64)
    reinf_counts = np.asarray(_rei_hist[_start:], dtype=np.float64)
    cum_new = np.concatenate(([0.0], np.cumsum(new_counts)))
    cum_rei = np.concatenate(([0.0], np.cumsum(reinf_counts)))
    T = new_counts.size
    idx = np.arange(max(0, T - _W), T)
    lo = np.maximum(0, idx - 49)
    denom = cum_new[idx + 1] - cum_new[lo]
    numer = cum_rei[idx + 1] - cum_rei[lo]
    # where 掩掉 denom==0（窗口内无新感染）并用 out=0：避免 0/0 的 NaN 进渲染曲线
    reinf_line = np.divide(numer * 100.0, denom,
                           out=np.zeros_like(denom, dtype=np.float64), where=denom > 0)
    recent_reinf = float(reinf_line[-1]) if reinf_line.size else 0.0

    clock_hh = int(clock_min // 60)
    clock_mm = int(clock_min % 60)

    # 更新标题（三行：疫情 / 病毒与封控 / 行为时钟与活动构成）。
    # harness 契约：tools/headless_check.py 的 parse_title 用正则抓 Inf/Symp/Med/Lock/t=/Day/
    #   Hosp/Unmet/Reinf 这些字段，改文案前先读它；legacy 分支没有 Hosp/Unmet 字段。
    iv_line = ""
    if sim.IV_ENABLED:
        # 只在 v3 开启时追加 Q/Test/Rep/mult 段，关闭时标题保持旧三行结构
        iv_line = (f" | Q {quarantined_count} Test {sim.tests_total} Rep {sim.reported_total}"
                   f" | mult {sim.behavior_contact_mult:.2f}")
    if CONFIG["NATURAL_HISTORY_ENABLED"]:
        title_text = (f"Inf: {total_infected} | Symp: {num_symp} | "
                      f"Med: {'COLLAPSE' if is_collapsed else 'OK'} | "
                      f"Hosp: {hospitalized}/{sim.beds_total} | Unmet: {untreated_count}\n"
                      f"Immunity: {avg_immunity:.3f} | Reinf: {recent_reinf:.1f}% | "
                      f"Lock: {'ON' if sim.lockdown_active else 'OFF'} | t={frame}\n"
                      f"Day {day_index + 1} {clock_hh:02d}:{clock_mm:02d} | "
                      f"home {cnt_home} work {cnt_work} school {cnt_school} "
                      f"leisure {cnt_leisure} travel {cnt_travel} | "
                      f"E {int(counts[2])} A {asymptomatic} I {num_symp} H {hospitalized}"
                      f"{iv_line}")
    else:
        title_text = (f"Inf: {total_infected} | Symp: {num_symp} | "
                      f"Med: {'COLLAPSE' if is_collapsed else 'OK'}\n"
                      f"Immunity: {avg_immunity:.3f} | Reinf: {recent_reinf:.1f}% | "
                      f"Lock: {'ON' if sim.lockdown_active else 'OFF'} | t={frame}\n"
                      f"Day {day_index + 1} {clock_hh:02d}:{clock_mm:02d} | "
                      f"home {cnt_home} work {cnt_work} school {cnt_school} "
                      f"leisure {cnt_leisure} travel {cnt_travel}")
    sim.title_obj.set_text(title_text)
    sim.title_obj.set_color('red' if is_collapsed else 'green')

    # 更新人口状态历史（只显示最近 500 点，render.py 把 x 轴固定为 0..500）
    #   不变式：history 是 8 条 list（status 0..7），全历史保留，这里只切窗口给线对象
    for s in range(8):
        sim.history[s].append(int(counts[s]))
        vals = sim.history[s][-500:]
        sim.lines_pop[s].set_data(range(len(vals)), vals)

    # 更新病毒进化历史（滚动 500 点窗口）
    vals = sim.virus_history['beta'][-500:]
    sim.line_beta.set_data(range(len(vals)), vals)
    vals = sim.virus_history['mu'][-500:]
    # mu 量级 0.001–0.15，×10 只为与 beta/gen 共用 0..1.6 的纵轴可读（纯显示变换）
    sim.line_mu.set_data(range(len(vals)), [v * 10 for v in vals])
    vals = sim.virus_history['immunity'][-500:]
    sim.line_immunity.set_data(range(len(vals)), vals)
    vals = sim.virus_history['gen'][-500:]
    sim.line_gen.set_data(range(len(vals)), vals)

    # 再感染率曲线（纵轴固定 0..100，不随代数轴范围裁剪）
    sim.line_reinf.set_data(range(len(reinf_line)), reinf_line)

    # G. 显存管理：大规模人口下接触对临时张量形状逐帧变化，避免 CUDA 缓存碎片累积。
    #   性能：每 50 帧才检查一次（frequency 高时 empty_cache 本身有代价），
    #   且只有"保留量-已分配量 > 2 GiB"（2*2**30）才真正回收，避免无谓抖动。
    if device.type == "cuda" and frame % 50 == 0:
        if torch.cuda.memory_reserved() - torch.cuda.memory_allocated() > 2 * 2 ** 30:
            torch.cuda.empty_cache()

    # harness 契约：返回的 artist 列表供 FuncAnimation blit；顺序/数量固定为
    #   scat + title + 8 条 population 线 + 5 条病毒/再感染线 = 15 个（headless_check 断言 N_ARTISTS=15）
    return [sim.scat, sim.title_obj] + sim.lines_pop + [sim.line_beta, sim.line_mu, sim.line_immunity, sim.line_gen, sim.line_reinf]
