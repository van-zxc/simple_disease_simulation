"""Schema-v3 社会响应层：检测、追踪、密接隔离与行为反馈。

负责：日界事件（发病/报告归档、行为乘子、当日检测预算）、检测结果与隔离排队、
追踪到期后的同住者/场所接触者隔离、密接隔离倒计时；不负责病程转移本身与几何枚举。

关键入口：_daily_step/_advance_quarantine 在 A0 日界段、_process_tests/_resolve_traces
在 D.5 段调用，调用点由 sim.IV_ENABLED 门控，各入口再按子开关早退。

不变式（toy/v2 回归契约）：开关全关时**不消耗任何随机数**——每条 rand/binomial 都写在
对应子开关分支内，同 seed 下轨迹与引入 v3 前逐位一致。子开关 = IV_ENABLED 总开关 ×
CONFIG 的 TESTING/TRACING/BEHAVIOR/REPORTING_ENABLED（在 state.py 一次性合成，本模块
只读 sim.*_ON）。
"""

import math

import numpy as np
import torch

from .config import CONFIG, device

# 空索引常量：所有"无结果"分支返回同一个空张量，避免每次新建
_EMPTY_INDEX = torch.empty(0, dtype=torch.long, device=device)


def _expand_members(key_idx, case_keys):
    """按 key 分组展开成员：返回 (agent_idx, case_pos) 两个等长张量。

    对有效个体排序后，用 searchsorted 求每个 case_key 在排序数组中的区间
    [starts, ends)，再 repeat_interleave 展开——复杂度 O((人+病例)log人 + 命中数)，
    避免 O(病例数×人口) 的布尔矩阵。key<0 的个体不进入任何分组（无工作者/无学校者）。
    case_pos 是 case_keys 的下标，调用方用它回指病例（例如排除"自己接触自己"）。
    """
    empty = _EMPTY_INDEX
    if case_keys.numel() == 0:
        return empty, empty
    # 只保留有该 key 的个体；key<0 表示"无此场所/无家"
    agents = (key_idx >= 0).nonzero().squeeze(1)
    if not agents.numel():
        return empty, empty
    order = torch.argsort(key_idx[agents])
    sorted_agents = agents[order]
    sorted_keys = key_idx[sorted_agents]
    starts = torch.searchsorted(sorted_keys, case_keys)
    ends = torch.searchsorted(sorted_keys, case_keys, right=True)
    counts = (ends - starts).clamp(min=0)
    total = int(counts.sum().item())
    if total <= 0:
        return empty, empty
    # 展开成"命中流"：case_pos 给出每个命中属于哪个病例，offsets 是段内偏移
    case_pos = torch.repeat_interleave(
        torch.arange(case_keys.numel(), device=device), counts)
    offsets = torch.arange(total, device=device) - torch.repeat_interleave(
        torch.cumsum(counts, 0) - counts, counts)
    flat = starts[case_pos] + offsets
    return sorted_agents[flat], case_pos


def _daily_step(sim, day_index):
    """日界事件（幂等：同一 day_index 重复调用直接返回）：归档 → 行为乘子 → 检测排程。

    顺序即语义。RNG 契约：唯一随机数是一次 binomial（仅在 REPORTING_ON、当天有发病、
    REPORTING_FRACTION>0 时抽取）；行为反馈与检测排程都是确定性计算，检测的灵敏度
    抽签发生在 D.5 段的 _process_tests。
    """
    if day_index == sim._last_stat_day:
        return
    if sim._last_stat_day >= 0:
        # 归档上一日发病数：onset_today 在 F 段按日累加，这里搬进逐日列表
        sim.onset_by_day.append(int(sim.onset_today))
        sim.onset_today = 0
        if sim.REPORTING_ON:
            # 报告延迟：取 delay_days 天前那天的发病数，每病例独立以 REPORTING_FRACTION
            # 报告 → 一次 binomial 直接抽总数（不必建 n 长的 rand 张量再求和）
            delay_days = int(round(CONFIG["REPORTING_DELAY_FRAMES"] / sim.FPD))
            src_day = sim._last_stat_day - delay_days
            cnt = 0
            if 0 <= src_day < len(sim.onset_by_day):
                n = int(sim.onset_by_day[src_day])
                if n > 0 and CONFIG["REPORTING_FRACTION"] > 0:
                    cnt = int(torch.binomial(
                        torch.tensor(float(n), dtype=torch.float32, device=device),
                        torch.tensor(float(CONFIG["REPORTING_FRACTION"]),
                                     dtype=torch.float32, device=device)).item())
            sim.reported_today = cnt
            sim.reported_total += cnt
            sim.report_events.append(int(cnt))
    sim._last_stat_day = day_index

    # 行为反馈：对"延迟后的 7 天发病/报告"做风险感知，并叠加疲劳
    if sim.BEHAVIOR_ON:
        delay_days = int(round(CONFIG["BEHAVIOR_RESPONSE_DELAY_FRAMES"] / sim.FPD))
        end = day_index - delay_days
        if end > 0:
            lo = max(0, end - 7)              # 7 天滑动窗口
            # REPORTING_ON 时用报告数（个体能"看到"的是报告），否则退回发病数
            if sim.REPORTING_ON:
                window = sum(sim.report_events[lo:end])
            else:
                window = sum(sim.onset_by_day[lo:end])
            # 每 10 万人日发病率（分母固定用总人口 N，含已死者）
            incidence = window / max(1, sim.N) * 100000.0
            # logistic 风险感知：incidence 越过阈值后趋近 1；scale 是斜率软度（1e-9 防 0 除）
            scale = max(1e-9, float(CONFIG["BEHAVIOR_SCALE_PER_100K"]))
            risk = 1.0 / (1.0 + math.exp(
                -(incidence - CONFIG["BEHAVIOR_THRESHOLD_PER_100K"]) / scale))
            # 疲劳：风险感知随疫情持续衰减，避免行为改变永久生效
            fatigue = 1.0 / (1.0 + CONFIG["BEHAVIOR_FATIGUE_STRENGTH"]
                             * max(0, day_index - delay_days)
                             / max(1e-9, CONFIG["BEHAVIOR_FATIGUE_DAYS"]))
            mult = 1.0 - CONFIG["BEHAVIOR_RESPONSE_STRENGTH"] * risk * fatigue
            # 不变式：乘子夹在 [BEHAVIOR_MIN_CONTACT_MULT, 1]，activity 侧只会降低外出概率
            mult = max(float(CONFIG["BEHAVIOR_MIN_CONTACT_MULT"]), min(1.0, mult))
            sim.behavior_contact_mult = float(mult)
        else:
            sim.behavior_contact_mult = 1.0

    # 当日检测预算：按症状持续时长降序（更久者优先）；预算由档案给出，不跨日累积
    sim.tests_today = 0
    if sim.TESTING_ON and sim.test_capacity_day > 0:
        eligible = ((sim.status == 3) & sim.wants_test & (~sim.tested) & (~sim.isolated)
                    & (sim.test_timer < 0))
        n_elig = int(eligible.sum().item())      # 一次同步同时当判空与名额上限
        if n_elig:
            take_n = min(sim.test_capacity_day, n_elig)
            idx = eligible.nonzero().squeeze(1)
            # 降序排 symp_elapsed：症状越久越优先（同分不保证稳定，只影响同日先后）
            order = torch.argsort(sim.symp_elapsed[idx], descending=True)
            take = idx[order[:take_n]]
            sim.test_timer[take] = max(1.0, float(CONFIG["TEST_DELAY_FRAMES"]))
            sim.tests_today = int(take.numel())
            sim.tests_total += sim.tests_today


def _process_tests(sim, ):
    """检测结果（D.5 段，仅 TESTING_ON）：计时到期 → 灵敏度 → 阳性隔离/触发追踪。

    RNG 契约（顺序固定）：先对全部到期者抽一次 TEST_SENSITIVITY，再对阳性者抽隔离
    依从性（HET_ON 时乘个体 compliance），最后对阳性者抽 TRACE_PROBABILITY；开关关闭
    时整段不抽随机数。
    """
    if not sim.TESTING_ON:
        return
    # 用"原地布尔式减 1"取代 running/timer[running]-=1 的掩码读改写（少 2 个 kernel），
    # 直接用 nonzero 取到期索引（原来两次 bool(any()) 会各触发一次主机同步）
    sim.test_timer.add_(sim.test_timer > 0, alpha=-1)
    didx = (sim.test_timer == 0).nonzero().squeeze(1)
    if not didx.numel():
        return
    # 到期者统一标记"已检测"并清计时（-1 = 不在计时中，也是再次排程的准入条件）
    sim.test_timer[didx] = -1.0
    sim.tested[didx] = True
    # 灵敏度抽签：整幅 rand，形状 = 到期人数
    positive = torch.rand(didx.numel(), device=device) < CONFIG["TEST_SENSITIVITY"]
    pos_idx = didx[positive]
    if pos_idx.numel() == 0:
        return
    if sim.HET_ON:
        # v4 依从性异质性：是否遵守"阳性后隔离"（四处 compliance 乘子之一）
        adhere = (torch.rand(pos_idx.numel(), device=device)
                  < CONFIG["TEST_ISOLATION_ADHERENCE"] * sim.compliance[pos_idx])
    else:
        adhere = (torch.rand(pos_idx.numel(), device=device)
                  < CONFIG["TEST_ISOLATION_ADHERENCE"])
    iso_idx = pos_idx[adhere]
    if iso_idx.numel():
        iso_idx = iso_idx[sim.iso_timer[iso_idx] < 0]  # 自然隔离已排队则保留更早者
        if iso_idx.numel():
            # 与自然隔离共用 iso_timer，不覆盖已排队的更早到期时间
            sim.iso_timer[iso_idx] = max(1.0, float(CONFIG["TEST_ISOLATION_DELAY_FRAMES"]))
    if sim.TRACING_ON:
        # 追踪触发：只对阳性者抽，命中者进入追踪计时（到期后由 _resolve_traces 展开密接）
        trig = torch.rand(pos_idx.numel(), device=device) < CONFIG["TRACE_PROBABILITY"]
        trg = pos_idx[trig]
        if trg.numel():
            sim.trace_timer[trg] = max(1.0, float(CONFIG["TRACE_DELAY_FRAMES"]))


def _resolve_traces(sim, ):
    """追踪到期（D.5 段，仅 TRACING_ON）：隔离同住者与部分工作/学校场所接触者。

    两路展开：家（TRACE_HOUSEHOLD 开时取同 household_id 全体）与工作/学校（同 venue
    下标全体），排除病例本人与死者。隔离概率：家庭 = QUARANTINE_ADHERENCE，场所 =
    TRACE_VENUE_FRACTION × QUARANTINE_ADHERENCE；HET_ON 再乘个体 compliance。
    """
    if not sim.TRACING_ON:
        return
    # 追踪计时递减（与 test_timer/_process_tests 对称）：>0 的计时器每帧减 1，
    # 减到 0 的当帧展开。门控 _n_trace_pending 统计 >0，因此设置当帧不递减，
    # 下一帧起才递减 —— 延迟 = TRACE_DELAY_FRAMES 帧后展开。
    # 【2026-11 审计修复】此前全代码没有递减：trace_timer 从 >=1 起、永远到不了 0，
    # 且 F 段门控读 trace_timer==0（恒 0）→ 本函数从不被调用，追踪/密接隔离是死机制。
    sim.trace_timer.add_(sim.trace_timer > 0, alpha=-1)
    case_idx = (sim.trace_timer == 0).nonzero().squeeze(1)
    if not case_idx.numel():
        return
    # 到期病例清追踪计时（-1），保证每个病例只展开一次
    sim.trace_timer[case_idx] = -1.0

    # 收集 (成员, 病例位置, kind) 三路；kind 0 = 家庭、1 = 场所（决定依从概率）
    pieces = []
    if CONFIG["TRACE_HOUSEHOLD"]:
        m, cp = _expand_members(sim.household_id, sim.household_id[case_idx])
        if m.numel():
            pieces.append((m, cp, 0))
    for key in (sim.work_venue_idx, sim.school_venue_idx):
        m, cp = _expand_members(key, key[case_idx])
        if m.numel():
            pieces.append((m, cp, 1))
    if not pieces:
        return
    agents = torch.cat([p[0] for p in pieces])
    case_of = torch.cat([p[1] for p in pieces])
    kind = torch.cat([torch.full((p[0].numel(),), p[2], dtype=torch.long, device=device)
                      for p in pieces])
    src = case_idx[case_of]
    # 排除"自己接触自己"与死者（死者不再需要隔离）
    keep = (agents != src) & (sim.status[agents] != 4)
    keep_idx = keep.nonzero().squeeze(1)
    if not keep_idx.numel():
        return
    agents = agents.index_select(0, keep_idx)
    kind = kind.index_select(0, keep_idx)
    # 家庭接触按依从率、场所接触再乘场所比例（标量 where，避免 ones_like/full_like 两个整幅张量）
    # 注意 p_base 是"被隔离"的概率，不是"被追踪到"的概率
    adm = float(CONFIG["QUARANTINE_ADHERENCE"])
    p_venue = float(CONFIG["TRACE_VENUE_FRACTION"]) * adm
    p_base = torch.where(kind == 0, adm, p_venue)
    if sim.HET_ON:
        # v4 依从性异质性：个体是否遵守密接隔离（四处 compliance 乘子之一）
        p_base = p_base * sim.compliance[agents]
    # 一次整幅 rand 抽隔离决定（RNG 契约：形状 = 候选人数）
    take_idx = (torch.rand(agents.numel(), device=device)
                < p_base).nonzero().squeeze(1)
    if not take_idx.numel():
        return
    cand = agents.index_select(0, take_idx)
    # 仅对新隔离者生效：已密接隔离/住院者跳过（索引化，替代整幅 N 布尔与三次 masked 写）
    ok_idx = ((~sim.quarantined[cand]) & (sim.status[cand] != 7)).nonzero().squeeze(1)
    cand = cand.index_select(0, ok_idx)
    n_new = cand.numel()
    if n_new:
        sim.quarantined[cand] = True
        sim.quarantine_timer[cand] = max(1.0, float(CONFIG["QUARANTINE_DAYS_FRAMES"]))
        sim.quarantine_total += n_new


def _advance_quarantine(sim, ):
    """密接隔离倒计时（A0 段，仅 TRACING_ON；调用点由 _n_quarantine_active 门控）。

    不变式：只有 quarantined 为真者计时递减；到期清标志并把计时置 -1（不在计时中）。
    """
    if not sim.TRACING_ON:
        return
    active = sim.quarantined & (sim.quarantine_timer > 0)
    sim.quarantine_timer.add_(active, alpha=-1)
    expired = sim.quarantined & (sim.quarantine_timer <= 0)
    sim.quarantined.masked_fill_(expired, False)
    sim.quarantine_timer.masked_fill_(expired, -1.0)
