"""接触对枚举：给定感染者/目标位置，枚举半径 INFECT_RADIUS 内的 (src, tar) 对。

负责把"空间邻近"变成候选接触对（返回相对 inf_pos/tar_pos 的局部索引）；不负责概率
判定与传播（simulation.py E 段），也不读状态机、不消耗随机数。入口：
``_contact_pairs_dense``（cdist 全矩阵，基准实现，仅小规模/校验用）与
``_contact_pairs_grid``（空间网格 + 9 邻格，O(n_tar + 候选对)，大人口默认路径）。

不变式（tests/test_contacts.py 守着）：两条实现给出的 (src, tar) **集合**必须完全
相同且 grid 无重复对；但 dense 是 nonzero 行主序、grid 是"src 升序 + 格序"，配对
顺序不同 → 喂给 ``torch.rand`` 的随机数配对不同，切换实现不保证逐位一致（10k 默认
走 dense，历史回归以 dense 为准）。
"""

import numpy as np
import torch

from .config import CONFIG, device

# 性能：空结果复用同一个模块级张量，避免每帧新建（小分配 + kernel 也在 ~10µs 量级）
_EMPTY_INDEX = torch.empty(0, dtype=torch.long, device=device)
# 旧版 9 邻域 (dx, dy) 偏移表（dx 外层、dy 内层）；grid 改用下面的扁平步长版本后已无引用，
# 仅作历史对照保留——若确认外部脚本也不 import，可以整块删掉
_NEIGHBOUR_OFFSETS = torch.tensor(
    [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)],
    dtype=torch.long, device=device)
# 带 1 格外边距的扁平步长缓存：offset = dx + dy * gxp（dy 外层 = 行主序，与上面的
# _NEIGHBOUR_OFFSETS 顺序不同）；键为 gxp，世界尺寸在仿真期内固定，故每进程只建一次
_STRIDES_CACHE: dict[int, torch.Tensor] = {}


def _padded_strides(gxp: int) -> torch.Tensor:
    strides = _STRIDES_CACHE.get(gxp)
    if strides is None:
        strides = torch.tensor([dx + dy * gxp for dy in (-1, 0, 1) for dx in (-1, 0, 1)],
                               dtype=torch.long, device=device)
        _STRIDES_CACHE[gxp] = strides
    return strides


def _contact_pairs_dense(inf_pos, tar_pos):
    """基准实现：cdist 全矩阵阈值（O(n_inf*n_tar)，仅小规模校验用）。

    返回 (src, tar) 为 nonzero 的行主序：src 升序，同一 src 内 tar 升序——这是"配对
    顺序"的基准口径；``del dists`` 及时释放全矩阵，避免大人口下多占一份显存。
    """
    dists = torch.cdist(inf_pos, tar_pos)
    contacts = dists < CONFIG["INFECT_RADIUS"]
    del dists
    return contacts.nonzero(as_tuple=True)


def _contact_pairs_grid(sim, inf_indices, target_indices, inf_pos, tar_pos):
    """GPU 空间网格邻居枚举：返回半径内所有"感染者-易感目标"对的**局部索引**。

    返回的 (src, tar) 是 inf_pos / tar_pos 的行号（调用方再用 inf_indices[src] /
    target_indices[tar] 换成全局下标）。只对目标建网格（感染者/死者不进候选），感染者
    按 GRID_INF_CHUNK 分块枚举自身格 + 8 邻格，距离过滤再按 GRID_CHUNK 分块以限定显存。

    不变式：与 ``_contact_pairs_dense`` 的 (src, tar) 集合完全一致且无重复对
    （tests/test_contacts.py）；``inf_indices``/``target_indices`` 目前只用来取长度，
    位置一律由 inf_pos/tar_pos 给出（保留参数是为与 dense 接口对称）。
    """
    # 格边长 = 感染半径：一个目标只会被"自身格 + 8 邻格"内的感染者命中；gx/gy 上取整
    # 保证最后一列/行也有格；gxp/gyp 多出的 2 是左右/上下各 1 格 padding
    r = float(CONFIG["INFECT_RADIUS"])
    inv_r = 1.0 / r
    gx = max(1, int(np.ceil(sim.W / r)))
    gy = max(1, int(np.ceil(sim.H / r)))
    gxp = gx + 2
    gyp = gy + 2
    n_cells = gxp * gyp
    n_inf = int(inf_indices.numel())
    n_tar = int(target_indices.numel())
    empty = _EMPTY_INDEX
    if n_inf == 0 or n_tar == 0:
        return empty, empty

    # 目标网格：格坐标 clamp 到 [0, gx-1]/[0, gy-1] 后 +1 进入 padding 空间——边界目标的
    # 越界邻格会落到计数恒 0 的 padding 格，枚举时不必再判越界（省 4 比较 + 1 cast）
    tcell_xy = (tar_pos * inv_r).long()
    tcell = ((tcell_xy[:, 0].clamp_(0, gx - 1) + 1)
             + (tcell_xy[:, 1].clamp_(0, gy - 1) + 1) * gxp)
    # CSR：order_t 是按格排序后的目标下标，cell_start 是每格在 order_t 里的起始位置
    order_t = torch.argsort(tcell)
    cell_counts = torch.bincount(tcell, minlength=n_cells)
    cell_start = torch.cumsum(cell_counts, 0) - cell_counts

    # 感染者格：同样的 padding 坐标；strides 是 9 个邻格相对自身格的扁平偏移（缓存）
    inf_cell_xy = (inf_pos * inv_r).long()
    inf_cell = ((inf_cell_xy[:, 0].clamp_(0, gx - 1) + 1)
                + (inf_cell_xy[:, 1].clamp_(0, gy - 1) + 1) * gxp)
    strides = _padded_strides(gxp)                              # (9,)

    src_list, tar_list = [], []
    # 分块参数（CONFIG）：感染者每块上限 GRID_INF_CHUNK 人，距离过滤每块上限
    # GRID_CHUNK 个候选对——两者一起给中间张量的显存峰值封顶
    inf_chunk = max(1, int(CONFIG["GRID_INF_CHUNK"]))
    dist_chunk = max(1, int(CONFIG["GRID_CHUNK"]))
    for c0 in range(0, n_inf, inf_chunk):
        c1 = min(c0 + inf_chunk, n_inf)
        nc = c1 - c0
        # 9 个邻格一次性展开成 (nc*9)，避免每个偏移单独启动 kernel
        ncell = inf_cell[c0:c1].unsqueeze(1) + strides.unsqueeze(0)
        cnt = cell_counts[ncell.reshape(-1)]
        rows_rep = torch.arange(c0, c1, device=device).unsqueeze(1).expand(nc, 9).reshape(-1)
        # 把"每个(感染者, 邻格)的候选数 cnt"展开成候选流：rows 是感染者行号；flat 是该
        # 候选在 order_t 中的位置（= 格起点 cell_start + 段内偏移 local）
        rows = torch.repeat_interleave(rows_rep, cnt)
        cum = torch.cumsum(cnt, 0)
        starts = torch.repeat_interleave(cum - cnt, cnt)
        local = torch.arange(starts.numel(), device=device) - starts
        flat = torch.repeat_interleave(cell_start[ncell.reshape(-1)], cnt) + local
        tar_local = order_t[flat]
        # 性能：按 dist_chunk 再分块过滤，中间张量峰值有界；用平方距离省一次 sqrt，
        # 与 dense 的 dist < r 数学等价（只有距离恰好等于半径时才有浮点边界差）
        for d0 in range(0, tar_local.numel(), dist_chunk):
            s = rows[d0:d0 + dist_chunk]
            t = tar_local[d0:d0 + dist_chunk]
            d2 = ((inf_pos[s] - tar_pos[t]) ** 2).sum(dim=1)
            keep = d2 < r * r
            src_list.append(s[keep])
            tar_list.append(t[keep])
    if not src_list:
        return empty, empty
    # 各分块按"感染者块序 + 格序"拼接；返回局部索引（见 docstring 的契约说明）
    return torch.cat(src_list), torch.cat(tar_list)
