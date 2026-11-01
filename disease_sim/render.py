"""Matplotlib 图形/坐标轴搭建与封控按钮（纯表现层，不参与模型语义）。

负责：3×3 网格布局（主图 + 3 个曲线子图）、15 个动态 artist、静态场所散点、封控按钮。
不负责：每帧的 artist 数据更新（在 simulation F 段）、模型状态与统计口径。

调用方契约：FuncAnimation 按固定顺序取这 15 个 artist（2 + 8 + 5，顺序见
simulation 的返回值），并要求每个都有 .axes；静态场所散点不在其中，不能挪进返回值。
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.widgets import Button

from .config import CONFIG, device


# 状态 → 颜色：下标即 status 码 0..7（顺序必须与 state 里的状态定义一致）
COLOR_MAP = ["#B0B0B0", '#00BFFF', '#FFD700', '#FF4500', '#000000', '#32CD32',
             '#FF8C00', '#8A2BE2']
# 预转 RGBA 的 (8,4) 数组：每帧整块喂给 mpl，省掉 Python 侧逐点颜色转换
COLOR_LUT = np.array([to_rgba(c) for c in COLOR_MAP])


def build_render(sim) -> None:
    """搭建图形、坐标轴、artist 与封控按钮；只在装配阶段调用一次。

    调用方契约：返回给 FuncAnimation 的 artist 顺序为 [scat, title_obj] + lines_pop(8)
    + [line_beta, line_mu, line_immunity, line_gen, line_reinf]（共 15 个，按数量校验的调用方
    校验数量与 .axes）；这里增删 artist 必须同步改 simulation 的返回列表。
    """
    sim.fig = plt.figure(figsize=(18, 10))

    # 主图：世界坐标 1:1（aspect equal），场所与散点都按世界单位绘制
    sim.ax_main = plt.subplot2grid((3, 3), (0, 0), rowspan=3, colspan=2)
    sim.ax_main.set_xlim(0, sim.W)
    sim.ax_main.set_ylim(0, sim.H)
    sim.ax_main.set_aspect('equal')
    sim.ax_main.set_title('Virus Evolution CA with Immunity System', fontsize=14, fontweight='bold')

    # 场所标记：静态 artist，只绘制一次（不参与 blit 重绘，也不在 update 返回值里）
    # marker 用形状区分四类场所，s=64 保证小图上也看得清
    sim._venue_styles = [
        ("Workplace", '#4C72B0', 's', sim.work_venue_np),
        ("School", '#8172B2', '^', sim.school_venue_np),
        ("Shop", '#DD8452', 'o', sim.shop_venue_np),
        ("Park", '#55A868', 'P', sim.park_venue_np),
    ]
    for _vlabel, _vcolor, _vmarker, _varr in sim._venue_styles:
        sim.ax_main.scatter(_varr[:, 0], _varr[:, 1], s=64, marker=_vmarker, c=_vcolor,
                        alpha=0.6, edgecolors='white', linewidths=0.5,
                        zorder=0.4, label=_vlabel)
    sim.ax_main.legend(loc='lower left', fontsize=7, framealpha=0.45)

    # 小方点：Agg 绘制每个 marker 的顶点数约为圆形的 1/6，10k 点交互绘制明显更快
    # （s=2 时肉眼几乎无差别；15 artist / 4 axes 的可视化 contract 不变）
    sim.scat = sim.ax_main.scatter([], [], s=2, marker='s', edgecolors='none', alpha=0.8)
    # 大人口下只绘制随机抽样点（完整状态仍全部参与模拟）；固定抽样保证画面结构稳定
    sim._RENDER_MAX = int(CONFIG["RENDER_MAX_POINTS"])
    if sim._RENDER_MAX and sim.N > sim._RENDER_MAX:
        sim.render_idx = torch.randperm(sim.N, device=device)[:sim._RENDER_MAX]
    else:
        # None = 不抽样：F 段走"全量 set_offsets + 整块 facecolor"路径
        sim.render_idx = None
    # 三行标题（overlay）：F 段每帧重写文本，三行分别是人口/传播/代数指标
    sim.title_obj = sim.ax_main.text(0.5, 0.99, "", transform=sim.ax_main.transAxes,
                             ha="center", va="top", fontsize=9.5, fontweight='bold',
                             clip_on=True,
                             bbox=dict(facecolor='white', alpha=0.55, edgecolor='none'))

    # 三个曲线子图共用 0..500 的 x 窗口（= 500 帧），超出后 F 段滚动历史
    sim.ax_pop = plt.subplot2grid((3, 3), (0, 2))
    sim.ax_pop.set_title('Population Status')
    sim.ax_pop.set_xlim(0, 500)
    sim.ax_pop.set_ylim(0, sim.N)

    # 8 条人口曲线：下标/颜色/标签顺序 = status 码 0..7
    sim.history = {s: [] for s in range(8)}
    sim.lines_pop = [sim.ax_pop.plot([], [], color=COLOR_MAP[i],
                 label=['Healthy', 'Immune', 'Incubating', 'Symptomatic', 'Dead', 'Recovered',
                        'Asymptomatic', 'Hospitalized'][i])[0]
                 for i in range(8)]
    sim.ax_pop.legend(loc='upper right', fontsize=7)

    sim.ax_virus = plt.subplot2grid((3, 3), (1, 2))
    sim.ax_virus.set_title('Virus Evolution')
    sim.ax_virus.set_xlim(0, 500)
    # y 上限 1.6：beta/immunity ∈ [0,1]，mu 按图例标注 ×10 后仍不超过 1.6
    sim.ax_virus.set_ylim(0, 1.6)

    sim.line_beta = sim.ax_virus.plot([], [], 'r-', label='Beta (transmit)', linewidth=2)[0]
    sim.line_mu = sim.ax_virus.plot([], [], 'orange', label='Mu (death x10)', linewidth=2)[0]
    sim.line_immunity = sim.ax_virus.plot([], [], 'cyan', label='Avg Immunity', linewidth=2)[0]
    sim.ax_virus.legend(loc='upper right', fontsize=8)

    sim.virus_history = {'beta': [], 'mu': [], 'gen': [], 'immunity': []}

    sim.ax_gen = plt.subplot2grid((3, 3), (2, 2))
    sim.ax_gen.set_title('Generation & Reinfections')
    sim.ax_gen.set_xlim(0, 500)
    # y 上限 105：同时容纳平均代数（<100）与再感染百分比（0..100）
    sim.ax_gen.set_ylim(0, 105)
    sim.line_gen = sim.ax_gen.plot([], [], 'cyan', linewidth=2, label='Avg Gen')[0]
    sim.line_reinf = sim.ax_gen.plot([], [], 'magenta', linewidth=2, label='Reinfection %')[0]
    sim.ax_gen.legend(loc='upper right', fontsize=8)

    # 按钮用 figure 绝对坐标放在主图下方（x=0.45, y=0.02，宽 0.1 高 0.04）
    sim.ax_btn = plt.axes([0.45, 0.02, 0.1, 0.04])
    sim.btn = Button(sim.ax_btn, 'LOCKDOWN', color='#FF4500')

    def toggle(event):
        # 位置参数 event 匹配 mpl 的回调签名（Button 以 event 调用，参数名无所谓但必须收）
        sim.lockdown_active = not sim.lockdown_active
        sim.btn.label.set_text('RESUME' if sim.lockdown_active else 'LOCKDOWN')
        # 同时更新 Button.color：mpl 的 _motion 在鼠标移出时用它恢复颜色，
        # 只改 ax.facecolor 会被 hover 事件覆盖（3.9.x 行为）
        sim.btn.color = '#32CD32' if sim.lockdown_active else '#FF4500'
        sim.btn.ax.set_facecolor(sim.btn.color)
        sim.btn.ax.figure.canvas.draw_idle()

    sim.btn.on_clicked(toggle)

    # 布局留白：给主图标题、底部按钮与右侧图例留空间
    sim.fig.subplots_adjust(left=0.06, right=0.97, bottom=0.07, top=0.93,
                        wspace=0.25, hspace=0.35)
    # 暴露给外部/测试，等价于点击按钮（headless 环境无法真的点）
    sim.toggle = toggle
