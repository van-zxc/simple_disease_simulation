import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button

# ==================== 配置参数 ====================
CONFIG = {
    # 基础参数
    "N_AGENTS": 10000, # 总人口数
    "WIDTH": 100.0, # 区域宽度
    "HEIGHT": 100.0, # 区域高度
    "N_HOTSPOTS": 3, # 热点数量
    "INFECT_RADIUS": 1.8, # 感染距离阈值
    "BASE_TRANSMIT": 0.3, # 基础传播率
    "BASE_RECOVERY": 0.008, # 基础恢复率
    "BASE_DEATH": 0.01, # 基础死亡率
    "MUTATION_RATE": 0.25, # 基础变异率
    "MUTATION_SCALE": 0.15, # 变异幅度
    "INCUBATION_MU": 40.0, # 潜伏期均值
    "MEDICAL_THRESHOLD": 1000, # 医疗系统崩溃阈值
    "TRANSMIT_BASE_PROB": 0.08, # 基础传染概率
    
    # 免疫系统参数
    "IMMUNITY_GAIN_PER_INFECTION": 0.4,   # 每次感染获得的免疫力
    "IMMUNITY_DECAY_RATE": 0.0005,        # 免疫力衰减速率（每步）
    "MAX_IMMUNITY": 0.98,                 # 最大免疫力（不会完全免疫）
    "IMMUNITY_EFFECTIVENESS": 3,        # 免疫力效果倍率
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==================== 初始化 ====================
W, H = CONFIG["WIDTH"], CONFIG["HEIGHT"]
N = CONFIG["N_AGENTS"]

margin = W * 0.1
hotspots = torch.zeros((CONFIG["N_HOTSPOTS"], 2), device=device)
hotspots[:, 0] = margin + torch.rand(CONFIG["N_HOTSPOTS"], device=device) * (W - 2*margin)
hotspots[:, 1] = margin + torch.rand(CONFIG["N_HOTSPOTS"], device=device) * (H - 2*margin)

pos = torch.rand((N, 2), device=device) * torch.tensor([W, H], device=device)
homes = pos.clone()
vel = torch.zeros((N, 2), device=device)

status = torch.full((N,), 0, dtype=torch.long, device=device)

virus_beta = torch.zeros(N, device=device)
virus_gamma = torch.zeros(N, device=device)
virus_mu = torch.zeros(N, device=device)
virus_sigma = torch.zeros(N, device=device)
virus_gen = torch.zeros(N, dtype=torch.long, device=device)

timer = torch.zeros(N, device=device)
max_incubation = torch.zeros(N, device=device)

# ==================== 免疫系统 ====================
immunity_level = torch.zeros(N, device=device)      # 免疫力水平 [0, 1]
infection_count = torch.zeros(N, dtype=torch.long, device=device)  # 感染次数
last_infected_gen = torch.zeros(N, dtype=torch.long, device=device)  # 上次感染的病毒代数

# 初始免疫 1%
initial_immune = torch.rand(N, device=device) < 0.01
status[initial_immune] = 1
immunity_level[initial_immune] = 0.9  # 初始免疫者有高免疫力

agent_target_hotspot = torch.randint(0, CONFIG["N_HOTSPOTS"], (N,), device=device)
individual_offsets = torch.randint(0, 200, (N,), device=device)
jitter_scale = W * 0.08
hotspot_jitters = (torch.rand((N, 2), device=device) - 0.5) * jitter_scale

# 初始感染源
dist_to_h1 = torch.norm(pos - hotspots[0], dim=1)
seeds = torch.argsort(dist_to_h1)[:20]
status[seeds] = 2
virus_beta[seeds] = CONFIG["BASE_TRANSMIT"]
virus_gamma[seeds] = CONFIG["BASE_RECOVERY"]
virus_mu[seeds] = CONFIG["BASE_DEATH"]
virus_sigma[seeds] = CONFIG["MUTATION_RATE"]
virus_gen[seeds] = 1
max_incubation[seeds] = CONFIG["INCUBATION_MU"]
infection_count[seeds] = 1
last_infected_gen[seeds] = 1

lockdown_active = False
mutation_events = 0

# 统计
strain_stats = {'avg_immunity': [], 'reinfection_rate': []}

# ==================== 可视化设置 ====================
COLOR_MAP = ["#B0B0B0", '#00BFFF', '#FFD700', '#FF4500', '#000000', '#32CD32']
fig = plt.figure(figsize=(18, 10))

ax_main = plt.subplot2grid((3, 3), (0, 0), rowspan=3, colspan=2)
ax_main.set_xlim(0, W)
ax_main.set_ylim(0, H)
ax_main.set_aspect('equal')
ax_main.set_title('Virus Evolution CA with Immunity System', fontsize=14, fontweight='bold')

for i in range(CONFIG["N_HOTSPOTS"]):
    circ = plt.Circle(hotspots[i].cpu().numpy(), jitter_scale/2, color='red', alpha=0.05)
    ax_main.add_patch(circ)

scat = ax_main.scatter([], [], s=2, edgecolors='none', alpha=0.8)
title_obj = ax_main.text(0.5, 1.05, "", transform=ax_main.transAxes, 
                         ha="center", fontsize=12, fontweight='bold')

ax_pop = plt.subplot2grid((3, 3), (0, 2))
ax_pop.set_title('Population Status')
ax_pop.set_xlim(0, 500)
ax_pop.set_ylim(0, N)

history = {s: [] for s in range(6)}
lines_pop = [ax_pop.plot([], [], color=COLOR_MAP[i], 
             label=['Healthy', 'Immune', 'Incubating', 'Symptomatic', 'Dead', 'Recovered'][i])[0] 
             for i in range(6)]
ax_pop.legend(loc='upper right', fontsize=8)

ax_virus = plt.subplot2grid((3, 3), (1, 2))
ax_virus.set_title('Virus Evolution')
ax_virus.set_xlim(0, 500)
ax_virus.set_ylim(0, 1.0)

line_beta = ax_virus.plot([], [], 'r-', label='Beta (transmit)', linewidth=2)[0]
line_mu = ax_virus.plot([], [], 'orange', label='Mu (death x10)', linewidth=2)[0]
line_immunity = ax_virus.plot([], [], 'cyan', label='Avg Immunity', linewidth=2)[0]
ax_virus.legend(loc='upper right', fontsize=8)

virus_history = {'beta': [], 'mu': [], 'gen': [], 'immunity': []}

ax_gen = plt.subplot2grid((3, 3), (2, 2))
ax_gen.set_title('Generation & Reinfections')
ax_gen.set_xlim(0, 500)
ax_gen.set_ylim(0, 100)
line_gen = ax_gen.plot([], [], 'cyan', linewidth=2, label='Avg Gen')[0]
line_reinf = ax_gen.plot([], [], 'magenta', linewidth=2, label='Reinfection %')[0]
ax_gen.legend(loc='upper right', fontsize=8)

ax_btn = plt.axes([0.45, 0.02, 0.1, 0.04])
btn = Button(ax_btn, 'LOCKDOWN', color='#FF4500')

def toggle(event):
    global lockdown_active
    lockdown_active = not lockdown_active
    btn.label.set_text('RESUME' if lockdown_active else 'LOCKDOWN')
    btn.color = '#32CD32' if lockdown_active else '#FF4500'

btn.on_clicked(toggle)

# ==================== 核心逻辑（含免疫系统）====================
def update(frame):
    global pos, vel, status, timer, virus_beta, virus_gamma, virus_mu, virus_sigma, virus_gen
    global lockdown_active, mutation_events, immunity_level, infection_count, last_infected_gen

    # A. 医疗系统压力计算
    num_symptomatic = (status == 3).sum().item()
    is_collapsed = num_symptomatic >= CONFIG["MEDICAL_THRESHOLD"]
    medical_pressure = 2.5 if is_collapsed else 1.0

    # 免疫力衰减
    alive_mask = status != 4
    immunity_level[alive_mask] = torch.clamp(
        immunity_level[alive_mask] - CONFIG["IMMUNITY_DECAY_RATE"],
        min=0.0
    )

    # B. 运动逻辑
    if not lockdown_active:
        # 移动更新
        alive = (status != 4)
        local_time = (frame + individual_offsets) % 200
        is_working = local_time < 80
        
        # 目标位置设定
        target = homes.clone()
        assigned_h = hotspots[agent_target_hotspot]
        target[is_working] = assigned_h[is_working] + hotspot_jitters[is_working]
        
        # 计算加速度
        dir_vec = target - pos
        dist = torch.norm(dir_vec, dim=1, keepdim=True) + 1e-5
        accel = torch.clamp(dist * 0.02, max=0.08)
        at_dest = (dist.squeeze() < W * 0.03)
        # 更新速度和位置
        vel[alive] += (dir_vec[alive] / dist[alive]) * accel[alive]
        vel[alive & at_dest] += torch.randn((torch.sum(alive & at_dest).item(), 2), device=device) * (W * 0.0005)
        vel *= 0.88
        pos[alive] += vel[alive]
        pos[:, 0] = torch.clamp(pos[:, 0], 0, W)
        pos[:, 1] = torch.clamp(pos[:, 1], 0, H)
    else:
        vel *= 0

    # C. 潜伏期逻辑
    incu_mask = (status == 2)
    # 潜伏期计时
    if incu_mask.any():
        timer[incu_mask] += 1
        to_symp = timer[incu_mask] >= max_incubation[incu_mask]
        status[torch.where(incu_mask)[0][to_symp]] = 3

        # 自愈判定
        self_recov = (~to_symp) & (torch.rand(incu_mask.sum().item(), device=device) > 0.9995)
        recovered_idx = torch.where(incu_mask)[0][self_recov]
        status[recovered_idx] = 5
        
        # 康复时增强免疫力
        immunity_level[recovered_idx] += CONFIG["IMMUNITY_GAIN_PER_INFECTION"]
        immunity_level[recovered_idx] = torch.clamp(immunity_level[recovered_idx], 0, CONFIG["MAX_IMMUNITY"])
        
        # 清除病毒属性
        virus_beta[recovered_idx] = 0
        virus_gamma[recovered_idx] = 0
        virus_mu[recovered_idx] = 0
        virus_sigma[recovered_idx] = 0
        virus_gen[recovered_idx] = 0

    # D. 发病判定
    symp_mask = (status == 3)
    # 病死和康复判定
    if symp_mask.any():
        d_p = virus_mu[symp_mask] * medical_pressure * 0.05
        r_p = virus_gamma[symp_mask] / medical_pressure

        # 进行判定
        rolls = torch.rand(symp_mask.sum().item(), device=device)
        idx = torch.where(symp_mask)[0]
        
        # 死亡判定
        is_dead = rolls < d_p
        dead_idx = idx[is_dead]
        status[dead_idx] = 4
        virus_beta[dead_idx] = 0
        virus_gamma[dead_idx] = 0
        virus_mu[dead_idx] = 0
        virus_sigma[dead_idx] = 0
        virus_gen[dead_idx] = 0
        
        # 康复判定
        is_rec = (~is_dead) & (rolls > (1 - r_p))
        rec_idx = idx[is_rec]
        status[rec_idx] = 5
        
        # 康复时增强免疫力
        immunity_level[rec_idx] += CONFIG["IMMUNITY_GAIN_PER_INFECTION"]
        immunity_level[rec_idx] = torch.clamp(immunity_level[rec_idx], 0, CONFIG["MAX_IMMUNITY"])
        
        # 清除病毒属性
        virus_beta[rec_idx] = 0
        virus_gamma[rec_idx] = 0
        virus_mu[rec_idx] = 0
        virus_sigma[rec_idx] = 0
        virus_gen[rec_idx] = 0

    # E. 传染逻辑
    inf_mask = (status == 2) | (status == 3)
    target_mask = (status == 0) | (status == 5)
    
    # 传染过程
    if inf_mask.any() and target_mask.any():
        inf_pos = pos[inf_mask]
        tar_pos = pos[target_mask]
        
        # 计算距离矩阵
        dists = torch.cdist(inf_pos, tar_pos)
        contacts = dists < CONFIG["INFECT_RADIUS"]
        
        # 仅处理有接触的情况
        if contacts.any():
            lockdown_mult = 0.15 if lockdown_active else 1.0
            status_mult = torch.where(status[inf_mask] == 2, 
                                     torch.tensor(0.15, device=device), 
                                     torch.tensor(1.0, device=device))
            
            beta_factor = torch.pow(virus_beta[inf_mask], 1.5)
            
            # 基础传播概率
            p_base = (beta_factor.unsqueeze(1) * 
                     status_mult.unsqueeze(1) * 
                     lockdown_mult * 
                     medical_pressure * 
                     CONFIG["TRANSMIT_BASE_PROB"])
            
            target_indices = torch.where(target_mask)[0]
            target_immunity = immunity_level[target_indices]

            immunity_resistance = torch.pow(1 - target_immunity, CONFIG["IMMUNITY_EFFECTIVENESS"])
            p_final = p_base * immunity_resistance.unsqueeze(0)
            
            success = contacts & (torch.rand(contacts.shape, device=device) < p_final)
            
            # 处理成功感染
            if success.any():
                was_inf = success.any(dim=0)
                src_map = success.float().argmax(dim=0)
                
                inf_indices = torch.where(inf_mask)[0]
                sources = inf_indices[src_map[was_inf]]
                targets = target_indices[was_inf]
                
                # 记录再感染
                is_reinfection = status[targets] == 5
                
                status[targets] = 2
                
                # 增加感染计数
                infection_count[targets] += 1
                last_infected_gen[targets] = virus_gen[sources]
                
                num_new = len(targets)
                
                # Beta变异
                mutation_mask_beta = torch.rand(num_new, device=device) < virus_sigma[sources]
                new_beta = virus_beta[sources].clone()
                delta_beta = (torch.rand(num_new, device=device) - 0.5) * CONFIG["MUTATION_SCALE"]
                new_beta[mutation_mask_beta] += delta_beta[mutation_mask_beta]
                virus_beta[targets] = torch.clamp(new_beta, 0.05, 1.0)
                mutation_events += mutation_mask_beta.sum().item()
                
                # Gamma变异
                mutation_mask_gamma = torch.rand(num_new, device=device) < virus_sigma[sources]
                new_gamma = virus_gamma[sources].clone()
                delta_gamma = (torch.rand(num_new, device=device) - 0.5) * CONFIG["MUTATION_SCALE"] * 0.3
                new_gamma[mutation_mask_gamma] += delta_gamma[mutation_mask_gamma]
                virus_gamma[targets] = torch.clamp(new_gamma, 0.001, 0.05)
                mutation_events += mutation_mask_gamma.sum().item()
                
                # Mu变异
                mutation_mask_mu = torch.rand(num_new, device=device) < virus_sigma[sources]
                new_mu = virus_mu[sources].clone()
                delta_mu = (torch.rand(num_new, device=device) - 0.5) * CONFIG["MUTATION_SCALE"] * 0.3
                new_mu[mutation_mask_mu] += delta_mu[mutation_mask_mu]
                virus_mu[targets] = torch.clamp(new_mu, 0.001, 0.15)
                mutation_events += mutation_mask_mu.sum().item()
                
                virus_sigma[targets] = virus_sigma[sources]
                virus_gen[targets] = virus_gen[sources] + 1
                
                max_incubation[targets] = torch.normal(CONFIG["INCUBATION_MU"], 10., (num_new,), device=device).clamp(min=10)
                timer[targets] = 0
                
                # 统计再感染率
                reinfection_count = is_reinfection.sum().item()
                total_new_infections = num_new
                reinfection_rate = reinfection_count / total_new_infections if total_new_infections > 0 else 0
                strain_stats['reinfection_rate'].append(reinfection_rate * 100)
            else:
                strain_stats['reinfection_rate'].append(0)
        else:
            strain_stats['reinfection_rate'].append(0)
    else:
        strain_stats['reinfection_rate'].append(0)

    # F. 统计
    scat.set_offsets(pos.cpu().numpy())
    scat.set_color([COLOR_MAP[s] for s in status.cpu().numpy()])
    
    total_infected = (status == 2).sum().item() + num_symptomatic
    
    # 计算平均免疫力
    alive_mask = status != 4
    avg_immunity = immunity_level[alive_mask].mean().item() if alive_mask.any() else 0
    strain_stats['avg_immunity'].append(avg_immunity)
    
    infected_mask = (status == 2) | (status == 3)
    if infected_mask.any():
        avg_beta = virus_beta[infected_mask].mean().item()
        avg_mu = virus_mu[infected_mask].mean().item()
        avg_gen = virus_gen[infected_mask].float().mean().item()
        
        virus_history['beta'].append(avg_beta)
        virus_history['mu'].append(avg_mu)
        virus_history['gen'].append(avg_gen)
        virus_history['immunity'].append(avg_immunity)
    else:
        virus_history['beta'].append(0)
        virus_history['mu'].append(0)
        virus_history['gen'].append(0)
        virus_history['immunity'].append(avg_immunity)
    
    # 更新标题
    recent_reinf = strain_stats['reinfection_rate'][-1] if strain_stats['reinfection_rate'] else 0
    title_text = f"Inf: {total_infected} | Symp: {num_symptomatic} | Med: {'COLLAPSE' if is_collapsed else 'OK'} | Immunity: {avg_immunity:.3f} | Reinf: {recent_reinf:.1f}%"
    title_obj.set_text(title_text)
    title_obj.set_color('red' if is_collapsed else 'green')

    # 更新人口状态历史
    for s in range(6):
        count = (status == s).sum().item()
        history[s].append(count)
        lines_pop[s].set_data(range(len(history[s])), history[s])
    
    # 限制显示范围
    if len(history[0]) > 500:
        ax_pop.set_xlim(len(history[0]) - 500, len(history[0]))

    # 更新病毒进化历史
    x = range(len(virus_history['beta']))
    line_beta.set_data(x, virus_history['beta'])
    line_mu.set_data(x, [m*10 for m in virus_history['mu']])
    line_immunity.set_data(x, virus_history['immunity'])
    
    if len(x) > 500:
        ax_virus.set_xlim(len(x) - 500, len(x))
    
    line_gen.set_data(x, virus_history['gen'])
    
    # 计算移动平均的再感染率
    window = 50
    if len(strain_stats['reinfection_rate']) >= window:
        smoothed_reinf = [sum(strain_stats['reinfection_rate'][max(0,i-window):i+1])/(min(window, i+1)) 
                         for i in range(len(strain_stats['reinfection_rate']))]
        line_reinf.set_data(range(len(smoothed_reinf)), smoothed_reinf)
    else:
        line_reinf.set_data(range(len(strain_stats['reinfection_rate'])), strain_stats['reinfection_rate'])
    
    if len(x) > 500:
        ax_gen.set_xlim(len(x) - 500, len(x))
    
    max_gen = max(virus_history['gen']) if virus_history['gen'] else 1
    ax_gen.set_ylim(0, max(50, max_gen + 5))

    return [scat, title_obj] + lines_pop + [line_beta, line_mu, line_immunity, line_gen, line_reinf]

# ==================== 主程序 ====================
print(f"Initialization complete, {N} agents")
print(f"\nImmunity System Parameters:")
print(f"  - Immunity gain per infection: {CONFIG['IMMUNITY_GAIN_PER_INFECTION']}")
print(f"  - Immunity decay rate: {CONFIG['IMMUNITY_DECAY_RATE']} per step")
print(f"  - Max immunity: {CONFIG['MAX_IMMUNITY']}")
print(f"  - Immunity effectiveness: {CONFIG['IMMUNITY_EFFECTIVENESS']}x")
print(f"\nMechanism: Recovered individuals gain immunity that decays over time")
print(f"Reinfection probability = base_prob × (1 - immunity)^{CONFIG['IMMUNITY_EFFECTIVENESS']}")
ani = FuncAnimation(fig, update, frames=5000, interval=1, blit=False)
plt.tight_layout()
plt.show()
print("Simulation ended")