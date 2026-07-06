import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
import matplotlib.ticker as ticker
from numba import njit

# ==========================================
# 0. 全局字体与样式设置 (针对 PRB 标准与大字号优化)
# ==========================================
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'mathtext.fontset': 'custom',
    'mathtext.rm': 'Times New Roman',
    'mathtext.it': 'Times New Roman:italic',
    'mathtext.bf': 'Times New Roman:bold',
    'font.size': 30,
    'axes.labelsize': 40,           
    'xtick.labelsize': 30,
    'ytick.labelsize': 30,
    'axes.linewidth': 2.5,          
    'xtick.major.width': 2.5,
    'ytick.major.width': 2.5,
    'xtick.major.size': 8,
    'ytick.major.size': 8,
    'xtick.direction': 'in',        
    'ytick.direction': 'in',
    'figure.dpi': 300
})

# ==========================================
# 1. 读取保存的数据
# ==========================================
# 注意：请替换为你实际生成的数据文件名
data = np.load("Dz_1.7320508075688772_0323_48x48_60x60_100000_400000_kagome_ultimate_phase_data.npz")

T_arr = data['T_arr']
Bz_arr = data['Bz_arr']
phase_chi_total = data['phase_chi_total']
phase_chi_bg = data['phase_chi_bg']
phase_delta_chi = data['phase_delta_chi']
phase_mz = data['phase_mz']
phase_mlocal = data['phase_mlocal']
phase_cv = data['phase_cv']
boundary_T = data['boundary_T']

# 如果数据中有 phase_skew_chi，取消注释即可
# phase_skew_chi = data['phase_skew_chi']

# ==========================================
# 2. 计算等温磁化率 chi_m = d(Mz)/d(Bz)
# ==========================================
phase_susceptibility = np.gradient(phase_mz, Bz_arr, axis=0)

# ==========================================
# 3. 高斯平滑滤波
# ==========================================
sigma_val = 2
phase_mz = ndimage.gaussian_filter(phase_mz, sigma=sigma_val)
phase_mlocal = ndimage.gaussian_filter(phase_mlocal, sigma=sigma_val)
phase_chi_bg = ndimage.gaussian_filter(phase_chi_bg, sigma=sigma_val)
phase_delta_chi = ndimage.gaussian_filter(phase_delta_chi, sigma=sigma_val)
boundary_T = ndimage.gaussian_filter(boundary_T, sigma=sigma_val)
phase_susceptibility = ndimage.gaussian_filter(phase_susceptibility, sigma=sigma_val)
phase_cv = ndimage.gaussian_filter(phase_cv, sigma=sigma_val)

# phase_skew_chi = ndimage.gaussian_filter(phase_skew_chi, sigma=sigma_val)

# ==========================================
# 4. 翻转温度轴数据用于画图 (适配 T_arr[::-1])
# ==========================================
T_arr_plot = T_arr[::-1]
phase_chi_total_plot = phase_chi_total[:, ::-1]
phase_mlocal_plot = phase_mlocal[:, ::-1]
phase_chi_bg_plot = phase_chi_bg[:, ::-1]
phase_delta_chi_plot = phase_delta_chi[:, ::-1]
phase_mz_plot = phase_mz[:, ::-1]
phase_cv_plot = phase_cv[:, ::-1]
phase_susceptibility_plot = phase_susceptibility[:, ::-1]

# phase_skew_chi_plot = phase_skew_chi[:, ::-1]

# ==========================================
# 5. 计算拓扑电荷 (Skyrmion Number Density) 的引擎
# ==========================================
@njit
def calc_topological_charge_density(spins_avg, L):
    total_Q = 0.0
    for x in range(L):
        for y in range(L):
            sA = spins_avg[x, y, 0].copy()
            sB = spins_avg[x, y, 1].copy()
            sC = spins_avg[x, y, 2].copy()
            
            nA = np.linalg.norm(sA); sA = sA / nA if nA > 1e-8 else sA * 0
            nB = np.linalg.norm(sB); sB = sB / nB if nB > 1e-8 else sB * 0
            nC = np.linalg.norm(sC); sC = sC / nC if nC > 1e-8 else sC * 0
            
            num_up = sA[0]*(sB[1]*sC[2] - sB[2]*sC[1]) - sA[1]*(sB[0]*sC[2] - sB[2]*sC[0]) + sA[2]*(sB[0]*sC[1] - sB[1]*sC[0])
            den_up = 1.0 + np.dot(sA, sB) + np.dot(sB, sC) + np.dot(sC, sA)
            omega_up = 2.0 * np.arctan2(num_up, den_up)
            
            sB_down = spins_avg[(x-1)%L, y, 1].copy()
            sC_down = spins_avg[x, (y-1)%L, 2].copy()
            
            nB_d = np.linalg.norm(sB_down); sB_down = sB_down / nB_d if nB_d > 1e-8 else sB_down * 0
            nC_d = np.linalg.norm(sC_down); sC_down = sC_down / nC_d if nC_d > 1e-8 else sC_down * 0
            
            num_dn = sA[0]*(sB_down[1]*sC_down[2] - sB_down[2]*sC_down[1]) - sA[1]*(sB_down[0]*sC_down[2] - sB_down[2]*sC_down[0]) + sA[2]*(sB_down[0]*sC_down[1] - sB_down[1]*sC_down[0])
            den_dn = 1.0 + np.dot(sA, sB_down) + np.dot(sB_down, sC_down) + np.dot(sC_down, sA)
            omega_dn = 2.0 * np.arctan2(num_dn, den_dn)
            
            total_Q += (omega_up + omega_dn)
            
    return total_Q / (4.0 * np.pi )

phase_spin_config = data['phase_spin_config'] 
L = phase_spin_config.shape[2]
print("L",L)
phase_topo_charge = np.zeros((len(Bz_arr), len(T_arr)))

print("开始计算全相图的拓扑电荷 (Topological Charge)...")
for i in range(len(Bz_arr)):
    for j in range(len(T_arr)):
        spins_avg = phase_spin_config[i, j]
        phase_topo_charge[i, j] = calc_topological_charge_density(spins_avg, L)
print("计算完成！")

phase_topo_charge = ndimage.gaussian_filter(phase_topo_charge, sigma=sigma_val)
phase_topo_charge_plot = phase_topo_charge[:, ::-1]

# ==========================================
# 6. 完整物理量配置列表 (确保不再报错)
# ==========================================
plot_configs = [
    (phase_chi_total_plot, r'$\chi_{\mathrm{total}}$', 'Chi_Total'),
    (phase_cv_plot, r'$C_{V}$', 'Cv'),
    (phase_mlocal_plot, r'$M_{\mathrm{total}}$', 'M_Total'),
    (phase_topo_charge_plot, r'$\chi_{Q}$', 'Chi_Q'),
    (phase_chi_bg_plot, r'$\chi_{\mathrm{bg}}$', 'Chi_BG'),
    (phase_delta_chi_plot, r'$\delta\chi$', 'Delta_Chi'),
    (phase_mz_plot, r'$M_{z}$', 'Mz'),
    (phase_susceptibility_plot, r'$\chi_{m}$', 'Chi_M')
]

# ==========================================
# 7. 批量循环绘图引擎 (彻底解决 10^n 重合问题)
# ==========================================
X, Y = np.meshgrid(T_arr_plot, Bz_arr)

for data_plot, cbar_label, suffix in plot_configs:
    fig, ax = plt.subplots(figsize=(8, 6.5))
    
    c = ax.pcolormesh(X, Y, data_plot, cmap='viridis', shading='auto', rasterized=True)
    ax.contour(X, Y, data_plot, levels=8, colors='white', linewidths=0.5, alpha=0.3)
    
    # Colorbar 设置
    cb = fig.colorbar(c, ax=ax, fraction=0.046, pad=0.05)
    
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-1, 1)) 
    cb.ax.yaxis.set_major_formatter(formatter)
    
    # 强制 Matplotlib 计算一次布局锁定内部状态
    plt.draw() 
    
    # 调整指数标签 (10^n) 的位置与字号
    offset_text = cb.ax.yaxis.get_offset_text()
    offset_text.set_fontsize(30)
    #offset_text.set_fontweight('bold')
    
    # 更改对齐方式，以左下角为锚点
    offset_text.set_ha('left')
    offset_text.set_va('bottom')
    
    # 悬浮微调：x=1.3(往右), y=1.05(往上)
    offset_text.set_position((1.3, 1.05)) 
    
    # Colorbar 侧边标题
    cb.set_label(label=cbar_label, size=40, weight='bold', labelpad=15)
    cb.ax.tick_params(labelsize=30)
    
    # 坐标轴设置
    ax.set_xscale('log') 
    ax.set_xlabel(r'$T/J_{\mathrm{H}}$', fontweight='bold', labelpad=10)
    ax.set_ylabel(r'$B/J_{\mathrm{H}}$', fontweight='bold', labelpad=10)
    
    ax.set_yticks([0, 0.05, 0.1, 0.15, 0.2])
    ax.tick_params(axis='both', which='both', pad=10)

    # 调大右侧与顶部的留白，给悬浮的 10^n 腾出空间
    fig.subplots_adjust(left=0.2, right=0.78, bottom=0.2, top=0.9)
    
    # 保存与清理
    save_name = f"Phase_{suffix}_Final_Sci.png"
    fig.savefig(save_name, dpi=300)
    plt.close(fig) 
    print(f"✅ 已成功导出：{save_name}")

print("\n🚀 所有相图已完成自动化生成，排版与代码架构完美统一！")