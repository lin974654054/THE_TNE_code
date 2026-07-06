import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import scipy.constants as const

# ==========================================
# 0. 全局字体与样式设置
# ==========================================
# 设置全局字体为 Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'
# 设置公式里的字体也保持一致
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Times New Roman'
plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'
# 3. 可选：调大全局默认字号，防止缩放后看不清
plt.rcParams['font.size'] = 20
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['legend.fontsize'] = 20

# ==========================================
# 1. 物理参数与真实的 Kagome 坐标
# ==========================================
t_hop = -1.0     # 跃迁能量 (eV)
T_elec = 0.05  # 电子展宽温度 (eV)

# 面内 120度 Néel 排列，面外具有统一倾角 theta
theta = np.pi / 3  
sin_t = np.sin(theta)
cos_t = np.cos(theta)

M_A = np.array([sin_t, 0.0, cos_t])
M_B = np.array([-0.5 * sin_t,  np.sqrt(3)/2 * sin_t, cos_t])
M_C = np.array([-0.5 * sin_t, -np.sqrt(3)/2 * sin_t, cos_t])    

sigma_0 = np.eye(2, dtype=complex)
sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)

def get_Hk(kx, ky, J_val):
    H = np.zeros((6, 6), dtype=complex)
    
    H[0:2, 0:2] = J_val * (M_A[0]*sigma_x + M_A[1]*sigma_y + M_A[2]*sigma_z)
    H[2:4, 2:4] = J_val * (M_B[0]*sigma_x + M_B[1]*sigma_y + M_B[2]*sigma_z)
    H[4:6, 4:6] = J_val * (M_C[0]*sigma_x + M_C[1]*sigma_y + M_C[2]*sigma_z)
    
    d_AB = np.array([0.5, 0.0])
    d_AC = np.array([0.25, np.sqrt(3)/4])
    d_BC = np.array([-0.25, np.sqrt(3)/4])
    
    k_vec = np.array([kx, ky])
    tAB = 2 * t_hop * np.cos(np.dot(k_vec, d_AB))
    tAC = 2 * t_hop * np.cos(np.dot(k_vec, d_AC))
    tBC = 2 * t_hop * np.cos(np.dot(k_vec, d_BC))

    H[0:2, 2:4] = tAB * sigma_0
    H[2:4, 0:2] = tAB * sigma_0
    H[0:2, 4:6] = tAC * sigma_0
    H[4:6, 0:2] = tAC * sigma_0
    H[2:4, 4:6] = tBC * sigma_0
    H[4:6, 2:4] = tBC * sigma_0
    
    return H

def get_velocity_operators(kx, ky, J_val, delta=1e-5):
    vx = (get_Hk(kx + delta, ky, J_val) - get_Hk(kx - delta, ky, J_val)) / (2 * delta)
    vy = (get_Hk(kx, ky + delta, J_val) - get_Hk(kx, ky - delta, J_val)) / (2 * delta)
    return vx, vy

def calculate_berry_curvature_kubo(J_val, Nk=100):
    b1 = np.array([2*np.pi, -2*np.pi/np.sqrt(3)])
    b2 = np.array([0, 4*np.pi/np.sqrt(3)])
    
    k1 = np.linspace(0, 1, Nk, endpoint=False)
    k2 = np.linspace(0, 1, Nk, endpoint=False)
    
    E_grid = np.zeros((Nk, Nk, 6))
    Omega_kubo = np.zeros((Nk, Nk, 6))
    
    for i in range(Nk):
        for j in range(Nk):
            kx, ky = k1[i]*b1 + k2[j]*b2
            H = get_Hk(kx, ky, J_val)
            vals, vecs = np.linalg.eigh(H)
            E_grid[i, j, :] = vals
            
            vx, vy = get_velocity_operators(kx, ky, J_val)
            vx_band = vecs.T.conj() @ vx @ vecs
            vy_band = vecs.T.conj() @ vy @ vecs
            
            for n in range(6):
                omega_n = 0.0
                for m in range(6):
                    if n == m: continue
                    dE = vals[n] - vals[m]
                    if abs(dE) < 1e-6: continue 
                    omega_n += -2 * np.imag(vx_band[n, m] * vy_band[m, n]) / (dE**2)
                Omega_kubo[i, j, n] = omega_n
                
    return E_grid, Omega_kubo

# ==========================================
# 2. 循环扫描 3 个特定的 J 值
# ==========================================
J_arr = [0.0, 0.2, -0.2]

Nk_grid = 100  
mu_arr = np.linspace(-5, 5, 200)

results_AHC = {}
results_ANE = {}

e = const.e
hbar = const.hbar
kB = const.k
A_c = np.sqrt(3) / 2.0 
dk_measure = 1.0 / (A_c * Nk_grid**2)
coeff_AHC = (e**2 / hbar) * dk_measure     
coeff_ANE = (kB * e / hbar) * dk_measure          

print(f"🚀 开始计算 J_sd = {J_arr} ...")

for J_val in J_arr:
    print(f" └─ 正在对角化与积分 J = {J_val} ...")
    E_grid, Omega_grid = calculate_berry_curvature_kubo(J_val, Nk=Nk_grid)
    
    sigma_xy = np.zeros_like(mu_arr)
    alpha_xy = np.zeros_like(mu_arr)
    
    for idx, mu in enumerate(mu_arr):
        f_fd = 1.0 / (np.exp((E_grid - mu) / T_elec) + 1.0)
        f_safe = np.clip(f_fd, 1e-15, 1 - 1e-15)
        s_factor = -f_safe * np.log(f_safe) - (1 - f_safe) * np.log(1 - f_safe)
        
        sigma_xy[idx] = -coeff_AHC * np.sum(f_fd * Omega_grid)
        alpha_xy[idx] = coeff_ANE * np.sum(s_factor * Omega_grid)
        
    results_AHC[J_val] = sigma_xy
    results_ANE[J_val] = alpha_xy

# ==========================================
# 3. 绘制独立双图 (专为单栏上下拼接优化：图1加高带全图例，图2无图例)
# ==========================================

# 保持全局样式设置 (字号微调以适应单栏画布)
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'mathtext.fontset': 'custom',
    'mathtext.rm': 'Times New Roman',
    'mathtext.it': 'Times New Roman:italic',
    'mathtext.bf': 'Times New Roman:bold',
    'font.size': 20,              
    'axes.labelsize': 24,         
    'xtick.labelsize': 24,        
    'ytick.labelsize': 24,        
    'legend.fontsize': 24,        
    'axes.linewidth': 2.5,        
    'xtick.major.width': 2.5,     
    'ytick.major.width': 2.5, 
    'xtick.major.size': 8,        
    'ytick.major.size': 8,
    'xtick.direction': 'in',      
    'ytick.direction': 'in',
    'legend.frameon': False,      
    'figure.dpi': 300 
})

color_map = {0.0: 'black', 0.2: '#d62728', -0.2: '#1f77b4'}
ls_map = {0.0: '-', 0.2: ':', -0.2: '--'} 

# -----------------
# 第一张图: AHC (加高，展示全部 3 个图例，排成一行)
# -----------------
# 高度设为 6.5，为上方图例腾出空间
fig1, ax1 = plt.subplots(figsize=(7, 6.5)) 
for J_val in J_arr:
    # 直接赋予所有曲线 label
    current_label = r'$J_{\mathrm{K}} = ' + f'{J_val}' + r' |t|$'
    
    ax1.plot(mu_arr, results_AHC[J_val], color=color_map[J_val], 
             linestyle=ls_map[J_val], linewidth=4.5, 
             label=current_label)

ax1.set_xlabel(r'$\mu$ (eV)')
ax1.set_ylabel(r'$\sigma^{\mathrm{int}}_{xy}$ (S)')
ax1.set_xlim([-4.5, 2.5])
ax1.grid(False) 

formatter1 = ticker.ScalarFormatter(useMathText=True)
formatter1.set_scientific(True)
formatter1.set_powerlimits((-2, 2))
ax1.yaxis.set_major_formatter(formatter1)

# 图例置顶，绝对居中，设置 ncol=3 排成一行
# 微调 columnspacing 以确保 3 个图例在 7 英寸宽度内不拥挤
ax1.legend(loc='lower center', bbox_to_anchor=(0.4, 1.02), 
           ncol=3, columnspacing=0.8, handlelength=1.5)

# 调整 top=0.85 留出上方空间
fig1.subplots_adjust(left=0.18, right=0.95, bottom=0.18, top=0.85)
fig1.savefig("AHC_SingleCol_TopWithLegend.png", dpi=300)


# -----------------
# 第二张图: ANE (标准高度，无图例)
# -----------------
# 高度设为 5.5，不留图例空间
fig2, ax2 = plt.subplots(figsize=(7, 5.5))
for J_val in J_arr:
    # 画线但不设置图例
    ax2.plot(mu_arr, results_ANE[J_val], color=color_map[J_val], 
             linestyle=ls_map[J_val], linewidth=4.5)

ax2.set_xlabel(r'$\mu$ (eV)')
ax2.set_ylabel(r'$\alpha^{\mathrm{int}}_{xy}$ (A/K)')
ax2.set_xlim([-4.5, 2.5])
ax2.grid(False) 

formatter2 = ticker.ScalarFormatter(useMathText=True)
formatter2.set_scientific(True)
formatter2.set_powerlimits((-2, 2))
ax2.yaxis.set_major_formatter(formatter2)

# 不调用 ax2.legend()

# 调整 top=0.95，因为不需要给图例留白，直接顶满
fig2.subplots_adjust(left=0.18, right=0.95, bottom=0.18, top=0.90)
fig2.savefig("ANE_SingleCol_BottomNoLegend.png", dpi=300)

plt.show()

print("\n✅ 修改完成：所有图例已集中在第一幅图上方并单排显示，双图已完美适配上下拼接！")