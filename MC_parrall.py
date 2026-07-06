import numpy as np
import matplotlib.pyplot as plt
from numba import njit
import time
import os
import concurrent.futures

# ==========================================
# 1. 物理参数与扫描网格设置
# ==========================================
J_spin = -1.0    # 反铁磁交换作用
Dxy = 0.5        # 【必须添加这一行】面内 DMI 强度 (如果你想跑文献参数，请改为 1.5)
Dz = np.sqrt(3)       # DMI 强度
Kz = 0.0        # 重要：2.5 meV 能垒 (假设 J=10meV)
L = 30          # 晶格尺寸 (10x10 = 300 个自旋)
N_spins = L * L * 3 

# 扫描网格 (建议测试时用 15x15，出图时用 30x30 或更高)
T_arr = np.linspace(2.0, 0.05, 30)   
Bz_arr = np.linspace(0.0, 2.5, 60)  

BURN_IN_SWEEPS = int(6*1e3)  
MEASURE_SWEEPS = int(24*1e3) 

# 数据保存文件名
SAVE_FILENAME = f"Dz_{Dz}_0323_{L}x{L}_{len(T_arr)}x{len(Bz_arr)}_{BURN_IN_SWEEPS}_{MEASURE_SWEEPS}_kagome_ultimate_phase_data.npz"  

print(f"🚀 初始化终极并行 1x5 相图扫描: 尺寸 {L}x{L}")

# ==========================================
# 2. 拓扑预处理 (Numba 加速表)
# ==========================================
neighbor_table = np.zeros((L, L, 3, 4, 4), dtype=np.int32)
for x in range(L):
    for y in range(L):
        neighbor_table[x, y, 0] = [[x, y, 1, 1], [x, y, 2, -1], [(x-1)%L, y, 1, 1], [x, (y-1)%L, 2, -1]]
        neighbor_table[x, y, 1] = [[x, y, 2, 1], [x, y, 0, -1], [(x+1)%L, (y-1)%L, 2, 1], [(x+1)%L, y, 0, -1]]
        neighbor_table[x, y, 2] = [[x, y, 0, 1], [x, y, 1, -1], [x, (y+1)%L, 0, 1], [(x-1)%L, (y+1)%L, 1, -1]]

# 【新增】计算面内 DMI 矢量方向 ( D \propto z \times r_ij )
# 形状为 (3个子晶格, 4个邻居, 2维向量xy)
dxy_vectors = np.zeros((3, 4, 2), dtype=np.float64)
s32 = np.sqrt(3) / 2.0

# Sublattice 0 (A) 的 4 条键的 Dxy 矢量 (dx, dy)
dxy_vectors[0, 0] = [0.0, 1.0]        # 指向同胞 B 
dxy_vectors[0, 1] = [-s32, 0.5]       # 指向同胞 C
dxy_vectors[0, 2] = [0.0, -1.0]       # 指向左侧胞 B
dxy_vectors[0, 3] = [s32, -0.5]        # 指向下方胞 C

# Sublattice 1 (B) 的 4 条键的 Dxy 矢量
dxy_vectors[1, 0] = [-s32, -0.5]      # 指向同胞 C
dxy_vectors[1, 1] = [0.0, -1.0]       # 指向同胞 A
dxy_vectors[1, 2] = [s32, 0.5]       # 指向右下胞 C
dxy_vectors[1, 3] = [0.0, 1.0]        # 指向右侧胞 A

# Sublattice 2 (C) 的 4 条键的 Dxy 矢量
dxy_vectors[2, 0] = [s32, -0.5]       # 指向同胞 A
dxy_vectors[2, 1] = [s32, 0.5]        # 指向同胞 B
dxy_vectors[2, 2] = [-s32, 0.5]      # 指向上方胞 A
dxy_vectors[2, 3] = [-s32, -0.5]       # 指向左上胞 B

# ==========================================
# 3. 核心计算引擎 (Numba JIT)
# ==========================================
@njit
def set_numba_seed(seed_value):
    """专门为 Numba 设置随机数种子，确保多进程安全"""
    np.random.seed(seed_value)

@njit
def calc_local_energy_fast(x, y, alpha, spin_val, spins, Bz):
    E = -Bz * spin_val[2] + Kz * (spin_val[2]**2)
    for i in range(4):
        nx, ny, n_alpha, dmi_sign = neighbor_table[x, y, alpha, i]
        sj = spins[nx, ny, n_alpha]
        
        # 提取当前键的面内 DMI 几何分量
        dx = dxy_vectors[alpha, i, 0]
        dy = dxy_vectors[alpha, i, 1]
        
        # 海森堡项
        E += J_spin * (spin_val[0]*sj[0] + spin_val[1]*sj[1] + spin_val[2]*sj[2])
        # 面外 DMI (Dz)
        E += dmi_sign * Dz * (spin_val[0]*sj[1] - spin_val[1]*sj[0])
        # 【新增】面内 DMI (Dxy) 
        E += Dxy * (dx * (spin_val[1]*sj[2] - spin_val[2]*sj[1]) + 
                    dy * (spin_val[2]*sj[0] - spin_val[0]*sj[2]))
    return E
@njit
def over_relaxation_sweep(spins, Bz, T):
    """微正则过弛豫更新：让自旋绕局域有效场做镜像反射"""
    for _ in range(N_spins):
        x, y, alpha = np.random.randint(0, L), np.random.randint(0, L), np.random.randint(0, 3)
        old_spin = spins[x, y, alpha]
        
        # 1. 计算局域有效场 H_eff (仅限线性交互项：塞曼、海森堡、DMI)
        # 注意符号：H_eff = - dE/dS
        hx, hy, hz = 0.0, 0.0, Bz
        for i in range(4):
            nx, ny, n_alpha, dmi_sign = neighbor_table[x, y, alpha, i]
            sj = spins[nx, ny, n_alpha]

            # 提取并计算物理 DMI 矢量分量
            Dx_val = Dxy * dxy_vectors[alpha, i, 0]
            Dy_val = Dxy * dxy_vectors[alpha, i, 1]
            Dz_val = Dz * dmi_sign
            # 【核心修改】包含 Dxy 的有效场分量推导结果
            hx -= J_spin * sj[0] + Dz_val * sj[1] - Dy_val * sj[2]
            hy -= J_spin * sj[1] - Dz_val * sj[0] + Dx_val * sj[2]
            hz -= J_spin * sj[2] + Dy_val * sj[0] - Dx_val * sj[1]
            
            
        heff_norm2 = hx**2 + hy**2 + hz**2
        if heff_norm2 < 1e-12:
            continue  # 有效场为 0 时无法确定进动轴，跳过
            
        # 2. 计算 S 与 H_eff 的点乘
        dot_product = old_spin[0]*hx + old_spin[1]*hy + old_spin[2]*hz
        
        # 3. 执行镜像反射公式：S' = 2 * (S·H_eff) * H_eff / |H_eff|^2 - S
        factor = 2.0 * dot_product / heff_norm2
        new_spin = np.empty(3)
        new_spin[0] = factor * hx - old_spin[0]
        new_spin[1] = factor * hy - old_spin[1]
        new_spin[2] = factor * hz - old_spin[2]
        
        # 归一化 (防止浮点数精度随着多次反射漂移)
        norm = np.sqrt(new_spin[0]**2 + new_spin[1]**2 + new_spin[2]**2)
        new_spin[0] /= norm
        new_spin[1] /= norm
        new_spin[2] /= norm

        # 4. 判定 (兼容 Kz)
        # 如果单离子各向异性 Kz == 0，此时能量严格守恒，100% 接受。
        # 如果你未来把 Kz 设置为非 0 值，由于各向异性是 S_z 的平方项（非线性），
        # 反射后总能量会有微小变化，因此补一个 Metropolis 判定。
        if Kz != 0.0:
            dE = Kz * (new_spin[2]**2 - old_spin[2]**2)
            if dE > 0 and np.random.rand() > np.exp(-dE / T):
                continue # 拒绝
                
        # 接受更新
        spins[x, y, alpha] = new_spin

@njit
def calc_total_energy_fast(spins, Bz):
    E_tot = 0.0
    for x in range(L):
        for y in range(L):
            for alpha in range(3):
                spin_val = spins[x, y, alpha]
                # 【修复1】补充上 Kz 项的能量贡献
                E_tot += -Bz * spin_val[2] + Kz * (spin_val[2]**2) 
                E_bond = 0.0
                for i in range(4):
                    nx, ny, n_alpha, dmi_sign = neighbor_table[x, y, alpha, i]
                    sj = spins[nx, ny, n_alpha]
                    
                    # 【修复2】提取面内 DMI 矢量
                    dx = dxy_vectors[alpha, i, 0]
                    dy = dxy_vectors[alpha, i, 1]
                    
                    E_bond += J_spin * (spin_val[0]*sj[0] + spin_val[1]*sj[1] + spin_val[2]*sj[2])
                    E_bond += dmi_sign * Dz * (spin_val[0]*sj[1] - spin_val[1]*sj[0])
                    # 【修复3】补充面内 DMI (Dxy) 的能量贡献
                    E_bond += Dxy * (dx * (spin_val[1]*sj[2] - spin_val[2]*sj[1]) + 
                                     dy * (spin_val[2]*sj[0] - spin_val[0]*sj[2]))
                E_tot += 0.5 * E_bond  # 0.5 用于消除双重计算，逻辑正确
    return E_tot
@njit
def mc_sweep_fast(spins, T, Bz):
    accept_count = 0
    for _ in range(N_spins):
        x, y, alpha = np.random.randint(0, L), np.random.randint(0, L), np.random.randint(0, 3)
        old_spin = spins[x, y, alpha].copy()
        new_spin = old_spin + np.random.randn(3) * 0.2
        new_spin = new_spin / np.sqrt(new_spin[0]**2 + new_spin[1]**2 + new_spin[2]**2)
        dE = calc_local_energy_fast(x, y, alpha, new_spin, spins, Bz) - calc_local_energy_fast(x, y, alpha, old_spin, spins, Bz)
        if dE < 0 or np.random.rand() < np.exp(-dE / T): 
            spins[x, y, alpha] = new_spin
            accept_count += 1
    return accept_count

@njit
def calc_instant_topo_charge_fast(spins, L):
    """
    计算当前 MC 步的瞬时拓扑电荷密度
    注意：MC中的 spins 已经是严格的单位矢量，无需再归一化
    """
    total_Q = 0.0
    for x in range(L):
        for y in range(L):
            sA, sB, sC = spins[x, y, 0], spins[x, y, 1], spins[x, y, 2]
            
            # 计算向上的三角形 (Upward triangle)
            num_up = sA[0]*(sB[1]*sC[2] - sB[2]*sC[1]) - sA[1]*(sB[0]*sC[2] - sB[2]*sC[0]) + sA[2]*(sB[0]*sC[1] - sB[1]*sC[0])
            den_up = 1.0 + sA[0]*sB[0]+sA[1]*sB[1]+sA[2]*sB[2] + \
                           sB[0]*sC[0]+sB[1]*sC[1]+sB[2]*sC[2] + \
                           sC[0]*sA[0]+sC[1]*sA[1]+sC[2]*sA[2]
            omega_up = 2.0 * np.arctan2(num_up, den_up)
            
            # 提取相邻胞计算向下的三角形 (Downward triangle)
            sB_down, sC_down = spins[(x-1)%L, y, 1], spins[x, (y-1)%L, 2]
            num_dn = sA[0]*(sB_down[1]*sC_down[2] - sB_down[2]*sC_down[1]) - sA[1]*(sB_down[0]*sC_down[2] - sB_down[2]*sC_down[0]) + sA[2]*(sB_down[0]*sC_down[1] - sB_down[1]*sC_down[0])
            den_dn = 1.0 + sA[0]*sB_down[0]+sA[1]*sB_down[1]+sA[2]*sB_down[2] + \
                           sB_down[0]*sC_down[0]+sB_down[1]*sC_down[1]+sB_down[2]*sC_down[2] + \
                           sC_down[0]*sA[0]+sC_down[1]*sA[1]+sC_down[2]*sA[2]
            omega_dn = 2.0 * np.arctan2(num_dn, den_dn)
            
            total_Q += (omega_up + omega_dn)
            
    # 返回单位原胞的瞬时拓扑电荷密度
    return total_Q / (4.0 * np.pi * L * L)

@njit
def calc_total_chirality_fast(spins):
    chi = 0.0
    mz = 0.0
    for x in range(L):
        for y in range(L):
            sA, sB, sC = spins[x, y, 0], spins[x, y, 1], spins[x, y, 2]
            mz += sA[2] + sB[2] + sC[2]
            chi += sA[0]*(sB[1]*sC[2] - sB[2]*sC[1]) - sA[1]*(sB[0]*sC[2] - sB[2]*sC[0]) + sA[2]*(sB[0]*sC[1] - sB[1]*sC[0])
            sB_down, sC_down = spins[(x-1)%L, y, 1], spins[x, (y-1)%L, 2]
            chi += sA[0]*(sB_down[1]*sC_down[2] - sB_down[2]*sC_down[1]) - sA[1]*(sB_down[0]*sC_down[2] - sB_down[2]*sC_down[0]) + sA[2]*(sB_down[0]*sC_down[1] - sB_down[1]*sC_down[0])
    return chi / (2 * L * L), mz / N_spins

@njit
def calc_bg_chirality_fast(M_avg):
    chi_bg = 0.0
    for x in range(L):
        for y in range(L):
            mA, mB, mC = M_avg[x, y, 0], M_avg[x, y, 1], M_avg[x, y, 2]
            chi_bg += mA[0]*(mB[1]*mC[2] - mB[2]*mC[1]) - mA[1]*(mB[0]*mC[2] - mB[2]*mC[0]) + mA[2]*(mB[0]*mC[1] - mB[1]*mC[0])
            mB_down, mC_down = M_avg[(x-1)%L, y, 1], M_avg[x, (y-1)%L, 2]
            chi_bg += mA[0]*(mB_down[1]*mC_down[2] - mB_down[2]*mC_down[1]) - mA[1]*(mB_down[0]*mC_down[2] - mB_down[2]*mC_down[0]) + mA[2]*(mB_down[0]*mC_down[1] - mB_down[1]*mC_down[0])
    return chi_bg / (2 * L * L)

@njit
def calc_pure_fluctuation_chirality(spins, M_avg):
    chi_fluct = 0.0
    for x in range(L):
        for y in range(L):
            sA, sB, sC = spins[x, y, 0], spins[x, y, 1], spins[x, y, 2]
            mA, mB, mC = M_avg[x, y, 0], M_avg[x, y, 1], M_avg[x, y, 2]
            dsA = sA - mA; dsB = sB - mB; dsC = sC - mC
            chi_fluct += dsA[0]*(dsB[1]*dsC[2] - dsB[2]*dsC[1]) - \
                         dsA[1]*(dsB[0]*dsC[2] - dsB[2]*dsC[0]) + \
                         dsA[2]*(dsB[0]*dsC[1] - dsB[1]*dsC[0])
            sB_down, sC_down = spins[(x-1)%L, y, 1], spins[x, (y-1)%L, 2]
            mB_down, mC_down = M_avg[(x-1)%L, y, 1], M_avg[x, (y-1)%L, 2]
            dsB_down = sB_down - mB_down; dsC_down = sC_down - mC_down
            chi_fluct += dsA[0]*(dsB_down[1]*dsC_down[2] - dsB_down[2]*dsC_down[1]) - \
                         dsA[1]*(dsB_down[0]*dsC_down[2] - dsB_down[2]*dsC_down[0]) + \
                         dsA[2]*(dsB_down[0]*dsC_down[1] - dsB_down[1]*dsC_down[0])
    return chi_fluct / (2 * L * L)

# ==========================================
# 4. 单个 Bz 扫描任务函数 (供多进程调用)
# ==========================================
def simulate_single_Bz(args):
    i, Bz_val = args
    
    # 【重要】确保 Numpy 和 Numba 的随机种子在各个进程中完全独立
    process_seed = int(time.time() * 1000) % 123456789 + i * 999
    np.random.seed(process_seed)
    set_numba_seed(process_seed)
    
    # 局部存储这一个 Bz 对应的整行数据
    row_chi_total = np.zeros(len(T_arr))
    row_chi_bg = np.zeros(len(T_arr))
    row_delta_chi = np.zeros(len(T_arr))
    row_mz = np.zeros(len(T_arr))
    row_mlocal = np.zeros(len(T_arr))
    row_cv = np.zeros(len(T_arr))
    row_skew_chi = np.zeros(len(T_arr))
    row_accept = np.zeros(len(T_arr))
    row_topo_charge = np.zeros(len(T_arr))  # <--- 【新增这一行】
    row_spin_config = np.zeros((len(T_arr), L, L, 3, 3))

    print(f"▶️ 进程启动: 扫描 Bz={Bz_val:>5.2f} (索引 {i})")

    spins = np.random.randn(L, L, 3, 3)
    for x in range(L):
        for y in range(L):
            for alpha in range(3): 
                spins[x,y,alpha] /= np.linalg.norm(spins[x,y,alpha])
                
    for j, T_val in enumerate(T_arr):
        for _ in range(BURN_IN_SWEEPS): 
            mc_sweep_fast(spins, T_val, Bz_val)
            for _ in range(3): 
                over_relaxation_sweep(spins, Bz_val, T_val)           
        # 在 MEASURE_SWEEPS 循环前定义累加器
        topo_charge_sum = 0.0
        chi_total_sum, mz_sum = 0.0, 0.0
        E_sum, E2_sum = 0.0, 0.0 
        sum_spins_FC = np.zeros((L, L, 3, 3)) 
        total_accepted_moves = 0

        for _ in range(MEASURE_SWEEPS):
            acc = mc_sweep_fast(spins, T_val, Bz_val)
            total_accepted_moves += acc
            # 3 次微正则过弛豫更新 (疯狂混合状态，脱离局域极小值)
            for _ in range(3): 
                over_relaxation_sweep(spins, Bz_val, T_val)

            c_tot, m = calc_total_chirality_fast(spins)
            E = calc_total_energy_fast(spins, Bz_val)
            
            chi_total_sum += c_tot; mz_sum += m
            E_sum += E; E2_sum += E**2
            sum_spins_FC += spins 
            # 【新增】计算并累加这一步的瞬时拓扑荷
            q_inst = calc_instant_topo_charge_fast(spins, L)
            topo_charge_sum += q_inst

        row_accept[j] = total_accepted_moves / (N_spins * MEASURE_SWEEPS)
        row_chi_total[j] = chi_total_sum / MEASURE_SWEEPS
        row_mz[j] = mz_sum / MEASURE_SWEEPS
        
        E_avg = E_sum / MEASURE_SWEEPS
        E2_avg = E2_sum / MEASURE_SWEEPS
        row_cv[j] = (E2_avg - E_avg**2) / (N_spins * T_val**2)
        
        spins_avg = sum_spins_FC / MEASURE_SWEEPS
        row_spin_config[j] = spins_avg
        row_mlocal[j] = np.mean(np.linalg.norm(spins_avg, axis=-1))
        # 在循环结束后，求平均值并存入对应数组
        row_topo_charge[j] = topo_charge_sum / MEASURE_SWEEPS

        EXTRA_SWEEPS = 1000 
        sum_skew_chi = 0.0
        for _ in range(EXTRA_SWEEPS):
            mc_sweep_fast(spins, T_val, Bz_val)
            sum_skew_chi += calc_pure_fluctuation_chirality(spins, spins_avg)
        row_skew_chi[j] = (sum_skew_chi / EXTRA_SWEEPS)

        c_bg = calc_bg_chirality_fast(spins_avg)
        row_chi_bg[j] = c_bg
        
        raw_total_avg = chi_total_sum / MEASURE_SWEEPS
        row_delta_chi[j] = raw_total_avg - c_bg
        
    print(f"✅ 完成: Bz={Bz_val:>5.2f}")
    return (i, row_chi_total, row_chi_bg, row_delta_chi, row_mz, row_mlocal, row_cv, row_skew_chi, row_accept, row_topo_charge, row_spin_config)


# ==========================================
# 5. 执行主程序 (多进程调度、合并保存、画图)
# ==========================================
if __name__ == '__main__':
    # 初始化主进程的相图数据存储矩阵
    phase_chi_total = np.zeros((len(Bz_arr), len(T_arr))) 
    phase_chi_bg = np.zeros((len(Bz_arr), len(T_arr)))    
    phase_delta_chi = np.zeros((len(Bz_arr), len(T_arr))) 
    phase_mz = np.zeros((len(Bz_arr), len(T_arr)))       
    phase_mlocal = np.zeros((len(Bz_arr), len(T_arr)))   
    phase_cv = np.zeros((len(Bz_arr), len(T_arr)))       
    phase_skew_chi = np.zeros((len(Bz_arr), len(T_arr))) 
    phase_accept = np.zeros((len(Bz_arr), len(T_arr))) 
    phase_topo_charge = np.zeros((len(Bz_arr), len(T_arr))) # <--- 【新增这一行】
    phase_spin_config = np.zeros((len(Bz_arr), len(T_arr), L, L, 3, 3))

    start_time = time.time()
    
    # 自动获取 CPU 核心数，预留 1 个核心以防系统卡顿
    max_workers = max(1, os.cpu_count() - 1)
    print(f"🔄 开始分配计算任务，启用 {max_workers} 个 CPU 核心...")

    # 构建任务列表
    tasks = [(i, Bz_val) for i, Bz_val in enumerate(Bz_arr)]

    # 启动进程池进行并行计算
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(simulate_single_Bz, tasks))

    # 汇总各个子进程的返回数据
    for res in results:
        i, r_chi, r_bg, r_delta, r_mz, r_mlocal, r_cv, r_skew, r_accept, r_topo, r_config = res
        phase_chi_total[i, :] = r_chi
        phase_chi_bg[i, :]    = r_bg
        phase_delta_chi[i, :] = r_delta
        phase_mz[i, :]        = r_mz
        phase_mlocal[i, :]    = r_mlocal
        phase_cv[i, :]        = r_cv
        phase_skew_chi[i, :]  = r_skew
        phase_accept[i, :]    = r_accept
        phase_topo_charge[i, :] = r_topo   # <--- 【新增这一行】
        phase_spin_config[i, :] = r_config

    print(f"\n✅ 所有扫描任务已完成！总耗时: {(time.time() - start_time) / 60:.2f} 分钟")

    # ==========================================
    # 6. 提取相边界、保存数据与绘制终极五联图
    # ==========================================
    boundary_T = np.zeros(len(Bz_arr))
    for i in range(len(Bz_arr)):
        idx_max_cv = np.argmax(phase_cv[i, :])
        boundary_T[i] = T_arr[idx_max_cv]

    # 保存计算结果
    np.savez(SAVE_FILENAME, 
             T_arr=T_arr, 
             Bz_arr=Bz_arr, 
             phase_accept=phase_accept,
             phase_chi_total=phase_chi_total,
             phase_chi_bg=phase_chi_bg,
             phase_delta_chi=phase_delta_chi,
             phase_mz=phase_mz,
             phase_mlocal=phase_mlocal,
             phase_cv=phase_cv,
             phase_skew_chi=phase_skew_chi,
             phase_topo_charge=phase_topo_charge,  # <--- 【新增这一行】
             phase_spin_config=phase_spin_config,
             boundary_T=boundary_T)

    print(f"💾 数据已安全保存至本地文件: {os.path.abspath(SAVE_FILENAME)}")

    # 数据翻转适配绘图习惯
    T_arr_plot = T_arr[::-1]
    phase_chi_total_plot = phase_chi_total[:, ::-1]
    phase_mlocal_plot = phase_mlocal[:, ::-1]
    phase_delta_chi_plot = phase_delta_chi[:, ::-1]
    phase_cv_plot = phase_cv[:, ::-1]
    
    # 这里将 mz 也放进来以保留你的 5 幅子图设计
    phase_mz_plot = phase_mz[:, ::-1]

    X, Y = np.meshgrid(T_arr_plot, Bz_arr)

    # 创建 1x5 子图画布
    fig, axes = plt.subplots(1, 5, figsize=(30, 6))

    plot_configs = [
        (phase_chi_total_plot, 'magma', 'Total $\chi_{total}$', '1. Total Chirality (Observed)'),
        (phase_mlocal_plot, 'plasma', 'Local Magnetization $M_{local}$', 'Magnetic Order: AFM vs PM'),
        (phase_delta_chi_plot, 'Blues_r', 'Fluctuation $\Delta\chi$', '3. Spin Deviation (Skew $\sigma_{xy}$)'), 
        (phase_mz_plot, 'viridis', '$|M_z|$', '4. Z-Polarization'), # 恢复 Z向磁化图，凑齐五联图
        (phase_cv_plot, 'inferno', '$C_v$', '5. Phase Boundary (Tc)')
    ]

    for ax, (data, cmap_name, cbar_label, title) in zip(axes, plot_configs):
        c = ax.pcolormesh(X, Y, data, cmap=cmap_name, shading='auto')
        ax.contour(X, Y, data, levels=6, colors='white', linewidths=0.3, alpha=0.3)
        
        # 叠加比热容提取出的相变分界线
        ax.plot(boundary_T, Bz_arr, 'w--', linewidth=2.5, alpha=0.9, label='Phase Boundary ($T_c$)')
        
        fig.colorbar(c, ax=ax, label=cbar_label)
        ax.set_title(title, fontsize=15, fontweight='bold')
        ax.set_xlabel('Temperature $T$', fontsize=13)
        if ax == axes[0]: ax.set_ylabel('Magnetic Field $B_z$', fontsize=13)
        ax.legend(loc='upper left', fontsize=11)

    plt.tight_layout()
    plt.show()
