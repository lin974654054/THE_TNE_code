import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects # 【新增】用于文字描边特效

# ==========================================
# 1. 全局学术绘图风格设置 (统一 PRB 标准与单栏大字号)
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
    'xtick.top': True,              
    'ytick.right': True,            
    'figure.dpi': 300
})

# ==========================================
# 2. 准备画布 (采用宽度 7 的单栏黄金尺寸)
# ==========================================
fig, ax = plt.subplots(figsize=(8, 6.5)) 

# 读取数据...
try:
    boundary_data = np.load("phase_boundaries.npz", allow_pickle=True)
except FileNotFoundError:
    print("⚠️ 找不到 'phase_boundaries.npz' 文件，请确保路径正确。")
    boundary_data = {}

# ==========================================
# 3. 高级感调色盘与填充逻辑
# ==========================================
phase_styles = {
    'phase_1': {'color': '#D62728'}, # SkX                 
    'phase_2': {'color': '#2CA02C'}, # Helical             
    'phase_3': {'color': '#1F77B4'}, # FP         
    'phase_4': {'color': '#7F7F7F'}, # Pm       
    'fluctuating_chi': {'color': '#9467BD'} # FC
}

xmin, xmax = 1e-3, 1.0
ymin, ymax = 0.0, 0.2

# 背景色：极浅黄，zorder=0 在最底层 (不再需要 label 参数)
ax.fill_between([xmin, xmax], ymin, ymax, color='#FFF8D6', alpha=1.0, zorder=0)

# 循环画其他的相（在黄色的上层 zorder=1）
for phase_name, style in phase_styles.items():
    if phase_name in boundary_data:
        segments = boundary_data[phase_name]
        for i, seg in enumerate(segments):
            fill_alpha = 1.0
            if phase_name == 'phase_2': 
                corner = np.array([[xmin, ymin]])
                closed_seg = np.vstack([seg, corner])
                ax.fill(closed_seg[:, 0], closed_seg[:, 1], color=style['color'], alpha=fill_alpha, zorder=1)
            elif phase_name == 'phase_3': 
                corner = np.array([[xmin, ymax]])
                closed_seg = np.vstack([seg, corner])
                ax.fill(closed_seg[:, 0], closed_seg[:, 1], color=style['color'], alpha=fill_alpha, zorder=1)
            elif phase_name == 'phase_4': 
                corner = np.array([[xmax, ymin]])
                closed_seg = np.vstack([seg, corner])
                ax.fill(closed_seg[:, 0], closed_seg[:, 1], color=style['color'], alpha=fill_alpha, zorder=1)
            else:
                ax.fill(seg[:, 0], seg[:, 1], color=style['color'], alpha=fill_alpha, zorder=1)

# ==========================================
# 4. 【核心新增】在图内直接添加带描边的文本标签
# ==========================================
# 字典格式: '文本': (x坐标, y坐标)
# 注意 x 轴是对数坐标，所以中点在视觉上会有偏移，我已根据你的原图估算了最佳位置
text_labels = {
    'SkX': (1e-2, 0.08),
    'Helical': (1e-2, 0.015),
    'FP': (1e-2, 0.17),
    'FC': (3e-1, 0.13),
    'Pm': (6e-1, 0.01),
    # 如果还需要标出 Transition，可以取消下行注释
    'TP': (8e-2, 0.1)
}

for text, (x, y) in text_labels.items():
    txt = ax.text(x, y, text, fontsize=40, fontweight='bold', 
                  color='white', ha='center', va='center', zorder=5)
    # 添加黑色描边，确保白字在浅色背景（如浅黄、浅绿）上依然极其清晰
    txt.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='black')])

# ==========================================
# 5. 优化标签、刻度与边距
# ==========================================
ax.set_xscale('log')
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax) 

# 取消坐标轴标签加粗，回归严谨的 PRB 正斜体格式
ax.set_xlabel(r'$T/J_{\mathrm{H}}$', labelpad=10)
ax.set_ylabel(r'$B/J_{\mathrm{H}}$', labelpad=10)

ax.set_yticks([0, 0.05, 0.1, 0.15, 0.2])
ax.tick_params(axis='both', which='both', pad=10)

# 【核心修改】因为完全没有外部图例了，顶部(top)和右侧(right)直接拉满，最大化绘图区！
# left=0.18 保证与热力图左侧完美共线
fig.subplots_adjust(left=0.2, right=0.95, bottom=0.2, top=0.95)

# 保存与展示
save_name = "Phase_Boundary_DirectLabel.png"
fig.savefig(save_name, dpi=300)
print(f"✅ 已成功导出：{save_name}")

plt.show()