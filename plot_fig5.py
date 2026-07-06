import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
import matplotlib.ticker as ticker
import matplotlib.patheffects as pe
import os


# ==========================================
# 0. 全局样式设置：完全照旧程序
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
# 1. 文件与固定参数
# ==========================================
data_file = "Transport_Old48_Tiled12_Nk6_mu_sweep_eta0p05.npz"
boundary_file = "phase_boundaries_old48_article.npz"
output_dir = "Transport_Plots_Final_mu_m4p0_old_style"

MU_VALUE = -4.0
sigma_val = 1.0

os.makedirs(output_dir, exist_ok=True)


# ==========================================
# 2. 加载数据
# ==========================================
print("📂 正在加载数据文件...")

try:
    data = np.load(data_file, allow_pickle=True)
    T_arr = np.asarray(data["T_arr"], dtype=float)
    Bz_arr = np.asarray(data["Bz_arr"], dtype=float)
    mu_values = np.asarray(data["mu_values"], dtype=float)
except FileNotFoundError:
    print(f"⚠️ 找不到文件 {data_file}，请检查路径。")
    raise SystemExit


# 选择 mu
mu_idx = int(np.argmin(np.abs(mu_values - MU_VALUE)))
mu_actual = float(mu_values[mu_idx])

if not np.isclose(mu_actual, MU_VALUE, rtol=0.0, atol=1e-10):
    available = ", ".join(f"{mu:g}" for mu in mu_values)
    raise ValueError(
        f"Requested mu={MU_VALUE:g}, but nearest available mu is {mu_actual:g}. "
        f"Available values: {available}"
    )

print(f"✅ 使用 mu/t = {mu_actual:g}")


# 读取相界数据
if os.path.exists(boundary_file):
    boundaries = np.load(boundary_file, allow_pickle=True)
else:
    boundaries = None
    print(f"⚠️ 未找到 {boundary_file}，将不绘制相界线。")


# ==========================================
# 3. 数据处理：完全照旧程序的 T 轴翻转逻辑
# ==========================================
T_plot = T_arr[::-1]
X, Y = np.meshgrid(T_plot, Bz_arr)


def get_mu_field(key):
    """
    新数据维度假设为 [mu, B, T]。
    这里取指定 mu 后，变成 [B, T]。
    """
    if key not in data:
        raise KeyError(f"数据文件中没有字段: {key}")
    return np.asarray(data[key][mu_idx], dtype=float)


# 提取基础输运分量，并照旧程序做 T 方向翻转
sig_int_raw = get_mu_field("phase_sigma_int")[:, ::-1]
sig_ext_raw = get_mu_field("phase_sigma_ext")[:, ::-1]
alp_int_raw = get_mu_field("phase_alpha_int")[:, ::-1]
alp_ext_raw = get_mu_field("phase_alpha_ext")[:, ::-1]


# 平滑：完全照旧程序
sig_int = ndimage.gaussian_filter(sig_int_raw, sigma=sigma_val)
sig_ext = ndimage.gaussian_filter(sig_ext_raw, sigma=sigma_val)
alp_int = ndimage.gaussian_filter(alp_int_raw, sigma=sigma_val)
alp_ext = ndimage.gaussian_filter(alp_ext_raw, sigma=sigma_val)


# total：优先使用数据文件中的 total；如果没有，就用 int + ext
if "phase_sigma_total" in data:
    sig_tot = ndimage.gaussian_filter(
        get_mu_field("phase_sigma_total")[:, ::-1],
        sigma=sigma_val
    )
else:
    sig_tot = sig_int + sig_ext

if "phase_alpha_total" in data:
    alp_tot = ndimage.gaussian_filter(
        get_mu_field("phase_alpha_total")[:, ::-1],
        sigma=sigma_val
    )
else:
    alp_tot = alp_int + alp_ext


# ==========================================
# 4. 定义 6 张图：顺序、height 完全照旧程序
# ==========================================
plot_configs = [
    # ---- ANE 分量 ----
    {
        'data': alp_tot,
        'label': r'$\alpha_{\mathrm{xy}}^{\mathrm{tot}}$ (A/K)',
        'name': 'ANE_Total',
        'legend_phases': [],
        'height': 6.5,
    },

    {
        'data': alp_int,
        'label': r'$\alpha_{\mathrm{xy}}^{\mathrm{int}}$ (A/K)',
        'name': 'ANE_Intrinsic',
        'legend_phases': ['fluctuating_chi'],
        'height': 7.7,
    },

    {
        'data': alp_ext,
        'label': r'$\alpha_{\mathrm{xy}}^{\mathrm{ext}}$ (A/K)',
        'name': 'ANE_Extrinsic',
        'legend_phases': [],
        'height': 6.5,
    },

    # ---- AHC 分量 ----
    {
        'data': sig_tot,
        'label': r'$\sigma_{\mathrm{xy}}^{\mathrm{tot}}$ (S)',
        'name': 'AHC_Total',
        'legend_phases': [],
        'height': 6.5,
    },

    {
        'data': sig_int,
        'label': r'$\sigma_{\mathrm{xy}}^{\mathrm{int}}$ (S)',
        'name': 'AHC_Intrinsic',
        'legend_phases': ['phase_1'],
        'height': 7.7,
    },

    {
        'data': sig_ext,
        'label': r'$\sigma_{\mathrm{xy}}^{\mathrm{ext}}$ (S)',
        'name': 'AHC_Extrinsic',
        'legend_phases': [],
        'height': 6.5,
    },
]


# ==========================================
# 5. 相界线配置：完全照旧程序
# ==========================================
phase_meta = {
    'phase_1': {
        'color': 'black',
        'label': 'SkX',
        'ls': '-',
        'lw': 2.5,
    },
    'fluctuating_chi': {
        'color': 'magenta',
        'label': 'FC',
        'ls': '--',
        'lw': 2.5,
    },
}


# ==========================================
# 6. Colorbar 格式：必须在 subplots_adjust 后调用
# ==========================================
def format_colorbar_final(cb, label):
    """
    固定 colorbar 样式。
    科学计数法不用 Matplotlib 自动 offset text，而是手动画。
    这样位置不会被 subplots_adjust / savefig 改掉。
    """
    vmin, vmax = cb.mappable.get_clim()
    scale_ref = max(abs(vmin), abs(vmax))

    if scale_ref > 0:
        exponent = int(np.floor(np.log10(scale_ref)))
    else:
        exponent = 0

    use_sci = exponent <= -1 or exponent >= 1

    if use_sci:
        scale = 10.0 ** exponent

        def scaled_formatter(x, pos):
            value = x / scale
            if abs(value) < 1e-14:
                value = 0.0
            return f"{value:g}"

        cb.ax.yaxis.set_major_formatter(ticker.FuncFormatter(scaled_formatter))
        cb.ax.yaxis.get_offset_text().set_visible(False)

        # 手动画科学计数法
        # x 越大越靠右；y 越大越靠上
        cb.ax.text(
            1.12,
            1.035,
            rf"$\times 10^{{{exponent}}}$",
            transform=cb.ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=25,
            clip_on=False,
        )

    else:
        formatter = ticker.ScalarFormatter(useMathText=True)
        formatter.set_scientific(False)
        cb.ax.yaxis.set_major_formatter(formatter)
        cb.ax.yaxis.get_offset_text().set_visible(False)

    cb.set_label(
        label=label,
        size=40,
        labelpad=5,
    )

    cb.ax.tick_params(
        labelsize=25,
        pad=2,
    )

    cb.ax.yaxis.set_label_position("right")
    cb.ax.yaxis.tick_right()


# ==========================================
# 7. 批量绘图：完全照旧程序
# ==========================================
print("🎨 开始生成 6 张旧版布局相图...")

for cfg in plot_configs:

    # 宽度、高度完全照旧程序
    fig, ax = plt.subplots(figsize=(8, cfg['height']))

    # 色标范围：完全照旧程序，用最大绝对值
    abs_max = np.max(np.abs(cfg['data']))
    if abs_max == 0:
        abs_max = 1e-10

    c = ax.pcolormesh(
        X,
        Y,
        cfg['data'],
        cmap='seismic',
        shading='auto',
        vmin=-abs_max,
        vmax=abs_max,
        rasterized=True,
    )

    # ----------------------------------------------------
    # 相界线：完全照旧程序，黑/紫线都加白色描边
    # ----------------------------------------------------
    if boundaries is not None:
        for p_key, style in phase_meta.items():
            if p_key in boundaries:
                for i, seg in enumerate(boundaries[p_key]):
                    seg = np.asarray(seg, dtype=float)
                    if seg.size == 0:
                        continue

                    lbl = style['label'] if (
                        i == 0 and p_key in cfg['legend_phases']
                    ) else None

                    line, = ax.plot(
                        seg[:, 0],
                        seg[:, 1],
                        color=style['color'],
                        ls=style['ls'],
                        lw=style['lw'],
                        label=lbl,
                    )

                    line.set_path_effects([
                        pe.Stroke(
                            linewidth=style['lw'] + 3.0,
                            foreground='white',
                        ),
                        pe.Normal(),
                    ])

    # ----------------------------------------------------
    # Colorbar：只创建，不在这里设置科学计数法
    # ----------------------------------------------------
    cb = fig.colorbar(
        c,
        ax=ax,
        fraction=0.046,
        pad=0.04,
    )

    # ----------------------------------------------------
    # 坐标轴设置：完全照旧程序
    # ----------------------------------------------------
    ax.set_xscale('log')
    ax.set_xlabel(r'$T/J_{\mathrm{H}}$', labelpad=10)
    ax.set_ylabel(r'$B/J_{\mathrm{H}}$', labelpad=10)
    ax.set_yticks([0, 0.05, 0.1, 0.15, 0.2])
    ax.tick_params(axis='both', which='both', pad=10)

    min_t = T_plot[T_plot > 0].min() if T_plot.min() <= 0 else T_plot.min()
    ax.set_xlim(min_t, T_plot.max())
    ax.set_ylim(0, 0.2)

    # ----------------------------------------------------
    # 边距与图例：完全照旧程序
    # ----------------------------------------------------
    if cfg['legend_phases']:
        ax.legend(
            loc='lower center',
            bbox_to_anchor=(0.5, 1.02),
            fontsize=40,
            frameon=False,
        )

        fig.subplots_adjust(
            left=0.2,
            right=0.8,
            bottom=0.2,
            top=0.83,
        )
    else:
        fig.subplots_adjust(
            left=0.2,
            right=0.8,
            bottom=0.2,
            top=0.83,
        )

    # ----------------------------------------------------
    # 关键：必须在 subplots_adjust 之后格式化 colorbar
    # ----------------------------------------------------
    format_colorbar_final(cb, cfg['label'])

    # ----------------------------------------------------
    # 保存：完全照旧程序，不加 bbox_inches，不加 pad_inches
    # ----------------------------------------------------
    mu_tag = f"{mu_actual:.1f}".replace("-", "m").replace(".", "p")
    save_path = os.path.join(
        output_dir,
        f"{cfg['name']}_mu{mu_tag}_StrokeLine.png",
    )

    plt.savefig(save_path, dpi=300)
    plt.close(fig)

    print(f"✅ 已保存: {save_path}")


# ==========================================
# 8. 保存参数记录
# ==========================================
meta_path = os.path.join(output_dir, "plot_parameters.txt")
with open(meta_path, "w", encoding="utf-8") as f:
    f.write(f"data_file = {data_file}\n")
    f.write(f"boundary_file = {boundary_file}\n")
    f.write(f"mu/t = {mu_actual:g}\n")
    f.write(f"smooth_sigma = {sigma_val:g}\n")
    f.write("layout = old_single_panel_style\n")
    f.write("figsize = (8, cfg['height'])\n")
    f.write("subplots_adjust = left=0.2, right=0.8, bottom=0.2, top=0.83\n")
    f.write("colorbar = fraction=0.046, pad=0.04\n")
    f.write("colorbar_labelpad = 5\n")
    f.write("scientific_text_position = x=1.12, y=1.035\n")
    f.write("savefig = plt.savefig(save_path, dpi=300)\n")

print("\n🚀 6 张旧版布局输运图全部生成完成！")