from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.colors as colors
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from matplotlib.widgets import Slider

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common import DEFAULT_PHASE_DATA, configure_matplotlib, project_path  # noqa: E402


def generate_kagome_real_space(lattice_size: int) -> np.ndarray:
    a1 = np.array([1.0, 0.0])
    a2 = np.array([0.5, np.sqrt(3) / 2.0])
    basis = np.array(
        [
            [0.0, 0.0],
            [0.5, 0.0],
            [0.25, np.sqrt(3) / 4.0],
        ]
    )

    coords = np.zeros((lattice_size * lattice_size * 3, 2))
    idx = 0
    for x in range(lattice_size):
        for y in range(lattice_size):
            cell_origin = x * a1 + y * a2
            for alpha in range(3):
                coords[idx] = cell_origin + basis[alpha]
                idx += 1
    return coords


def draw_lattice_background(ax, lattice_size: int) -> None:
    a1 = np.array([1.0, 0.0])
    a2 = np.array([0.5, np.sqrt(3) / 2.0])
    for x in range(lattice_size):
        for y in range(lattice_size):
            origin = x * a1 + y * a2
            points = np.array(
                [
                    origin + np.array([0.0, 0.0]),
                    origin + np.array([0.5, 0.0]),
                    origin + np.array([0.25, np.sqrt(3) / 4.0]),
                    origin + np.array([0.0, 0.0]),
                ]
            )
            ax.plot(points[:, 0], points[:, 1], "gray", linewidth=0.5, alpha=0.3)


def interactive_spin_viewer(
    input_path: str | Path = DEFAULT_PHASE_DATA,
    init_bz_index: int = 20,
    init_t_index: int = 50,
    init_scale: float = 0.5,
) -> None:
    configure_matplotlib("paper")
    input_path = project_path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    with np.load(input_path) as data:
        t_arr = data["T_arr"]
        bz_arr = data["Bz_arr"]
        spin_configs = data["phase_spin_config"]

    n_bz, n_t, lattice_size = spin_configs.shape[:3]
    init_bz_index = min(max(init_bz_index, 0), n_bz - 1)
    init_t_index = min(max(init_t_index, 0), n_t - 1)

    coords = generate_kagome_real_space(lattice_size)
    x_coords = coords[:, 0]
    y_coords = coords[:, 1]

    fig, ax = plt.subplots(figsize=(12, 10))
    plt.subplots_adjust(bottom=0.3)

    draw_lattice_background(ax, lattice_size)
    init_spins = spin_configs[init_bz_index, init_t_index].reshape(-1, 3)
    quiver = ax.quiver(
        x_coords,
        y_coords,
        init_spins[:, 0],
        init_spins[:, 1],
        init_spins[:, 2],
        cmap="turbo",
        norm=colors.Normalize(vmin=-1, vmax=1),
        pivot="mid",
        units="xy",
        scale=init_scale,
        width=0.06,
        headwidth=3.5,
        headlength=4,
        alpha=0.9,
    )

    cbar = fig.colorbar(quiver, ax=ax, label=r"$S_z$")
    cbar.set_label(r"$S_z$", fontsize=40)
    cbar.ax.tick_params(labelsize=40)

    title = ax.set_title(
        rf"$\mathrm{{L}}={lattice_size}, "
        rf"\mathrm{{T}}/J_\mathrm{{H}}={t_arr[init_t_index]:.4f}, "
        rf"\mathrm{{B_z}}/J_\mathrm{{H}}={bz_arr[init_bz_index]:.4f}$",
        fontsize=40,
        fontweight="bold",
    )
    ax.set_xlabel("Nx", fontsize=40)
    ax.set_ylabel("Ny", fontsize=40)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", alpha=0.2)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(axis="both", which="major", labelsize=30, length=6, width=1.5)
    ax.tick_params(axis="both", which="minor", labelsize=25, length=3, width=1)

    slider_face = "lightgoldenrodyellow"
    scale_ax = plt.axes([0.15, 0.15, 0.65, 0.03], facecolor=slider_face)
    t_ax = plt.axes([0.15, 0.10, 0.65, 0.03], facecolor=slider_face)
    bz_ax = plt.axes([0.15, 0.05, 0.65, 0.03], facecolor=slider_face)

    scale_slider = Slider(
        scale_ax,
        "Arrow Scale",
        0.0001,
        3.0,
        valinit=init_scale,
        valstep=0.0001,
    )
    t_slider = Slider(t_ax, "T Index", 0, n_t - 1, valinit=init_t_index, valstep=1, valfmt="%0.0f")
    bz_slider = Slider(bz_ax, "Bz Index", 0, n_bz - 1, valinit=init_bz_index, valstep=1, valfmt="%0.0f")

    for slider in (scale_slider, t_slider, bz_slider):
        slider.label.set_size(14)
        slider.valtext.set_size(14)

    def update(_):
        t_idx = int(t_slider.val)
        bz_idx = int(bz_slider.val)
        spins = spin_configs[bz_idx, t_idx].reshape(-1, 3)
        quiver.set_UVC(spins[:, 0], spins[:, 1], spins[:, 2])
        quiver.scale = scale_slider.val
        title.set_text(
            rf"$\mathrm{{L}}={lattice_size}, "
            rf"\mathrm{{T}}/J_\mathrm{{H}}={t_arr[t_idx]:.4f}, "
            rf"\mathrm{{B_z}}/J_\mathrm{{H}}={bz_arr[bz_idx]:.4f}$"
        )
        fig.canvas.draw_idle()

    scale_slider.on_changed(update)
    t_slider.on_changed(update)
    bz_slider.on_changed(update)
    plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive spin configuration viewer.")
    parser.add_argument("--input", default=DEFAULT_PHASE_DATA)
    parser.add_argument("--bz-index", type=int, default=20)
    parser.add_argument("--t-index", type=int, default=50)
    parser.add_argument("--scale", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    interactive_spin_viewer(
        input_path=args.input,
        init_bz_index=args.bz_index,
        init_t_index=args.t_index,
        init_scale=args.scale,
    )


if __name__ == "__main__":
    main()
