"""Illustrate 1111-slot CFM grid vs OESCL-active slots (sum ≈ 878).

Same geometry as ``compute_throughput_cfm._build_multiband_channel_grid``.
Run: python plot_oesc_grid_sketch.py  → writes PNG under data/figures/
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
except ImportError as e:
    raise SystemExit("matplotlib required: pip install matplotlib") from e

c = 299792458.0
CH_SPACING_HZ = 50e9
NUM_CHANNELS_TOTAL = 1111
REF_LAMBDA_M = 1419.4e-9

_BAND_BOUNDS = {
    "O": (1260e-9, 1358e-9),
    "E": (1405e-9, 1464e-9),
    "S": (1470e-9, 1526e-9),
    "C": (1530e-9, 1566e-9),
    "L": (1573e-9, 1625e-9),
}


def main() -> None:
    chs = np.arange(NUM_CHANNELS_TOTAL) - (NUM_CHANNELS_TOTAL - 1) / 2
    ch_lambda = c / (chs * CH_SPACING_HZ + c / REF_LAMBDA_M)
    wl_nm = ch_lambda * 1e9

    active_bands = ["O", "E", "S", "C", "L"]
    band_masks: dict[str, np.ndarray] = {}
    for letter, (wl_lo, wl_hi) in _BAND_BOUNDS.items():
        band_masks[letter] = (ch_lambda >= wl_lo) & (ch_lambda <= wl_hi)

    channel_idx = np.zeros(NUM_CHANNELS_TOTAL, dtype=bool)
    for b in active_bands:
        channel_idx |= band_masks[b]

    n_active = int(channel_idx.sum())
    n_gap = NUM_CHANNELS_TOTAL - n_active

    out_dir = Path(__file__).resolve().parent / "data" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / "oesc_grid_1111_slots_vs_active.png"

    slots = np.arange(NUM_CHANNELS_TOTAL)

    fig, axes = plt.subplots(3, 1, figsize=(14, 7), gridspec_kw={"height_ratios": [1.2, 1.2, 2.5]})

    # --- Row 1: slot index (0…1110): green = lit for OESCL, gray = not in any band window ---
    ax = axes[0]
    green = np.array([0.15, 0.65, 0.25])
    gray = np.array([0.78, 0.78, 0.78])
    rgb = np.zeros((1, NUM_CHANNELS_TOTAL, 3))
    rgb[0, :, :] = gray
    rgb[0, channel_idx, :] = green
    ax.imshow(
        rgb,
        aspect="auto",
        extent=[-0.5, NUM_CHANNELS_TOTAL - 0.5, -0.5, 0.5],
        interpolation="nearest",
    )
    ax.set_yticks([])
    ax.set_xlim(-0.5, NUM_CHANNELS_TOTAL - 0.5)
    ax.set_xlabel("slot index (total grid = 1111 slots)")
    ax.set_title(
        f"Along fixed slot index: OESCL-active = {n_active} (green), "
        f"inter-band / unused slots = {n_gap} (not green)"
    )

    # --- Row 2: same, x = wavelength (nm) — shows five clusters + gaps ---
    ax = axes[1]
    order = np.argsort(wl_nm)
    wl_sorted = wl_nm[order]
    act_sorted = channel_idx[order]
    rgb2 = np.zeros((1, NUM_CHANNELS_TOTAL, 3))
    rgb2[0, :, :] = gray
    rgb2[0, np.flatnonzero(act_sorted), :] = green
    ax.imshow(
        rgb2,
        aspect="auto",
        extent=[wl_sorted[0], wl_sorted[-1], -0.5, 0.5],
        interpolation="nearest",
    )
    ax.set_yticks([])
    ax.set_xlabel("wavelength λ (nm), sorted low → high")
    ax.set_title("Same 1111 slots mapped to wavelength (sorted): five bands + dark gaps between them")

    # --- Row 3: stem-style — λ vs slot (unsorted index shows grid order is not “middle chunk”) ---
    ax = axes[2]
    colors = np.tile(gray, (NUM_CHANNELS_TOTAL, 1))
    colors[channel_idx] = green
    ax.scatter(
        wl_nm,
        slots,
        c=colors,
        s=8,
        marker="s",
        linewidths=0,
        rasterized=True,
    )
    for letter, (lo, hi) in _BAND_BOUNDS.items():
        ax.axvspan(lo * 1e9, hi * 1e9, alpha=0.06, zorder=0)
    ax.set_xlabel("wavelength λ (nm)")
    ax.set_ylabel("slot index")
    ax.set_title(
        "Each point is one slot: color = OESCL-active or not. "
        "878 is not “the middle 878 indices” — it is every slot whose λ falls in O+E+S+C+L windows."
    )
    ax.grid(True, alpha=0.25)
    leg = [
        mpatches.Patch(color=green, label=f"OESCL active ({n_active})"),
        mpatches.Patch(color=gray, label=f"inactive on grid ({n_gap})"),
    ]
    ax.legend(handles=leg, loc="upper left")

    fig.suptitle(
        "CFM grid: NUM_CHANNELS_TOTAL = 1111 fixed slots; OESCL uses union of band windows (≈878 channels)",
        fontsize=11,
        y=1.02,
    )
    plt.tight_layout()
    fig.savefig(png_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {png_path}")
    print(f"  total_slots={NUM_CHANNELS_TOTAL}, oescl_active={n_active}, inactive={n_gap}")


if __name__ == "__main__":
    main()
