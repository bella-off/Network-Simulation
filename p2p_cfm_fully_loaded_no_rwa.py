"""
Point-to-point CFM test (no MongoDB / no RWA).

One fibre route modelled as ``NSPANS`` identical reference spans of length
``SPAN_LENGTH_KM`` (default 10 × 80 km = 800 km). All in-band channels are lit.

Edit the ``USER CONFIGURATION`` block under ``if __name__ == "__main__":``.
"""
from __future__ import annotations

import os
import pathlib
import sys
import time

import matplotlib

# Interactive window by default on Windows (PyCharm / local runs).
if "MPLBACKEND" in os.environ:
    _MPL_BACKEND = os.environ["MPLBACKEND"]
elif os.environ.get("CI", "").lower() in ("1", "true", "yes"):
    _MPL_BACKEND = "Agg"
else:
    _MPL_BACKEND = "TkAgg" if sys.platform == "win32" else "Agg"
matplotlib.use(_MPL_BACKEND)
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import c

_SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import compute_throughput_cfm as cfm


def run_p2p(
    band_selection: str,
    span_length_km: float,
    nspans: int,
    launch_power_dBm: float,
    save_snr_to_disk: bool,
) -> float:
    band_cfg = cfm.BAND_CONFIGS[band_selection]
    active_bands = band_cfg["bands"]

    ch_lambda, channel_idx, ch_idx_oband_full, band_masks, ref_lambda = (
        cfm._build_multiband_channel_grid(active_bands)
    )
    num_active = int(np.sum(channel_idx))
    ch_idx_oband_active = ch_idx_oband_full[channel_idx]
    active_slot_ix = np.flatnonzero(np.asarray(channel_idx))

    total_km = float(nspans) * float(span_length_km)
    print("=" * 60)
    print("CFM point-to-point (fully loaded, no RWA)")
    print("=" * 60)
    print(f"Band              : {band_cfg['name']} ({band_cfg['description']})")
    print(f"Active channels   : {num_active}")
    print(f"Span length       : {span_length_km} km × {nspans} spans = {total_km:.0f} km")
    print(f"Launch power      : {launch_power_dBm} dBm / channel")
    print(f"Ref lambda (disp) : {ref_lambda * 1e9:.1f} nm")
    print("=" * 60)

    P_channel, nf, snr_trx = cfm._build_per_channel_params(
        ch_lambda,
        band_masks,
        active_bands,
        launch_power_dBm=launch_power_dBm,
    )
    setup = cfm._build_setup(
        span_length_km,
        ch_lambda,
        channel_idx,
        P_channel,
        nf,
        snr_trx,
        ref_lambda=ref_lambda,
    )

    mask = np.ones((num_active, 1), dtype=np.int32)
    t0 = time.perf_counter()
    link_nsr, fit_params, eta_spm, eta_xpm, eta_fwm, link_nsr_ase = cfm.calc_NSR_link(
        setup,
        float(nspans),
        mask,
        ch_idx_oband_active,
    )
    dt = time.perf_counter() - t0
    nsr = np.asarray(link_nsr, dtype=np.float64).reshape(-1)
    nsr_ase = np.asarray(link_nsr_ase, dtype=np.float64).reshape(-1)

    valid = np.isfinite(nsr) & (nsr > 0)
    if np.any(valid):
        mean_snr_db = float(np.mean(10.0 * np.log10(1.0 / nsr[valid])))
    else:
        mean_snr_db = float("nan")

    capacity_total = 0.0
    for w in range(num_active):
        slot_row = int(active_slot_ix[w])
        ch_bw_hz = float(np.asarray(setup.ch_bandwidth_ij)[slot_row, 0])
        n = nsr[w]
        if not np.isfinite(n) or n <= 0:
            continue
        snr = 1.0 / n
        capacity_total += 2.0 * ch_bw_hz * np.log2(1.0 + snr)

    print(f"calc_NSR_link wall time : {dt:.2f} s")
    print(f"Mean SNR (occupied)     : {mean_snr_db:.2f} dB")
    print(f"Total Shannon capacity  : {capacity_total / 1e12:.6f} Tbps")
    print(f"Per-channel mean rate   : {(capacity_total / max(num_active, 1)) / 1e9:.3f} Gbps")

    # --- Power evolution plot ---
    j = 0
    chs = setup.channel_idx
    l = float(setup.length_j[j])
    import jax.numpy as jnp
    z = jnp.arange(0, l, 1000.0)
    ch_power_W_i = np.asarray(setup.ch_power_W_ij)[chs, :] * mask
    Aeff_i = jnp.array(setup.spans[j].A_eff_at(np.asarray(setup.ch_lambda_ij)[chs, j]))
    _, power_evo = cfm.raman_solver.solve_isrs_evolution(
        ch_centre_i=np.asarray(setup.ch_centre_ij)[chs, j],
        A_eff=Aeff_i,
        raman_profile=setup.raman_profile_j[j],
        length=setup.length_j[j],
        attenuation_i=np.asarray(setup.attenuation_ij)[chs, j],
        ch_power_W_i=ch_power_W_i[:, j],
        ref_lambda=setup.ref_lambda,
        zspan=z,
    )
    p = np.asarray(power_evo)                        # (num_z, num_active)
    wl_nm = np.asarray(ch_lambda)[np.asarray(channel_idx)] * 1e9

    mat = p.copy()
    mat[mat == 0] = np.nan
    Z = 10.0 * np.log10(mat.T * 1e3)                # (num_active, num_z) dBm

    # 2D: power vs wavelength at fibre input (z=0) and end of span (z=L)
    fig, ax = plt.subplots(figsize=(10, 6))
    ms = 2.0
    ax.plot(
        wl_nm,
        Z[:, 0],
        ".",
        markersize=ms,
        color="C0",
        linestyle="none",
        label=r"$P(z=0)$",
    )
    ax.plot(
        wl_nm,
        Z[:, -1],
        ".",
        markersize=ms,
        color="C1",
        linestyle="none",
        label=r"$P(z=L)$",
    )
    ax.set_xlabel("Wavelength [nm]")
    ax.set_ylabel("Power [dBm]")
    ax.set_title("Power 2D")
    ax.grid(True, alpha=0.35)
    ax.legend(loc="lower right", framealpha=0.95)
    ax.margins(x=0.01)
    fig.tight_layout()
    plot_dir = _SCRIPT_DIR / "data" / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_fname = (
        f"power_evo_{band_selection}_{int(float(nspans) * float(span_length_km))}km_"
        f"{nspans}x{int(span_length_km)}km_lp{launch_power_dBm}.png"
    )
    fig.savefig(plot_dir / plot_fname, dpi=150)
    if os.environ.get("MPLNOSHOW", "").lower() not in ("1", "true", "yes"):
        plt.show()
    plt.close(fig)
    print(f"Saved plot: {plot_dir / plot_fname}")

    if save_snr_to_disk:
        snr_dir = _SCRIPT_DIR / "data" / "snr"
        snr_dir.mkdir(parents=True, exist_ok=True)
        ch_lambda_active = np.asarray(ch_lambda)[np.asarray(channel_idx)]
        ch_freq_active = c / ch_lambda_active
        snr_dB = np.full_like(nsr, np.nan, dtype=np.float64)
        snr_dB[valid] = 10.0 * np.log10(1.0 / nsr[valid])
        snr_ase_dB = np.full_like(nsr_ase, np.nan, dtype=np.float64)
        valid_ase = np.isfinite(nsr_ase) & (nsr_ase > 0)
        snr_ase_dB[valid_ase] = 10.0 * np.log10(1.0 / nsr_ase[valid_ase])
        fp = np.asarray(fit_params.squeeze(-1))
        fname = (
            f"p2p_no_rwa_{band_selection}_{int(total_km)}km_"
            f"{nspans}x{int(span_length_km)}km_lp{launch_power_dBm}.npz"
        )
        np.savez_compressed(
            snr_dir / fname,
            channel_index=np.arange(num_active),
            wavelength_m=ch_lambda_active,
            frequency_hz=ch_freq_active,
            occupancy=np.ones(num_active, dtype=np.int32),
            snr_dB=snr_dB,
            nsr_linear=nsr,
            nsr_ase_linear=nsr_ase,
            snr_ase_dB=snr_ase_dB,
            eta_spm_linear=np.asarray(eta_spm, dtype=np.float64).reshape(-1),
            eta_xpm_linear=np.asarray(eta_xpm, dtype=np.float64).reshape(-1),
            eta_fwm_linear=np.asarray(eta_fwm, dtype=np.float64).reshape(-1),
            span_length_km=np.float64(span_length_km),
            nspans=np.int32(nspans),
            total_km=np.float64(total_km),
            a=fp[:, 0],
            a_bar=fp[:, 1],
            Cr=fp[:, 2],
        )
        print(f"Saved: {snr_dir / fname}")

    return float(capacity_total)


if __name__ == "__main__":
    # ================================================================
    #  USER CONFIGURATION  (edit here; run from PyCharm / SSH)
    # ================================================================
    SAVE_SNR_TO_DISK = True
    BAND_SELECTION = "OESCL"
    SPAN_LENGTH_KM = 80.0
    NSPANS = 10
    LAUNCH_POWER_DBM = -2.0

    if BAND_SELECTION not in cfm.BAND_CONFIGS:
        raise ValueError(
            f"Invalid BAND_SELECTION={BAND_SELECTION!r}. "
            f"Options: {list(cfm.BAND_CONFIGS.keys())}"
        )
    if NSPANS < 1:
        raise ValueError("NSPANS must be >= 1")
    if SPAN_LENGTH_KM <= 0:
        raise ValueError("SPAN_LENGTH_KM must be positive")

    run_p2p(
        band_selection=BAND_SELECTION,
        span_length_km=SPAN_LENGTH_KM,
        nspans=NSPANS,
        launch_power_dBm=LAUNCH_POWER_DBM,
        save_snr_to_disk=SAVE_SNR_TO_DISK,
    )
