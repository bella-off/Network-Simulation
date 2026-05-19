"""
Sweep per-band launch power on a point-to-point link to find optimal values.

Uses a simple single-link (configurable number of spans) with all channels
occupied (mask=1) to compute the mean SNR per band as a function of launch
power.  Two stages:

  1. Coarse 1-D grid sweep: vary one band at a time (others at default),
     producing per-band SNR-vs-power curves for visualisation.
  2. Joint optimisation: scipy L-BFGS-B on all 5 band powers simultaneously
     to maximise total Shannon capacity.

Results are printed to stdout and saved to a CSV + a matplotlib figure.

Usage
-----
    python optimize_launch_power.py                          # defaults
    python optimize_launch_power.py --nspans 5 --bands OESCL
    python optimize_launch_power.py --nspans 1 --bands C     # C-band only
"""
from __future__ import annotations

import os
import ctypes

cuda_lib = "/apps/cuda/cuda-13.0/lib64/libcudart.so.13"
cupti_lib = "/apps/cuda/cuda-13.0/extras/CUPTI/lib64/libcupti.so.13"
cudnn_lib = "/apps/cuda/cudnn-linux-x86_64-9.14.0.64_cuda13/lib/libcudnn.so.9"

try:
    ctypes.CDLL(cuda_lib)
    ctypes.CDLL(cupti_lib)
    ctypes.CDLL(cudnn_lib)
except Exception as e:
    print(f"CUDA check: {e}")

os.environ["XLA_FLAGS"] = (
    "--xla_gpu_cuda_data_dir=/apps/cuda/cuda-13.0"
    " --xla_gpu_deterministic_ops=true"
)

import jax

print(f"GPU count: {jax.device_count()}")
print(f"Devices: {jax.devices()}")

import argparse
import csv
import pathlib
import sys
import time

import numpy as np
from jax import numpy as jnp
from scipy.optimize import minimize

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
_EXTERNAL_DIR = _SCRIPT_DIR.parents[2] / "external"
_ONG_SRC = _EXTERNAL_DIR / "ong-python-toolbox" / "src"

for p in (_EXTERNAL_DIR, str(_ONG_SRC)):
    p = str(p)
    if p not in sys.path:
        sys.path.insert(0, p)

import compute_throughput_cfm as cfm

ALL_BANDS = ["O", "E", "S", "C", "L"]

dBm = lambda x: 10 * np.log10(x / 1e-3)
idBm = lambda x: 10 ** (x / 10) * 1e-3
dB = lambda x: 10 * np.log10(x)
idB = lambda x: 10 ** (x / 10)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _build_per_channel_params_perband(ch_lambda, band_masks, active_bands,
                                      band_power_dBm: dict):
    """Like cfm._build_per_channel_params but with per-band launch power."""
    n = len(ch_lambda)
    P_channel = -jnp.inf * jnp.ones(n)
    nf = jnp.zeros(n)
    snr_trx = jnp.zeros(n)

    for b in active_bands:
        mask = band_masks[b]
        P_channel = P_channel.at[mask].set(band_power_dBm[b])
        nf = nf.at[mask].set(cfm.BANDS[b]['noise_figure'])
        snr_trx = snr_trx.at[mask].set(cfm.BANDS[b]['TransceiverSNR'])

    return P_channel, nf, snr_trx


def _eval_link(active_bands, band_power_dBm, nspans, span_length_km,
               ch_lambda, channel_idx, ch_idx_oband_full, band_masks,
               ref_lambda):
    """Compute per-channel NSR on a fully-loaded point-to-point link.

    Returns
    -------
    nsr : ndarray (num_active_channels,)
    """
    ch_idx_oband_active = ch_idx_oband_full[channel_idx]
    num_active = int(np.sum(channel_idx))

    P_channel, nf, snr_trx = _build_per_channel_params_perband(
        ch_lambda, band_masks, active_bands, band_power_dBm,
    )
    setup = cfm._build_setup(
        span_length_km, ch_lambda, channel_idx, P_channel, nf, snr_trx,
        ref_lambda=ref_lambda,
    )
    mask = np.ones((num_active, 1), dtype=int)
    nsr, *_ = cfm.calc_NSR_link(setup, nspans, mask, ch_idx_oband_active)
    return np.asarray(nsr).squeeze()


def _capacity_from_nsr(nsr, ch_bw_hz):
    """Shannon capacity [bps] summed over all finite-SNR channels."""
    valid = np.isfinite(nsr) & (nsr > 0)
    snr = np.where(valid, 1.0 / nsr, 0.0)
    rates = np.where(valid, 2 * ch_bw_hz * np.log2(1.0 + snr), 0.0)
    return float(np.sum(rates))


def _mean_snr_per_band(nsr, band_masks_active):
    """Mean SNR [dB] per band among active channels."""
    result = {}
    for b, mask in band_masks_active.items():
        vals = nsr[mask]
        valid = np.isfinite(vals) & (vals > 0)
        if np.any(valid):
            result[b] = float(np.mean(10 * np.log10(1.0 / vals[valid])))
        else:
            result[b] = float('nan')
    return result


# ---------------------------------------------------------------------------
# Stage 1: per-band 1-D sweep
# ---------------------------------------------------------------------------
def sweep_single_band(active_bands, target_band, default_powers,
                      nspans, span_length_km,
                      ch_lambda, channel_idx, ch_idx_oband_full,
                      band_masks, ref_lambda,
                      power_range_dBm=(-6, 2), steps=21):
    """Sweep one band's power, keeping others at default."""
    powers = np.linspace(power_range_dBm[0], power_range_dBm[1], steps)
    results = []

    band_masks_active = {
        b: np.asarray(band_masks[b])[np.asarray(channel_idx)]
        for b in active_bands
    }

    for p_dBm in powers:
        bp = dict(default_powers)
        bp[target_band] = float(p_dBm)
        t0 = time.perf_counter()
        nsr = _eval_link(active_bands, bp, nspans, span_length_km,
                         ch_lambda, channel_idx, ch_idx_oband_full,
                         band_masks, ref_lambda)
        dt = time.perf_counter() - t0
        cap = _capacity_from_nsr(nsr, cfm.CH_BW_HZ)
        snr_bands = _mean_snr_per_band(nsr, band_masks_active)
        results.append({
            'power_dBm': float(p_dBm),
            'capacity_Tbps': cap / 1e12,
            'snr_target_band_dB': snr_bands.get(target_band, float('nan')),
            'snr_all': snr_bands,
            'time_s': dt,
        })
        print(f"    {target_band}-band sweep: P={p_dBm:+6.2f} dBm  "
              f"cap={cap/1e12:.4f} Tbps  "
              f"SNR({target_band})={snr_bands.get(target_band, float('nan')):.2f} dB  "
              f"({dt:.1f}s)")

    best = max(results, key=lambda r: r['capacity_Tbps'])
    return powers, results, best


# ---------------------------------------------------------------------------
# Stage 2: joint optimisation
# ---------------------------------------------------------------------------
def joint_optimize(active_bands, initial_powers,
                   nspans, span_length_km,
                   ch_lambda, channel_idx, ch_idx_oband_full,
                   band_masks, ref_lambda,
                   bounds_dBm=(-6, 2)):
    """L-BFGS-B on total capacity (negative, for minimisation)."""
    band_list = [b for b in ALL_BANDS if b in active_bands]
    x0 = np.array([initial_powers[b] for b in band_list])
    bounds = [(bounds_dBm[0], bounds_dBm[1])] * len(band_list)
    eval_count = [0]

    def _neg_capacity(x):
        bp = {b: float(x[i]) for i, b in enumerate(band_list)}
        nsr = _eval_link(active_bands, bp, nspans, span_length_km,
                         ch_lambda, channel_idx, ch_idx_oband_full,
                         band_masks, ref_lambda)
        cap = _capacity_from_nsr(nsr, cfm.CH_BW_HZ)
        eval_count[0] += 1
        print(f"  [joint opt #{eval_count[0]}] "
              + "  ".join(f"{b}={bp[b]:+.2f}" for b in band_list)
              + f"  -> {cap/1e12:.4f} Tbps")
        return -cap

    res = minimize(_neg_capacity, x0, method='L-BFGS-B', bounds=bounds,
                   options={'maxiter': 50, 'eps': 0.2, 'ftol': 1e-10})

    opt_powers = {b: float(res.x[i]) for i, b in enumerate(band_list)}
    opt_cap = -res.fun
    return opt_powers, opt_cap, res


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def _plot_sweeps(sweep_data, active_bands, output_path):
    try:
        from matplotlib import pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not available; skipping plot.")
        return

    n = len(active_bands)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)
    axes = axes[0]

    for ax, band in zip(axes, active_bands):
        if band not in sweep_data:
            continue
        powers, results, best = sweep_data[band]
        caps = [r['capacity_Tbps'] for r in results]
        snrs = [r['snr_target_band_dB'] for r in results]

        ax.plot(powers, snrs, 'o-', color='tab:blue', label='Mean SNR')
        ax.set_xlabel('Launch Power [dBm]')
        ax.set_ylabel('Mean SNR [dB]', color='tab:blue')
        ax.set_title(f'{band}-band sweep')
        ax.axvline(best['power_dBm'], color='red', ls='--', alpha=0.7,
                    label=f"best={best['power_dBm']:+.1f} dBm")
        ax.legend(loc='lower left', fontsize=8)
        ax.grid(True, alpha=0.3)

        ax2 = ax.twinx()
        ax2.plot(powers, caps, 's--', color='tab:orange', markersize=4,
                 label='Capacity')
        ax2.set_ylabel('Capacity [Tbps]', color='tab:orange')

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"Saved plot: {output_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Optimise per-band launch power on a point-to-point link"
    )
    parser.add_argument("--nspans", type=int, default=1,
                        help="Number of spans (default 1)")
    parser.add_argument("--span_km", type=float, default=80.0,
                        help="Span length in km (default 80)")
    parser.add_argument("--bands", type=str, default="OESCL",
                        choices=list(cfm.BAND_CONFIGS.keys()),
                        help="Band selection (default OESCL)")
    parser.add_argument("--default_power", type=float, default=0.0,
                        help="Default per-channel launch power [dBm] (default 0)")
    parser.add_argument("--sweep_lo", type=float, default=-4.0,
                        help="Sweep lower bound [dBm]")
    parser.add_argument("--sweep_hi", type=float, default=4.0,
                        help="Sweep upper bound [dBm]")
    parser.add_argument("--sweep_steps", type=int, default=21,
                        help="Number of sweep points per band")
    parser.add_argument("--skip_joint", action="store_true",
                        help="Skip joint optimisation (sweep only)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: <script_dir>/data)")
    args = parser.parse_args()

    band_key = args.bands
    band_cfg = cfm.BAND_CONFIGS[band_key]
    active_bands = band_cfg["bands"]
    nspans = args.nspans
    span_km = args.span_km
    default_power = args.default_power

    out_dir = pathlib.Path(args.output_dir) if args.output_dir else _SCRIPT_DIR / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    ch_lambda, channel_idx, ch_idx_oband_full, band_masks, ref_lambda = (
        cfm._build_multiband_channel_grid(active_bands)
    )
    num_active = int(np.sum(channel_idx))

    print("=" * 65)
    print("Launch Power Optimisation  (point-to-point link)")
    print("=" * 65)
    print(f"Bands          : {band_key} -> {active_bands}")
    print(f"Active channels: {num_active}")
    print(f"Spans           : {nspans} x {span_km} km")
    print(f"Default power  : {default_power} dBm")
    print(f"Sweep range    : [{args.sweep_lo}, {args.sweep_hi}] dBm, {args.sweep_steps} steps")
    print(f"Output dir     : {out_dir}")
    print("=" * 65)

    default_powers = {b: default_power for b in active_bands}

    # ---- Stage 1: 1-D sweep per band ----
    print("\n>>> Stage 1: per-band 1-D sweep")
    sweep_data = {}
    sweep_csv_rows = []

    for band in active_bands:
        print(f"\n  --- Sweeping {band}-band ---")
        powers, results, best = sweep_single_band(
            active_bands, band, default_powers,
            nspans, span_km,
            ch_lambda, channel_idx, ch_idx_oband_full,
            band_masks, ref_lambda,
            power_range_dBm=(args.sweep_lo, args.sweep_hi),
            steps=args.sweep_steps,
        )
        sweep_data[band] = (powers, results, best)
        print(f"  Best for {band}-band: P={best['power_dBm']:+.2f} dBm, "
              f"cap={best['capacity_Tbps']:.4f} Tbps, "
              f"SNR={best['snr_target_band_dB']:.2f} dB")

        for r in results:
            sweep_csv_rows.append({
                "swept_band": band,
                "power_dBm": r['power_dBm'],
                "capacity_Tbps": r['capacity_Tbps'],
                "snr_target_dB": r['snr_target_band_dB'],
                "nspans": nspans,
                "span_km": span_km,
                "band_selection": band_key,
            })

    csv_path = out_dir / f"launch_power_sweep_{band_key}_{nspans}spans.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sweep_csv_rows[0].keys()))
        w.writeheader()
        w.writerows(sweep_csv_rows)
    print(f"\nSaved sweep CSV: {csv_path}")

    fig_path = out_dir / f"launch_power_sweep_{band_key}_{nspans}spans.png"
    _plot_sweeps(sweep_data, active_bands, fig_path)

    # ---- Stage 2: joint optimisation ----
    if not args.skip_joint and len(active_bands) > 1:
        print("\n>>> Stage 2: joint optimisation (L-BFGS-B)")
        initial = {b: sweep_data[b][2]['power_dBm'] for b in active_bands}
        print(f"  Starting from sweep best: {initial}")

        opt_powers, opt_cap, res = joint_optimize(
            active_bands, initial,
            nspans, span_km,
            ch_lambda, channel_idx, ch_idx_oband_full,
            band_masks, ref_lambda,
            bounds_dBm=(args.sweep_lo, args.sweep_hi),
        )
        print(f"\n  Joint optimum:")
        for b in active_bands:
            print(f"    {b}-band: {opt_powers[b]:+.2f} dBm")
        print(f"  Total capacity: {opt_cap/1e12:.4f} Tbps")
        print(f"  Optimiser: success={res.success}, nfev={res.nfev}, message={res.message}")
    elif len(active_bands) == 1:
        b = active_bands[0]
        opt_powers = {b: sweep_data[b][2]['power_dBm']}
        opt_cap = sweep_data[b][2]['capacity_Tbps'] * 1e12
        print(f"\n  Single band ({b}): optimal P={opt_powers[b]:+.2f} dBm, "
              f"cap={opt_cap/1e12:.4f} Tbps")
    else:
        opt_powers = {b: sweep_data[b][2]['power_dBm'] for b in active_bands}
        opt_cap = 0.0

    # ---- Summary ----
    print("\n" + "=" * 65)
    print("OPTIMAL PER-BAND LAUNCH POWERS")
    print("=" * 65)
    for b in active_bands:
        print(f"  '{b}': {{'p_launch_opt': {opt_powers[b]:+.2f}}},")
    print("=" * 65)
    print(f"Paste into BANDS dict or use as per-band launch_power_dBm.")
    print("Done.\n")


if __name__ == "__main__":
    main()