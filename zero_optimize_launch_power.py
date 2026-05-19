"""
Per-band launch power sweep on a point-to-point link (Stage 1 + optional joint).

**Configure runs** by editing the ``USER CONFIGURATION`` block in
``if __name__ == "__main__":`` at the bottom of this file.

**``other_bands_mode``**

* ``isolated`` — only the swept band has ``mask=1`` in ``calc_NSR_link``; other
  bands use ``mask=0``.
* ``default_power`` — like ``optimize_launch_power.py``: all ``mask=1``;
  non-swept bands at ``default_power_dBm`` while the swept band's power varies.

**``snr_metric``**: ``mean_linear_dB`` or ``mean_dB``.

Stage 2 (optional): L-BFGS-B on **fully loaded** link when ``skip_joint`` is False.
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

SNR_METRIC_CHOICES = ("mean_dB", "mean_linear_dB")
OTHER_BANDS_MODE_CHOICES = ("isolated", "default_power")


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
               ref_lambda, occupancy_mask=None):
    """NSR via ``cfm.calc_NSR_link``; optional per-channel binary ``mask``."""
    ch_idx_oband_active = ch_idx_oband_full[channel_idx]
    num_active = int(np.sum(channel_idx))

    P_channel, nf, snr_trx = _build_per_channel_params_perband(
        ch_lambda, band_masks, active_bands, band_power_dBm,
    )
    setup = cfm._build_setup(
        span_length_km, ch_lambda, channel_idx, P_channel, nf, snr_trx,
        ref_lambda=ref_lambda,
    )
    if occupancy_mask is None:
        mask = np.ones((num_active, 1), dtype=np.int32)
    else:
        mask = np.asarray(occupancy_mask).reshape(num_active, 1)
        mask = (mask != 0).astype(np.int32)
    nsr, *_ = cfm.calc_NSR_link(setup, nspans, mask, ch_idx_oband_active)
    return np.asarray(nsr).squeeze()


def _capacity_from_nsr(nsr, ch_bw_hz, lit_mask=None):
    """Shannon capacity [bps] over lit channels with finite NSR."""
    nsr = np.asarray(nsr).reshape(-1)
    if lit_mask is None:
        lit = np.ones_like(nsr, dtype=bool)
    else:
        lit = np.asarray(lit_mask).reshape(-1).astype(bool)
    valid = np.isfinite(nsr) & (nsr > 0) & lit
    snr = np.where(valid, 1.0 / nsr, 0.0)
    rates = np.where(valid, 2 * ch_bw_hz * np.log2(1.0 + snr), 0.0)
    return float(np.sum(rates))


def _aggregate_snr_dB(vals: np.ndarray, snr_metric: str) -> float:
    """Scalar SNR [dB] from NSR samples ``vals`` (linear NSR, >0)."""
    valid = np.isfinite(vals) & (vals > 0)
    if not np.any(valid):
        return float("nan")
    v = vals[valid]
    if snr_metric == "mean_dB":
        return float(np.mean(10.0 * np.log10(1.0 / v)))
    if snr_metric == "mean_linear_dB":
        gamma = 1.0 / v
        m = float(np.mean(gamma))
        if m > 0.0:
            return float(10.0 * np.log10(m))
        return float("nan")
    raise ValueError(f"unknown snr_metric: {snr_metric!r}")


def _mean_snr_per_band(nsr, band_masks_active, lit_mask=None,
                       snr_metric: str = "mean_linear_dB"):
    """Per-band SNR [dB]; see ``SNR_METRIC_CHOICES`` for ``snr_metric``."""
    if snr_metric not in SNR_METRIC_CHOICES:
        raise ValueError(f"snr_metric must be one of {SNR_METRIC_CHOICES}")
    nsr = np.asarray(nsr).reshape(-1)
    result = {}
    for b, mband in band_masks_active.items():
        sel = np.asarray(mband, dtype=bool)
        if lit_mask is not None:
            sel = sel & np.asarray(lit_mask, dtype=bool).reshape(-1)
        vals = nsr[sel]
        result[b] = _aggregate_snr_dB(vals, snr_metric)
    return result


# ---------------------------------------------------------------------------
# Stage 1: per-band 1-D sweep
# ---------------------------------------------------------------------------
def sweep_single_band_stage1(
    active_bands,
    target_band,
    default_powers,
    nspans,
    span_length_km,
    ch_lambda,
    channel_idx,
    ch_idx_oband_full,
    band_masks,
    ref_lambda,
    power_range_dBm=(-6, 2),
    steps=21,
    snr_metric: str = "mean_linear_dB",
    other_bands_mode: str = "isolated",
):
    """Sweep ``target_band`` launch power; see ``OTHER_BANDS_MODE_CHOICES``."""
    if other_bands_mode not in OTHER_BANDS_MODE_CHOICES:
        raise ValueError(
            f"other_bands_mode must be one of {OTHER_BANDS_MODE_CHOICES}"
        )

    powers = np.linspace(power_range_dBm[0], power_range_dBm[1], steps)
    results = []

    ci = np.asarray(channel_idx, dtype=bool)
    band_masks_active = {
        b: np.asarray(band_masks[b])[ci]
        for b in active_bands
    }
    num_active = int(np.sum(ci))

    if other_bands_mode == "isolated":
        lit_1d = np.zeros(num_active, dtype=bool)
        lit_1d[band_masks_active[target_band]] = True
        occ = lit_1d.astype(np.int32).reshape(-1, 1)
        cap_lit_mask = lit_1d
        tag = "isolated"
    else:
        lit_1d = np.ones(num_active, dtype=bool)
        occ = None
        cap_lit_mask = None
        tag = "default_power"

    for p_dBm in powers:
        bp = dict(default_powers)
        bp[target_band] = float(p_dBm)
        t0 = time.perf_counter()
        nsr = _eval_link(
            active_bands, bp, nspans, span_length_km,
            ch_lambda, channel_idx, ch_idx_oband_full,
            band_masks, ref_lambda,
            occupancy_mask=occ,
        )
        dt = time.perf_counter() - t0
        cap = _capacity_from_nsr(nsr, cfm.CH_BW_HZ, lit_mask=cap_lit_mask)
        snr_bands = _mean_snr_per_band(
            nsr,
            band_masks_active,
            lit_mask=cap_lit_mask,
            snr_metric=snr_metric,
        )
        results.append({
            'power_dBm': float(p_dBm),
            'capacity_Tbps': cap / 1e12,
            'snr_target_band_dB': snr_bands.get(target_band, float('nan')),
            'snr_all': snr_bands,
            'time_s': dt,
            'snr_metric': snr_metric,
            'other_bands_mode': other_bands_mode,
        })
        print(
            f"    {target_band}-band sweep [{tag}]: P={p_dBm:+6.2f} dBm  "
            f"cap={cap/1e12:.4f} Tbps  "
            f"SNR({target_band},{snr_metric})="
            f"{snr_bands.get(target_band, float('nan')):.2f} dB  "
            f"({dt:.1f}s)"
        )

    best = max(results, key=lambda r: r['capacity_Tbps'])
    return powers, results, best


# ---------------------------------------------------------------------------
# Stage 2: joint optimisation (full load — all mask=1)
# ---------------------------------------------------------------------------
def joint_optimize(active_bands, initial_powers,
                   nspans, span_length_km,
                   ch_lambda, channel_idx, ch_idx_oband_full,
                   band_masks, ref_lambda,
                   bounds_dBm=(-6, 2)):
    """L-BFGS-B on total capacity with all channels on (same as original)."""
    band_list = [b for b in ALL_BANDS if b in active_bands]
    x0 = np.array([initial_powers[b] for b in band_list])
    bounds = [(bounds_dBm[0], bounds_dBm[1])] * len(band_list)
    eval_count = [0]

    def _neg_capacity(x):
        bp = {b: float(x[i]) for i, b in enumerate(band_list)}
        nsr = _eval_link(active_bands, bp, nspans, span_length_km,
                         ch_lambda, channel_idx, ch_idx_oband_full,
                         band_masks, ref_lambda,
                         occupancy_mask=None)
        cap = _capacity_from_nsr(nsr, cfm.CH_BW_HZ, lit_mask=None)
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
def _plot_sweeps(
    sweep_data,
    active_bands,
    output_path,
    snr_metric: str,
    other_bands_mode: str,
):
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

        ylab = (
            'mean(10·log10 SNR) [dB]' if snr_metric == 'mean_dB'
            else '10·log10(mean SNR_lin) [dB]'
        )
        ax.plot(powers, snrs, 'o-', color='tab:blue', label=ylab)
        ax.set_xlabel('Launch Power [dBm]')
        ax.set_ylabel(ylab, color='tab:blue')
        sub = (
            'others dark (mask=0)'
            if other_bands_mode == 'isolated'
            else 'all bands on, others @ default P'
        )
        ax.set_title(f'{band}-band — {sub}\n{snr_metric}')
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
# Sweep driver (parameters supplied from ``if __name__ == "__main__"`` block)
# ---------------------------------------------------------------------------
def run_launch_power_sweep(
    *,
    nspans: int,
    span_km: float,
    band_selection: str,
    default_power_dbm: float,
    sweep_lo_dbm: float,
    sweep_hi_dbm: float,
    sweep_steps: int,
    skip_joint: bool,
    output_dir: pathlib.Path | str | None,
    snr_metric: str,
    other_bands_mode: str,
) -> None:
    if band_selection not in cfm.BAND_CONFIGS:
        raise ValueError(
            "band_selection must be one of "
            f"{list(cfm.BAND_CONFIGS.keys())}, got {band_selection!r}"
        )
    if snr_metric not in SNR_METRIC_CHOICES:
        raise ValueError(f"snr_metric must be one of {SNR_METRIC_CHOICES}")
    if other_bands_mode not in OTHER_BANDS_MODE_CHOICES:
        raise ValueError(f"other_bands_mode must be one of {OTHER_BANDS_MODE_CHOICES}")

    band_key = band_selection
    band_cfg = cfm.BAND_CONFIGS[band_key]
    active_bands = band_cfg["bands"]
    default_power = float(default_power_dbm)
    sweep_lo = float(sweep_lo_dbm)
    sweep_hi = float(sweep_hi_dbm)

    out_dir = pathlib.Path(output_dir) if output_dir else _SCRIPT_DIR / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    ch_lambda, channel_idx, ch_idx_oband_full, band_masks, ref_lambda = (
        cfm._build_multiband_channel_grid(active_bands)
    )
    num_active = int(np.sum(channel_idx))

    print("=" * 65)
    print("Launch power sweep (Stage 1 + optional joint)")
    print("=" * 65)
    print(f"Bands              : {band_key} -> {active_bands}")
    print(f"Active channels    : {num_active}")
    print(f"Spans               : {nspans} x {span_km} km")
    print(f"default_power_dBm  : {default_power} dBm")
    print(f"Sweep range         : [{sweep_lo}, {sweep_hi}] dBm, {int(sweep_steps)} steps")
    print(f"Output dir          : {out_dir}")
    print(f"snr_metric          : {snr_metric}")
    print(f"other_bands_mode    : {other_bands_mode}")
    if other_bands_mode == "isolated":
        print("Stage 1            : only swept band mask=1 (others mask=0)")
    else:
        print("Stage 1            : all bands mask=1; non-swept @ "
              f"{default_power} dBm/ch (swept band varies)")
    print(f"Stage 2            : {'skipped' if skip_joint else 'full WDM joint opt'}")
    print("=" * 65)

    default_powers = {b: default_power for b in active_bands}

    print("\n>>> Stage 1: per-band 1-D sweep")
    sweep_data = {}
    sweep_csv_rows = []

    for band in active_bands:
        print(f"\n  --- Sweeping {band}-band ---")
        powers, results, best = sweep_single_band_stage1(
            active_bands, band, default_powers,
            nspans, span_km,
            ch_lambda, channel_idx, ch_idx_oband_full,
            band_masks, ref_lambda,
            power_range_dBm=(sweep_lo, sweep_hi),
            steps=int(sweep_steps),
            snr_metric=snr_metric,
            other_bands_mode=other_bands_mode,
        )
        sweep_data[band] = (powers, results, best)
        print(f"  Best for {band}-band: P={best['power_dBm']:+.2f} dBm, "
              f"cap={best['capacity_Tbps']:.4f} Tbps, "
              f"SNR({snr_metric})={best['snr_target_band_dB']:.2f} dB")

        for r in results:
            sweep_csv_rows.append({
                "swept_band": band,
                "power_dBm": r['power_dBm'],
                "capacity_Tbps": r['capacity_Tbps'],
                "snr_target_dB": r['snr_target_band_dB'],
                "nspans": nspans,
                "span_km": span_km,
                "band_selection": band_key,
                "other_bands_mode": other_bands_mode,
                "default_power_dBm": default_power,
                "snr_metric": snr_metric,
            })

    csv_path = (
        out_dir
        / f"zero_launch_power_sweep_{band_key}_{nspans}spans_"
        f"{other_bands_mode}_{snr_metric}.csv"
    )
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sweep_csv_rows[0].keys()))
        w.writeheader()
        w.writerows(sweep_csv_rows)
    print(f"\nSaved sweep CSV: {csv_path}")

    fig_path = (
        out_dir
        / f"zero_launch_power_sweep_{band_key}_{nspans}spans_"
        f"{other_bands_mode}_{snr_metric}.png"
    )
    _plot_sweeps(
        sweep_data,
        active_bands,
        fig_path,
        snr_metric,
        other_bands_mode,
    )

    if not skip_joint and len(active_bands) > 1:
        print("\n>>> Stage 2: joint optimisation (L-BFGS-B, full load)")
        initial = {b: sweep_data[b][2]['power_dBm'] for b in active_bands}
        print(f"  Starting from Stage-1 sweep bests: {initial}")

        opt_powers, opt_cap, res = joint_optimize(
            active_bands, initial,
            nspans, span_km,
            ch_lambda, channel_idx, ch_idx_oband_full,
            band_masks, ref_lambda,
            bounds_dBm=(sweep_lo, sweep_hi),
        )
        print(f"\n  Joint optimum (full WDM):")
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

    print("\n" + "=" * 65)
    print("OPTIMAL PER-BAND LAUNCH POWERS (from Stage-1 sweeps / joint)")
    print("=" * 65)
    for b in active_bands:
        print(f"  '{b}': {{'p_launch_opt': {opt_powers[b]:+.2f}}},")
    print("=" * 65)
    print("Stage-1 optimum powers can differ from full-WDM joint optimum when "
          "Stage 2 runs. Done.\n")


if __name__ == "__main__":
    # ================================================================
    #  USER CONFIGURATION  (edit here, then run this file)
    # ================================================================
    NSPANS = 1
    SPAN_KM = 80.0
    # Key in cfm.BAND_CONFIGS, e.g. "OESCL", "SCL", "C", ...
    BAND_SELECTION = "OESCL"
    # Non-swept bands in default_power mode; baseline per-band dict each step.
    DEFAULT_POWER_DBM = 0.0
    SWEEP_LO_DBM = -4.0
    SWEEP_HI_DBM = 4.0
    SWEEP_STEPS = 21
    SKIP_JOINT = False
    # None -> <this script directory>/data
    OUTPUT_DIR = None

    # "mean_linear_dB" or "mean_dB"
    SNR_METRIC = "mean_linear_dB"
    # "isolated" or "default_power"
    OTHER_BANDS_MODE = "isolated"

    run_launch_power_sweep(
        nspans=NSPANS,
        span_km=SPAN_KM,
        band_selection=BAND_SELECTION,
        default_power_dbm=DEFAULT_POWER_DBM,
        sweep_lo_dbm=SWEEP_LO_DBM,
        sweep_hi_dbm=SWEEP_HI_DBM,
        sweep_steps=SWEEP_STEPS,
        skip_joint=SKIP_JOINT,
        output_dir=OUTPUT_DIR,
        snr_metric=SNR_METRIC,
        other_bands_mode=OTHER_BANDS_MODE,
    )
