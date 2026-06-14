"""
Joint per-band launch power optimisation on a real network (e.g. NSFNET) with
RWA, using a gradient-based optimiser instead of a brute-force 1-D sweep.

This is the ``simulation.py`` (watt-band-next-ofc2026) analogue of
``network_optimize_launch_power.py``.  Instead of sweeping every band's launch
power on a 1-D grid and taking the argmax (one band at a time, the rest held at
a default), here ALL active bands are optimised *jointly* with
``scipy.optimize.minimize(method='L-BFGS-B')``.

How this mirrors ``simulation.py``:
  * Optimisation variable ``x``: a per-band launch power vector [dBm]
    (the network model applies one launch power per band).  This is the
    network analogue of simulation.py's per-band power control points.
  * Objective: negative total network throughput (``-capacity``), i.e. the
    optimiser maximises capacity -- matching simulation.py's ``sim_num=3``.
    An alternative "SNR mean minus std" objective (simulation.py ``sim_num=4/5``)
    is also provided, with an optional SNR cut-off (``sim_num=7/8``).
  * Solver: ``scipy.optimize.minimize`` with ``method='L-BFGS-B'`` and no
    analytic Jacobian -> SciPy estimates the gradient by finite differences
    (``options={'eps': ...}``), exactly like simulation.py.
  * Bounds: simple box bounds via ``scipy.optimize.Bounds`` (cf. Bounds(-10,10)).
  * Optional total-power constraint via linear-domain rescaling
    (the per-band analogue of simulation.py's ``p /= p.sum()/min(p.sum(),p_lim)``).
  * Optional coarse-to-fine warm start: a loose-tolerance pass whose solution
    seeds a tight-tolerance pass (simulation.py reuses coarse results as p_init).
  * Multi-start: L-BFGS-B is a local optimiser, so it is launched from several
    starting points (the default init, explicit warm starts such as a PSO
    solution, and random points within the bounds) and the best final
    capacity wins.  Stopping tolerances are given explicitly as ``ftol`` /
    ``gtol`` -- the L-BFGS-B ``ftol`` test is *relative* to ``max(|f|, 1)``,
    so with costs of order 1e3 Tbps a generic ``tol=1e-2`` would stop ~14 Tbps
    short of the optimum.

Edit the USER CONFIGURATION block (or override via CLI) and run.
"""
from __future__ import annotations

import argparse
import csv
import pathlib
import sys
import time

import numpy as np
from scipy.optimize import minimize, Bounds

_SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
_EXTERNAL_DIR = _SCRIPT_DIR.parents[2] / "external"
_ONG_SRC = _EXTERNAL_DIR / "ong-python-toolbox" / "src"

for _p in (str(_SCRIPT_DIR), str(_EXTERNAL_DIR), str(_ONG_SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import compute_throughput_cfm_new as cfm  # noqa: E402  (path setup above)

# Reuse the data-loading / evaluation infrastructure of the sweep script so the
# physics (CFM NLI model + RWA occupancy) is identical -- only the optimisation
# method differs.
from network_optimize_launch_power import (  # noqa: E402
    SNR_METRIC_CHOICES,
    _build_wavelength_band_map,
    eval_network_metrics,
    load_topologies,
    precompute,
)

OBJECTIVE_CHOICES = ("capacity", "snr_mean_std")


# ---------------------------------------------------------------------------
# Joint launch-power optimisation ( L-BFGS-B)
# ---------------------------------------------------------------------------
def _channels_per_band(active_bands, band_masks, channel_idx):
    """Number of *active* channels in each band (for the power constraint)."""
    ci = np.asarray(channel_idx, dtype=bool)
    return {
        b: int(np.sum(np.asarray(band_masks[b], dtype=bool) & ci))
        for b in active_bands
    }


def _apply_power_limit(band_power, power_limit_dBm, chans_per_band):
    """Rescale per-band linear powers so the total stays <= the limit.

    Network analogue of simulation.py's
    ``p /= p.sum() / jnp.minimum(p.sum(), p_lim)``: convert each band's per
    channel dBm to linear, weight by the band's channel count to get the total
    launch power, and shrink everything by a common factor if it exceeds the
    limit (a uniform dBm offset).  No-op when the total is already within budget.
    """
    if power_limit_dBm is None:
        return band_power
    lin = {b: 10.0 ** ((p - 30.0) / 10.0) for b, p in band_power.items()}
    total = sum(lin[b] * chans_per_band.get(b, 0) for b in band_power)
    p_lim = 10.0 ** ((power_limit_dBm - 30.0) / 10.0)
    if total <= p_lim or total <= 0.0:
        return band_power
    scale = p_lim / total
    return {b: 10.0 * np.log10(lin[b] * scale) + 30.0 for b in band_power}


def optimize_launch_power(active_bands, grid, precomp_list, w_to_band, *,
                          x0_dBm, bound_lo_dBm, bound_hi_dBm, ftol, gtol, eps,
                          maxiter, objective, snr_metric, snr_cutoff_dB,
                          power_limit_dBm, chans_per_band, history):
    """Run one L-BFGS-B pass; return the SciPy ``solution`` object.

    The cost function builds a per-band launch power dict from ``x``, evaluates
    the network throughput / SNR with the shared CFM model, and returns the
    scalar to minimise.  No Jacobian is supplied, so SciPy uses finite
    differences with step ``eps`` (identical to simulation.py).
    """
    if objective not in OBJECTIVE_CHOICES:
        raise ValueError(f"objective must be one of {OBJECTIVE_CHOICES}")

    state = {"nfev": 0}

    def band_power_from_x(x):
        bp = {b: float(x[i]) for i, b in enumerate(active_bands)}
        return _apply_power_limit(bp, power_limit_dBm, chans_per_band)

    def cost_fun(x):
        bp = band_power_from_x(x)
        t0 = time.perf_counter()
        cap, snr_bands = eval_network_metrics(
            bp, active_bands, grid, precomp_list, w_to_band,
            snr_metric=snr_metric,
        )
        dt = time.perf_counter() - t0
        state["nfev"] += 1

        if objective == "capacity":
            cost = -cap / 1e12  # work in Tbps to keep gradients well scaled
        else:  # "snr_mean_std" -- simulation.py sim_num=4/5 (+7/8 cut-off)
            vals = np.array([snr_bands[b] for b in active_bands], dtype=float)
            if snr_cutoff_dB is not None:
                vals = np.where(vals < snr_cutoff_dB, np.nan, vals)
            cost = -(np.nanmean(vals) - np.nanstd(vals))

        history.append({
            "nfev": state["nfev"],
            "cost": float(cost),
            "capacity_Tbps": cap / 1e12,
            "powers_dBm": {b: bp[b] for b in active_bands},
            "snr_all": snr_bands,
            "time_s": dt,
        })
        pwr_str = "  ".join(f"{b}={bp[b]:+5.2f}" for b in active_bands)
        print(
            f"    eval {state['nfev']:3d}  cost={cost:+.4f}  "
            f"cap={cap/1e12:.4f} Tbps  [{pwr_str}]  ({dt:.1f}s)"
        )
        return float(cost)

    bounds = Bounds(lb=bound_lo_dBm, ub=bound_hi_dBm)
    solution = minimize(
        fun=cost_fun,
        x0=np.asarray(x0_dBm, dtype=float),
        bounds=bounds,
        method="L-BFGS-B",
        options=dict(disp=True, eps=eps, maxiter=maxiter,
                     ftol=ftol, gtol=gtol),
    )
    return solution


def plot_convergence(history, active_bands, out_path):
    try:
        from matplotlib import pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not available; skipping plot.")
        return
    if not history:
        return
    nfev = [h["nfev"] for h in history]
    caps = [h["capacity_Tbps"] for h in history]

    fig, (ax_cap, ax_pow) = plt.subplots(1, 2, figsize=(11, 4))
    ax_cap.plot(nfev, caps, "-o", color="tab:orange", ms=3)
    # Vertical separators + labels where a new multi-start run begins
    prev_start = None
    for h in history:
        s = h.get("start")
        if s != prev_start:
            if prev_start is not None:
                ax_cap.axvline(h["nfev"] - 0.5, color="grey", ls="--", lw=0.8)
                ax_pow.axvline(h["nfev"] - 0.5, color="grey", ls="--", lw=0.8)
            if s is not None:
                ax_cap.annotate(s, (h["nfev"], min(caps)), fontsize=7,
                                color="grey", rotation=90, va="bottom")
            prev_start = s
    ax_cap.set_xlabel("Objective evaluation")
    ax_cap.set_ylabel("Network Capacity [Tbps]")
    ax_cap.set_title("Convergence (multi-start L-BFGS-B, finite-diff grad)")
    ax_cap.grid(True, alpha=0.3)

    for b in active_bands:
        pw = [h["powers_dBm"][b] for h in history]
        ax_pow.plot(nfev, pw, "-", label=f"{b}-band")
    ax_pow.set_xlabel("Objective evaluation")
    ax_pow.set_ylabel("Launch Power [dBm]")
    ax_pow.set_title("Per-band launch power trajectory")
    ax_pow.legend(loc="best", fontsize=8)
    ax_pow.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot: {out_path}")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def run(*, db, collection, topology_name, route_function, band_selection,
        span_length_km, init_power_dbm, bound_lo_dbm, bound_hi_dbm,
        ftol, gtol, eps, maxiter, coarse_to_fine, objective, snr_cutoff_dB,
        power_limit_dbm, output_dir, snr_metric: str = "mean_linear_dB",
        x0_list=None, n_random_starts: int = 0, seed=None):
    """Multi-start L-BFGS-B: launch from the default init, every explicit
    warm start in ``x0_list`` (e.g. a PSO solution, dict band -> dBm), and
    ``n_random_starts`` uniform random points in the bounds; keep the best."""
    if band_selection not in cfm.BAND_CONFIGS:
        raise ValueError(
            f"band_selection must be one of {list(cfm.BAND_CONFIGS.keys())}, "
            f"got {band_selection!r}"
        )
    if snr_metric not in SNR_METRIC_CHOICES:
        raise ValueError(f"snr_metric must be one of {SNR_METRIC_CHOICES}")
    if objective not in OBJECTIVE_CHOICES:
        raise ValueError(f"objective must be one of {OBJECTIVE_CHOICES}")
    active_bands = cfm.BAND_CONFIGS[band_selection]["bands"]

    out_dir = pathlib.Path(output_dir) if output_dir else _SCRIPT_DIR / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    topos, rwa_key = load_topologies(
        db, collection, topology_name, route_function, band_selection)
    if not topos:
        raise RuntimeError(f"No usable topologies for '{topology_name}'")

    ch_lambda, channel_idx, ch_idx_oband_full, band_masks, ref_lambda = (
        cfm._build_multiband_channel_grid(active_bands)
    )
    num_active = int(np.sum(channel_idx))
    ch_idx_oband_active = ch_idx_oband_full[channel_idx]
    active_slot_ix = np.flatnonzero(np.asarray(channel_idx))
    grid = (ch_lambda, channel_idx, ch_idx_oband_active, band_masks,
            ref_lambda, span_length_km, active_slot_ix)

    precomp_list = [precompute(g, rwa, num_active) for _, g, rwa in topos]
    w_to_band = _build_wavelength_band_map(
        active_slot_ix, band_masks, active_bands)
    chans_per_band = _channels_per_band(active_bands, band_masks, channel_idx)

    # ---- Multi-start: default init + explicit warm starts + random points ----
    starts: list[tuple[str, np.ndarray]] = [
        ("default", np.full(len(active_bands), float(init_power_dbm))),
    ]
    for j, d in enumerate(x0_list or []):
        missing = [b for b in active_bands if b not in d]
        if missing:
            raise ValueError(f"x0_list[{j}] is missing bands {missing}")
        starts.append((
            f"warm{j}",
            np.array([float(d[b]) for b in active_bands]),
        ))
    rng = np.random.default_rng(seed)
    for j in range(int(n_random_starts)):
        starts.append((
            f"rand{j}",
            rng.uniform(bound_lo_dbm, bound_hi_dbm, len(active_bands)),
        ))

    print("=" * 65)
    print("Network joint launch power optimisation (L-BFGS-B, simulation.py-style)")
    print("=" * 65)
    print(f"Topology           : {topology_name} ({len(topos)} doc(s))")
    print(f"RWA key            : {rwa_key}")
    print(f"Bands              : {band_selection} -> {active_bands}")
    print(f"Active channels    : {num_active}")
    print(f"Channels per band  : {chans_per_band}")
    print(f"Span length        : {span_length_km} km")
    print(f"Init power         : {init_power_dbm} dBm (all bands)")
    print(f"Bounds             : [{bound_lo_dbm}, {bound_hi_dbm}] dBm")
    print(f"Objective          : {objective}"
          + (f" (cut-off {snr_cutoff_dB} dB)" if snr_cutoff_dB is not None else ""))
    print(f"Power limit        : "
          f"{power_limit_dbm if power_limit_dbm is not None else 'none'} dBm")
    print(f"L-BFGS-B           : ftol={ftol}, gtol={gtol}, eps={eps}, "
          f"maxiter={maxiter}, finite-diff gradient (no jac)")
    print(f"Multi-start        : {len(starts)} starts = 1 default "
          f"+ {len(x0_list or [])} warm + {n_random_starts} random "
          f"(seed={seed})")
    print(f"coarse_to_fine     : {coarse_to_fine}")
    print(f"snr_metric         : {snr_metric}")
    print(f"Output dir         : {out_dir}")
    print("=" * 65)

    history: list[dict] = []   # merged across starts (nfev renumbered globally)
    results: list[dict] = []   # one final record per start

    t_start = time.perf_counter()
    for label, x0 in starts:
        print(f"\n### Start '{label}': x0 = "
              f"{dict(zip(active_bands, np.round(x0, 2)))}")
        start_hist: list[dict] = []

        if coarse_to_fine:
            print("  --- Coarse pass (loose tol) ---")
            coarse = optimize_launch_power(
                active_bands, grid, precomp_list, w_to_band,
                x0_dBm=x0, bound_lo_dBm=bound_lo_dbm, bound_hi_dBm=bound_hi_dbm,
                ftol=max(ftol * 1e3, 1e-4), gtol=max(gtol * 1e2, 1e-4),
                eps=eps * 2, maxiter=max(maxiter // 2, 5),
                objective=objective, snr_metric=snr_metric,
                snr_cutoff_dB=snr_cutoff_dB, power_limit_dBm=power_limit_dbm,
                chans_per_band=chans_per_band, history=start_hist,
            )
            x0 = coarse.x
            print(f"  Coarse solution: "
                  f"{dict(zip(active_bands, np.round(x0, 3)))}")
            print("  --- Fine pass (tight tol) ---")

        solution = optimize_launch_power(
            active_bands, grid, precomp_list, w_to_band,
            x0_dBm=x0, bound_lo_dBm=bound_lo_dbm, bound_hi_dBm=bound_hi_dbm,
            ftol=ftol, gtol=gtol, eps=eps, maxiter=maxiter,
            objective=objective, snr_metric=snr_metric,
            snr_cutoff_dB=snr_cutoff_dB, power_limit_dBm=power_limit_dbm,
            chans_per_band=chans_per_band, history=start_hist,
        )

        for h in start_hist:
            h["start"] = label
            h["nfev"] = len(history) + 1
            history.append(h)

        powers = _apply_power_limit(
            {b: float(solution.x[i]) for i, b in enumerate(active_bands)},
            power_limit_dbm, chans_per_band,
        )
        cap, snr = eval_network_metrics(
            powers, active_bands, grid, precomp_list, w_to_band,
            snr_metric=snr_metric,
        )
        results.append({
            "label": label,
            "powers": powers,
            "cap": cap,
            "snr": snr,
            "success": bool(solution.success),
            "n_evals": len(start_hist),
        })
        print(f"  -> start '{label}' done: {cap / 1e12:.4f} Tbps, "
              f"{len(start_hist)} evals, success={solution.success}")
    total_time = time.perf_counter() - t_start

    print("\n  --- Multi-start summary ---")
    for r in sorted(results, key=lambda r: -r["cap"]):
        pwr_str = "  ".join(f"{b}={r['powers'][b]:+5.2f}" for b in active_bands)
        print(f"    {r['label']:>8s}: {r['cap'] / 1e12:9.4f} Tbps  [{pwr_str}]")

    best = max(results, key=lambda r: r["cap"])
    best_powers = best["powers"]
    best_cap = best["cap"]
    best_snr = best["snr"]

    tag = (f"{topology_name}_{band_selection}_{route_function}_"
           f"{objective}_{snr_metric}")

    # Final per-band result CSV
    csv_path = out_dir / f"network_launch_power_lbfgsb_{tag}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "topology", "rwa_key", "band_selection", "band", "power_dBm",
            "snr_band_dB", "total_capacity_Tbps", "objective", "snr_metric",
            "power_limit_dBm", "span_length_km", "n_evals", "success", "time_s",
            "best_start", "n_starts",
        ])
        w.writeheader()
        for b in active_bands:
            w.writerow({
                "topology": topology_name,
                "rwa_key": rwa_key,
                "band_selection": band_selection,
                "band": b,
                "power_dBm": best_powers[b],
                "snr_band_dB": best_snr.get(b, float("nan")),
                "total_capacity_Tbps": best_cap / 1e12,
                "objective": objective,
                "snr_metric": snr_metric,
                "power_limit_dBm": (power_limit_dbm
                                    if power_limit_dbm is not None else ""),
                "span_length_km": span_length_km,
                "n_evals": len(history),
                "success": best["success"],
                "time_s": total_time,
                "best_start": best["label"],
                "n_starts": len(starts),
            })
    print(f"\nSaved result CSV: {csv_path}")

    # Optimisation history CSV (one row per objective evaluation)
    hist_path = out_dir / f"network_launch_power_lbfgsb_{tag}_history.csv"
    with open(hist_path, "w", newline="") as f:
        fields = (["nfev", "start", "cost", "capacity_Tbps", "time_s"]
                  + [f"P_{b}_dBm" for b in active_bands])
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for h in history:
            row = {
                "nfev": h["nfev"],
                "start": h.get("start", ""),
                "cost": h["cost"],
                "capacity_Tbps": h["capacity_Tbps"],
                "time_s": h["time_s"],
            }
            for b in active_bands:
                row[f"P_{b}_dBm"] = h["powers_dBm"][b]
            w.writerow(row)
    print(f"Saved history CSV: {hist_path}")

    plot_convergence(history, active_bands,
                     out_dir / f"network_launch_power_lbfgsb_{tag}.png")

    print("\n" + "=" * 65)
    print("OPTIMAL JOINT PER-BAND LAUNCH POWERS (multi-start L-BFGS-B)")
    print("=" * 65)
    for b in active_bands:
        print(f"  '{b}': {{'p_launch_opt': {best_powers[b]:+.2f}}},  "
              f"# SNR={best_snr.get(b, float('nan')):.2f} dB")
    print("-" * 65)
    print(f"  Total capacity : {best_cap / 1e12:.4f} Tbps")
    print(f"  Objective      : {objective}")
    print(f"  Best start     : '{best['label']}' of {len(starts)} starts "
          f"(success={best['success']})")
    print(f"  Evaluations    : {len(history)} total across all starts")
    print(f"  Total time     : {total_time:.1f} s")
    print("=" * 65)
    print("All bands optimised jointly with finite-difference L-BFGS-B. Done.\n")


def _parse_cli():
    p = argparse.ArgumentParser(
        description="Joint per-band launch power optimisation on a network "
                    "(NSFNET) via RWA, simulation.py-style L-BFGS-B")
    p.add_argument("--db", type=str, default=None)
    p.add_argument("--collection", type=str, default=None)
    p.add_argument("--topology", type=str, default=None)
    p.add_argument("--route_function", type=str, default=None)
    p.add_argument("--band", type=str, default=None,
                   choices=list(cfm.BAND_CONFIGS.keys()))
    p.add_argument("--span_length_km", type=float, default=None)
    p.add_argument("--init_power_dBm", type=float, default=None)
    p.add_argument("--bound_lo_dBm", type=float, default=None)
    p.add_argument("--bound_hi_dBm", type=float, default=None)
    p.add_argument("--ftol", type=float, default=None)
    p.add_argument("--gtol", type=float, default=None)
    p.add_argument("--eps", type=float, default=None)
    p.add_argument("--maxiter", type=int, default=None)
    p.add_argument("--n_random_starts", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--coarse_to_fine", dest="coarse_to_fine",
                   action="store_true", default=None)
    p.add_argument("--no_coarse_to_fine", dest="coarse_to_fine",
                   action="store_false", default=None)
    p.add_argument("--objective", type=str, default=None,
                   choices=OBJECTIVE_CHOICES)
    p.add_argument("--snr_cutoff_dB", type=float, default=None)
    p.add_argument("--power_limit_dBm", type=float, default=None)
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--snr_metric", type=str, default=None,
                   choices=SNR_METRIC_CHOICES)
    return p.parse_args()


if __name__ == "__main__":
    # ================================================================
    #  USER CONFIGURATION  (edit here, then run this file)
    # ================================================================
    DB = "Topology_Data"
    COLLECTION = "real"
    TOPOLOGY_NAME = "NSFNET"
    ROUTE_FUNCTION = "kSP-FF"
    BAND_SELECTION = "OESCL"

    SPAN_LENGTH_KM = 80.0
    INIT_POWER_DBM = 0.0          # starting launch power for every band
    BOUND_LO_DBM = -5.0           # box bound lower (cf. simulation.py Bounds)
    BOUND_HI_DBM = 2.0            # box bound upper
    # L-BFGS-B's ftol test is RELATIVE to max(|f|, 1); with |cost| ~ 1400 Tbps
    # the old tol=1e-2 stopped once an iteration gained < ~14 Tbps.  These
    # explicit values stop at ~1e-4 Tbps improvement / tiny projected gradient.
    FTOL = 1e-7                   # relative per-iteration improvement threshold
    GTOL = 1e-5                   # projected-gradient threshold [Tbps/dB]
    EPS = 1e-1                    # finite-difference step (simulation.py eps=1e-1)
    MAXITER = 100
    COARSE_TO_FINE = False        # multi-start supersedes the coarse seeding pass

    # ---- Multi-start (local L-BFGS-B from several x0, keep the best) ----
    # Explicit warm starts: dict band -> dBm.  E.g. the PSO gbest from
    # network_pso.py (NSFNET/OESCL, ~1400 Tbps) so L-BFGS-B refines it.
    X0_LIST = [
        {"O": -1.65, "E": -2.70, "S": -2.24, "C": -4.76, "L": -4.53},
    ]
    N_RANDOM_STARTS = 3           # extra uniform-random starts in the bounds
    SEED = 42                     # RNG seed for the random starts
    OBJECTIVE = "capacity"        # "capacity" (sim_num=3) | "snr_mean_std" (4/5)
    SNR_CUTOFF_DB = None          # e.g. 4.0 to drop low-SNR bands (sim_num=7/8)
    POWER_LIMIT_DBM = None        # e.g. 23.0 to cap total launch power; None=off
    OUTPUT_DIR = None             # None -> <this script directory>/data
    SNR_METRIC = "mean_linear_dB"

    cli = _parse_cli()
    run(
        db=cli.db or DB,
        collection=cli.collection or COLLECTION,
        topology_name=cli.topology or TOPOLOGY_NAME,
        route_function=cli.route_function or ROUTE_FUNCTION,
        band_selection=cli.band or BAND_SELECTION,
        span_length_km=(cli.span_length_km if cli.span_length_km is not None
                        else SPAN_LENGTH_KM),
        init_power_dbm=(cli.init_power_dBm if cli.init_power_dBm is not None
                        else INIT_POWER_DBM),
        bound_lo_dbm=(cli.bound_lo_dBm if cli.bound_lo_dBm is not None
                      else BOUND_LO_DBM),
        bound_hi_dbm=(cli.bound_hi_dBm if cli.bound_hi_dBm is not None
                      else BOUND_HI_DBM),
        ftol=(cli.ftol if cli.ftol is not None else FTOL),
        gtol=(cli.gtol if cli.gtol is not None else GTOL),
        eps=(cli.eps if cli.eps is not None else EPS),
        maxiter=(cli.maxiter if cli.maxiter is not None else MAXITER),
        coarse_to_fine=(cli.coarse_to_fine if cli.coarse_to_fine is not None
                        else COARSE_TO_FINE),
        objective=(cli.objective or OBJECTIVE),
        snr_cutoff_dB=(cli.snr_cutoff_dB if cli.snr_cutoff_dB is not None
                       else SNR_CUTOFF_DB),
        power_limit_dbm=(cli.power_limit_dBm if cli.power_limit_dBm is not None
                         else POWER_LIMIT_DBM),
        output_dir=cli.output_dir or OUTPUT_DIR,
        snr_metric=cli.snr_metric or SNR_METRIC,
        x0_list=X0_LIST,
        n_random_starts=(cli.n_random_starts
                         if cli.n_random_starts is not None
                         else N_RANDOM_STARTS),
        seed=(cli.seed if cli.seed is not None else SEED),
    )
