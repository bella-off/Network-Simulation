"""
Joint per-band launch power optimisation via Particle Swarm Optimization (PSO).

Replaces the 1-D per-band sweep with a JOINT multi-dimensional PSO search
across all active bands simultaneously. This captures inter-band interactions
(ISRS-induced power tilt, cross-band NLI, OSNR coupling) that a sequential
per-band sweep misses by construction.

Decision variables : per-band launch power [dBm], one per active band
                     (search-space dimension D = len(active_bands))
Objective          : total network throughput [bps], computed by the
                     CFM NLI model with the existing RWA occupancy
Search method      : standard PSO with linear decreasing inertia weight,
                     velocity clamping, reflective boundary handling, and
                     stagnation-based early stopping.

Edit the USER CONFIGURATION block (or override via CLI) and run.
"""
from __future__ import annotations

import argparse
import csv
import pathlib
import sys
import time

import networkx as nx
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)


def _log(*args, **kwargs):
    """Print with immediate flush so long runs show progress in the terminal."""
    kwargs.setdefault("flush", True)
    print(*args, **kwargs)

_SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
_EXTERNAL_DIR = _SCRIPT_DIR.parents[2] / "external"
_ONG_SRC = _EXTERNAL_DIR / "ong-python-toolbox" / "src"
for _p in (str(_SCRIPT_DIR), str(_EXTERNAL_DIR), str(_ONG_SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import compute_throughput_cfm_new as cfm  # noqa: E402
import NetworkToolkit as nt  # noqa: E402

ALL_BANDS = ["O", "E", "S", "C", "L"]

SNR_METRIC_CHOICES = ("mean_dB", "mean_linear_dB")


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


def _build_wavelength_band_map(active_slot_ix, band_masks, active_bands):
    """Map RWA wavelength index -> band letter."""
    w_to_band = {}
    for w in range(len(active_slot_ix)):
        slot = int(active_slot_ix[w])
        for b in active_bands:
            if band_masks[b][slot]:
                w_to_band[w] = b
                break
    return w_to_band


# ---------------------------------------------------------------------------
# MongoDB topology + RWA loading (unchanged from sweep version)
# ---------------------------------------------------------------------------
def load_topologies(db, collection, topology_name, route_function, band_selection):
    """Return ``[(doc_id, graph, rwa), ...]`` for ``topology_name``."""
    rwa_key = cfm._build_band_aware_rwa_key(route_function, band_selection)
    graph_list = nt.Database.read_topology_dataset_list(
        db, collection, find_dic={"name": topology_name}, node_data=True
    )
    _log(f"Found {len(graph_list)} topologies for '{topology_name}'")

    out = []
    for graph, _id in graph_list:
        doc = list(nt.Database.read_data(
            db, collection, find_dic={"_id": _id}, max_count=1))
        if not doc:
            _log(f"[WARN] No document for _id={_id}; skipped")
            continue
        if rwa_key not in doc[0]:
            raise KeyError(
                f"Band-aware key '{rwa_key}' not found for _id={_id}. "
                "Stop run to avoid using mismatched legacy RWA keys."
            )
        rwa = nt.Tools.read_database_dict(doc[0][rwa_key])
        graph = nx.relabel.convert_node_labels_to_integers(graph, first_label=1)
        out.append((_id, graph, rwa))
    return out, rwa_key


def precompute(graph, rwa, num_active):
    """Per-topology RWA structures (built once, reused for every fitness eval)."""
    (edges, edge_to_row, span_count_per_edge,
     occupancy_matrix, rwa_edge_paths, wavelength_to_col) = (
        cfm._build_rwa_occupancy(graph, rwa, num_active)
    )
    return {
        "edges": edges,
        "edge_to_row": edge_to_row,
        "span_count": span_count_per_edge,
        "occupancy": occupancy_matrix,
        "rwa_edge_paths": rwa_edge_paths,
        "wavelength_to_col": wavelength_to_col,
        "n_links": int(span_count_per_edge.shape[0]),
    }


# ---------------------------------------------------------------------------
# Capacity objective + per-band path SNR (PSO hits this many times)
# ---------------------------------------------------------------------------
def eval_network_metrics(band_power, active_bands, grid, precomp_list,
                         w_to_band, snr_metric: str = "mean_linear_dB"):
    """Return total throughput [bps] and per-band mean path SNR [dB]."""
    if snr_metric not in SNR_METRIC_CHOICES:
        raise ValueError(f"snr_metric must be one of {SNR_METRIC_CHOICES}")

    ch_lambda, channel_idx, ch_idx_oband_active, band_masks, ref_lambda, \
        span_length_km, active_slot_ix = grid

    P_channel, nf, snr_trx = cfm._build_per_channel_params(
        ch_lambda, band_masks, active_bands, launch_power_dBm=band_power,
    )
    setup = cfm._build_setup(
        span_length_km, ch_lambda, channel_idx, P_channel, nf, snr_trx,
        ref_lambda=ref_lambda,
    )
    ch_bw_col = np.asarray(setup.ch_bandwidth_ij)

    total_bps = 0.0
    path_nsr_by_band = {b: [] for b in active_bands}
    for pc in precomp_list:
        occ = pc["occupancy"]
        nsr_link = np.empty_like(occ, dtype=np.float64)
        for i in range(pc["n_links"]):
            link_nsr, *_ = cfm.calc_NSR_link(
                setup, float(pc["span_count"][i]),
                occ[i, :, None], ch_idx_oband_active,
            )
            nsr_link[i] = np.asarray(link_nsr).squeeze(-1)

        edge_to_row = pc["edge_to_row"]
        w2c = pc["wavelength_to_col"]
        for w_str, path_infos in pc["rwa_edge_paths"].items():
            w = int(w_str)
            if w not in w2c:
                continue
            col = w2c[w]
            band = w_to_band.get(w)
            ch_bw_hz = float(ch_bw_col[int(active_slot_ix[w]), 0])
            for info in path_infos:
                path_nsr = 0.0
                for edge in info["edge_path"]:
                    path_nsr += float(nsr_link[edge_to_row[edge], col])
                if not np.isfinite(path_nsr) or path_nsr <= 0:
                    continue
                total_bps += 2 * ch_bw_hz * np.log2(1.0 + 1.0 / path_nsr)
                if band is not None:
                    path_nsr_by_band[band].append(path_nsr)

    snr_bands = {
        b: _aggregate_snr_dB(np.asarray(path_nsr_by_band[b]), snr_metric)
        for b in active_bands
    }
    return total_bps, snr_bands


# ---------------------------------------------------------------------------
# PSO core
# ---------------------------------------------------------------------------
def pso_optimize(
    objective,
    active_bands,
    bounds_dBm,
    n_particles=20,
    max_iter=30,
    w_max=0.9,
    w_min=0.4,
    c1=1.5,
    c2=1.5,
    v_max_frac=0.2,
    stagnation_patience=8,
    seed=None,
    verbose=True,
    particle_verbose=True,
    on_iteration=None,
):
    """Joint PSO over per-band launch powers (MAXIMISATION).

    Parameters
    ----------
    objective : callable
        ``objective(dict[band -> dBm]) -> (float, dict[band -> SNR dB])``.
        Fitness is capacity in Tbps (higher is better). PSO maximises fitness.
    active_bands : list[str]
        Decision dimensions, e.g. ['O','E','S','C','L'].
    bounds_dBm : tuple(float, float)
        (lo, hi) launch power bounds, shared by all bands.
    n_particles : int
        Swarm size N.
    max_iter : int
        Maximum number of PSO iterations (excluding initialisation).
    w_max, w_min : float
        Inertia weight schedule. Linear decreasing from w_max (iter 1) to
        w_min (iter max_iter). Early high w favours exploration; late low w
        favours exploitation.
    c1, c2 : float
        Cognitive / social acceleration coefficients.
    v_max_frac : float
        Velocity clamp as a fraction of the per-dimension search range.
        |v_max| = v_max_frac * (hi - lo).
    stagnation_patience : int
        Early stopping: terminate if gbest has not improved for this many
        consecutive iterations.
    seed : int | None
        RNG seed for reproducibility.
    verbose : bool
        Print per-iteration PSO summary.
    particle_verbose : bool
        Print one line after each particle fitness evaluation (slow but
        confirms the run is alive during long CFM evals).
    on_iteration : callable | None
        Optional ``on_iteration(record: dict) -> None`` invoked after init and
        each PSO iteration (for incremental CSV / checkpointing).

    Returns
    -------
    best_powers : dict[band -> dBm]
    best_fitness : float (Tbps)
    history : dict with per-iter records (for plotting / CSV export)
    """
    rng = np.random.default_rng(seed)
    D = len(active_bands)
    lo, hi = float(bounds_dBm[0]), float(bounds_dBm[1])
    bounds_lo = np.full(D, lo)
    bounds_hi = np.full(D, hi)
    range_d = bounds_hi - bounds_lo
    v_max = v_max_frac * range_d

    # ---- Initialise positions and velocities ----
    X = rng.uniform(bounds_lo, bounds_hi, size=(n_particles, D))
    V = rng.uniform(-v_max, v_max, size=(n_particles, D))

    def _eval_row(x):
        bp = {band: float(x[i]) for i, band in enumerate(active_bands)}
        fit, snr = objective(bp)
        return float(fit), snr

    def _eval_swarm(positions, *, stage: str, it: int | None = None):
        evals = []
        n = positions.shape[0]
        for i in range(n):
            t_eval = time.perf_counter()
            result = _eval_row(positions[i])
            dt_eval = time.perf_counter() - t_eval
            evals.append(result)
            if particle_verbose:
                fit_i, snr_i = result
                if verbose and it is not None:
                    ctx = f"it={it:3d}/{max_iter}"
                else:
                    ctx = stage
                _log(f"[PSO] {ctx} eval {i + 1:2d}/{n} | "
                     f"fit={fit_i:.4f} Tbps | {dt_eval:.1f}s")
            elif verbose and ((i + 1) % max(1, n // 4) == 0 or i + 1 == n):
                _log(f"[PSO] {stage} eval {i + 1}/{n} ...")
        return evals

    def _record_iteration(it, w_val, gbest_fit_val, gbest_pos, gbest_snr_val,
                          swarm_mean, swarm_std):
        if on_iteration is None:
            return
        record = {
            "iter": it,
            "gbest_fit_Tbps": gbest_fit_val,
            "swarm_mean_Tbps": swarm_mean,
            "swarm_std_x": swarm_std,
            "w": w_val,
            "gbest_powers": {
                b: float(gbest_pos[i]) for i, b in enumerate(active_bands)
            },
            "gbest_snr_bands": dict(gbest_snr_val),
        }
        on_iteration(record)

    if verbose:
        _log(f"[PSO] Initialising swarm: evaluating {n_particles} particles "
             f"(first summary line appears after all {n_particles} finish)...")

    evals = _eval_swarm(X, stage="init")
    fitness = np.array([e[0] for e in evals])
    snr_rows = [e[1] for e in evals]
    pbest = X.copy()
    pbest_fit = fitness.copy()
    pbest_snr = [dict(s) for s in snr_rows]

    g_idx = int(np.argmax(pbest_fit))
    gbest = pbest[g_idx].copy()
    gbest_fit = float(pbest_fit[g_idx])
    gbest_snr = dict(pbest_snr[g_idx])

    history = {
        "iter": [0],
        "gbest_fit_Tbps": [gbest_fit],
        "gbest_powers": [gbest.copy()],
        "gbest_snr_bands": [gbest_snr],
        "swarm_mean_Tbps": [float(np.mean(fitness))],
        "swarm_std_x": [float(np.mean(np.std(X, axis=0)))],
        "w": [w_max],
        "all_positions": [X.copy()],
        "all_fitness": [fitness.copy()],
    }

    if verbose:
        bp_str = ", ".join(f"{b}={gbest[i]:+.2f}"
                           for i, b in enumerate(active_bands))
        snr_str = ", ".join(f"{b}={gbest_snr[b]:.1f}dB"
                            for b in active_bands)
        _log(f"[PSO] init   | gbest={gbest_fit:.4f} Tbps | {bp_str} | SNR: {snr_str}")

    _record_iteration(
        0, w_max, gbest_fit, gbest, gbest_snr,
        float(np.mean(fitness)), float(np.mean(np.std(X, axis=0))),
    )

    stagnation = 0

    # ---- Main loop ----
    for it in range(1, max_iter + 1):
        # Linear decreasing inertia weight
        if max_iter > 1:
            w = w_max - (w_max - w_min) * (it - 1) / (max_iter - 1)
        else:
            w = w_max

        r1 = rng.uniform(0, 1, size=(n_particles, D))
        r2 = rng.uniform(0, 1, size=(n_particles, D))
        V = (w * V
             + c1 * r1 * (pbest - X)
             + c2 * r2 * (gbest - X))
        # Velocity clamping (per dimension)
        V = np.clip(V, -v_max, v_max)

        X_new = X + V

        # Reflective boundary handling: bounce + reverse velocity component
        below = X_new < bounds_lo
        above = X_new > bounds_hi
        X_new = np.where(below, 2 * bounds_lo - X_new, X_new)
        X_new = np.where(above, 2 * bounds_hi - X_new, X_new)
        X_new = np.clip(X_new, bounds_lo, bounds_hi)  # safety net
        V = np.where(below | above, -V, V)

        X = X_new
        evals = _eval_swarm(X, stage="iter", it=it)
        fitness = np.array([e[0] for e in evals])
        snr_rows = [e[1] for e in evals]

        # pbest update
        improved = fitness > pbest_fit
        pbest[improved] = X[improved]
        pbest_fit[improved] = fitness[improved]
        for i in np.flatnonzero(improved):
            pbest_snr[i] = dict(snr_rows[i])

        # gbest update
        g_idx = int(np.argmax(pbest_fit))
        new_best = float(pbest_fit[g_idx])
        if new_best > gbest_fit + 1e-9:
            gbest = pbest[g_idx].copy()
            gbest_fit = new_best
            stagnation = 0
        else:
            stagnation += 1
        gbest_snr = dict(pbest_snr[g_idx])

        history["iter"].append(it)
        history["gbest_fit_Tbps"].append(gbest_fit)
        history["gbest_powers"].append(gbest.copy())
        history["gbest_snr_bands"].append(gbest_snr)
        history["swarm_mean_Tbps"].append(float(np.mean(fitness)))
        history["swarm_std_x"].append(float(np.mean(np.std(X, axis=0))))
        history["w"].append(float(w))
        history["all_positions"].append(X.copy())
        history["all_fitness"].append(fitness.copy())

        if verbose:
            bp_str = ", ".join(f"{b}={gbest[i]:+.2f}"
                               for i, b in enumerate(active_bands))
            snr_str = ", ".join(f"{b}={gbest_snr[b]:.1f}dB"
                                for b in active_bands)
            _log(f"[PSO] it={it:3d}/{max_iter} w={w:.3f} | "
                 f"gbest={gbest_fit:.4f} Tbps | mean={np.mean(fitness):.4f} | "
                 f"spread={np.mean(np.std(X, axis=0)):.3f} dB | {bp_str} | "
                 f"SNR: {snr_str}")

        _record_iteration(
            it, float(w), gbest_fit, gbest, gbest_snr,
            float(np.mean(fitness)), float(np.mean(np.std(X, axis=0))),
        )

        if stagnation >= stagnation_patience:
            if verbose:
                _log(f"[PSO] Early stop: no gbest improvement for "
                     f"{stagnation_patience} consecutive iters.")
            break

    best_powers = {b: float(gbest[i]) for i, b in enumerate(active_bands)}
    return best_powers, gbest_fit, history


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_pso(history, active_bands, out_path):
    try:
        from matplotlib import pyplot as plt
    except ImportError:
        _log("[WARN] matplotlib not available; skipping plot.")
        return

    iters = np.array(history["iter"])
    gbest_fit = np.array(history["gbest_fit_Tbps"])
    swarm_mean = np.array(history["swarm_mean_Tbps"])
    swarm_std = np.array(history["swarm_std_x"])
    gbest_powers = np.array(history["gbest_powers"])  # (n_iters, D)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # (a) Convergence: gbest and swarm mean fitness
    ax = axes[0, 0]
    ax.plot(iters, gbest_fit, "o-", color="tab:red", label="gbest (best so far)")
    ax.plot(iters, swarm_mean, "s-", color="tab:blue", alpha=0.6,
            label="swarm mean fitness")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Network capacity [Tbps]")
    ax.set_title("(a) PSO convergence")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # (b) Swarm diversity: should shrink as PSO transitions exploration -> exploit
    ax = axes[0, 1]
    ax.plot(iters, swarm_std, "o-", color="tab:purple")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Mean per-dim std of X [dB]")
    ax.set_title("(b) Swarm diversity (exploration -> exploitation)")
    ax.grid(True, alpha=0.3)

    # (c) gbest power trajectories per band
    ax = axes[1, 0]
    for d, band in enumerate(active_bands):
        ax.plot(iters, gbest_powers[:, d], "o-", label=f"{band}-band")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("gbest launch power [dBm]")
    ax.set_title("(c) Best-so-far per-band launch powers")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)

    # (d) Final swarm in parallel-coordinates view
    ax = axes[1, 1]
    X_final = history["all_positions"][-1]
    fit_final = history["all_fitness"][-1]
    ptp = fit_final.max() - fit_final.min()
    fit_norm = (fit_final - fit_final.min()) / (ptp + 1e-12)
    for i in range(X_final.shape[0]):
        ax.plot(range(len(active_bands)), X_final[i], "-",
                alpha=0.25 + 0.75 * fit_norm[i],
                color=plt.cm.viridis(fit_norm[i]),
                lw=1)
    ax.plot(range(len(active_bands)), gbest_powers[-1], "k-", lw=3,
            label="gbest", marker="o")
    ax.set_xticks(range(len(active_bands)))
    ax.set_xticklabels(active_bands)
    ax.set_xlabel("Band")
    ax.set_ylabel("Launch power [dBm]")
    ax.set_title("(d) Final swarm -- parallel coordinates")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    _log(f"Saved plot: {out_path}")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def run(*, db, collection, topology_name, route_function, band_selection,
        span_length_km, sweep_lo_dbm, sweep_hi_dbm, output_dir,
        n_particles, max_iter, w_max, w_min, c1, c2,
        v_max_frac, stagnation_patience, seed,
        snr_metric: str = "mean_linear_dB",
        particle_verbose: bool = True,
        incremental_csv: bool = True):

    if band_selection not in cfm.BAND_CONFIGS:
        raise ValueError(
            f"band_selection must be one of {list(cfm.BAND_CONFIGS.keys())}, "
            f"got {band_selection!r}")
    if snr_metric not in SNR_METRIC_CHOICES:
        raise ValueError(f"snr_metric must be one of {SNR_METRIC_CHOICES}")
    active_bands = cfm.BAND_CONFIGS[band_selection]["bands"]

    out_dir = pathlib.Path(output_dir) if output_dir else _SCRIPT_DIR / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    _log("Loading topology + RWA from MongoDB ...")
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

    _log("Precomputing RWA occupancy structures ...")
    precomp_list = [precompute(g, rwa, num_active) for _, g, rwa in topos]
    w_to_band = _build_wavelength_band_map(
        active_slot_ix, band_masks, active_bands)

    tag = f"{topology_name}_{band_selection}_{route_function}_pso_{snr_metric}"
    csv_path = out_dir / f"network_launch_power_{tag}.csv"
    csv_fieldnames = (["iter", "gbest_fit_Tbps", "swarm_mean_Tbps",
                       "swarm_std_x", "w", "snr_metric"]
                      + [f"gbest_{b}_dBm" for b in active_bands]
                      + [f"gbest_{b}_SNR_dB" for b in active_bands])
    csv_file = None
    csv_writer = None

    def _append_csv_row(record):
        nonlocal csv_file, csv_writer
        if not incremental_csv:
            return
        if csv_writer is None:
            csv_file = open(csv_path, "w", newline="")
            csv_writer = csv.DictWriter(csv_file, fieldnames=csv_fieldnames)
            csv_writer.writeheader()
            csv_file.flush()
            _log(f"Incremental CSV: {csv_path}")
        row = {
            "iter": record["iter"],
            "gbest_fit_Tbps": record["gbest_fit_Tbps"],
            "swarm_mean_Tbps": record["swarm_mean_Tbps"],
            "swarm_std_x": record["swarm_std_x"],
            "w": record["w"],
            "snr_metric": snr_metric,
        }
        for b in active_bands:
            row[f"gbest_{b}_dBm"] = record["gbest_powers"][b]
            row[f"gbest_{b}_SNR_dB"] = record["gbest_snr_bands"].get(
                b, float("nan"))
        csv_writer.writerow(row)
        csv_file.flush()

    _log("=" * 72)
    _log("Joint per-band launch power optimisation via PSO")
    _log("=" * 72)
    _log(f"Topology           : {topology_name} ({len(topos)} doc(s))")
    _log(f"RWA key            : {rwa_key}")
    _log(f"Bands              : {band_selection} -> {active_bands}")
    _log(f"Search dim D       : {len(active_bands)}")
    _log(f"Active channels    : {num_active}")
    _log(f"Span length        : {span_length_km} km")
    _log(f"Power bounds [dBm] : [{sweep_lo_dbm}, {sweep_hi_dbm}]")
    _log(f"PSO                : N={n_particles}, max_iter={max_iter}, "
         f"w=[{w_min},{w_max}], c1={c1}, c2={c2}, "
         f"v_max_frac={v_max_frac}, patience={stagnation_patience}, "
         f"seed={seed}")
    _log(f"Max fitness evals  : {n_particles * (max_iter + 1)}  "
         f"(initialisation + max_iter rounds)")
    _log(f"Output dir         : {out_dir}")
    _log(f"snr_metric         : {snr_metric} (mean over lit path NSR per band)")
    _log(f"particle_verbose   : {particle_verbose}")
    _log(f"incremental_csv    : {incremental_csv}")
    _log("=" * 72)

    def objective(band_power):
        cap, snr_bands = eval_network_metrics(
            band_power, active_bands, grid, precomp_list, w_to_band,
            snr_metric=snr_metric,
        )
        return cap / 1e12, snr_bands

    t0 = time.perf_counter()
    try:
        best_powers, best_fit, history = pso_optimize(
            objective=objective,
            active_bands=active_bands,
            bounds_dBm=(sweep_lo_dbm, sweep_hi_dbm),
            n_particles=n_particles,
            max_iter=max_iter,
            w_max=w_max, w_min=w_min,
            c1=c1, c2=c2,
            v_max_frac=v_max_frac,
            stagnation_patience=stagnation_patience,
            seed=seed,
            particle_verbose=particle_verbose,
            on_iteration=_append_csv_row,
        )
    finally:
        if csv_file is not None:
            csv_file.close()
    dt = time.perf_counter() - t0

    if incremental_csv:
        _log(f"\nPSO history CSV: {csv_path}")
    else:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_fieldnames)
            writer.writeheader()
            for k in range(len(history["iter"])):
                row = {
                    "iter": history["iter"][k],
                    "gbest_fit_Tbps": history["gbest_fit_Tbps"][k],
                    "swarm_mean_Tbps": history["swarm_mean_Tbps"][k],
                    "swarm_std_x": history["swarm_std_x"][k],
                    "w": history["w"][k],
                    "snr_metric": snr_metric,
                }
                for d, b in enumerate(active_bands):
                    row[f"gbest_{b}_dBm"] = float(history["gbest_powers"][k][d])
                snr_k = history["gbest_snr_bands"][k]
                for b in active_bands:
                    row[f"gbest_{b}_SNR_dB"] = float(snr_k.get(b, float("nan")))
                writer.writerow(row)
        _log(f"\nSaved PSO history CSV: {csv_path}")

    plot_pso(history, active_bands,
             out_dir / f"network_launch_power_{tag}.png")

    _log("\n" + "=" * 72)
    _log("OPTIMAL PER-BAND LAUNCH POWERS (joint PSO argmax)")
    _log("=" * 72)
    for b in active_bands:
        _log(f"  '{b}': {{'p_launch_opt': {best_powers[b]:+.3f}}},")
    _log("-" * 72)
    _log(f"Best network capacity : {best_fit:.4f} Tbps")
    final_snr = history["gbest_snr_bands"][-1]
    for b in active_bands:
        _log(f"  gbest SNR({b},{snr_metric}) : {final_snr[b]:.2f} dB")
    _log(f"PSO wall time         : {dt:.1f} s")
    _log(f"Iterations executed   : {history['iter'][-1]} / {max_iter}")
    total_evals = n_particles * (history['iter'][-1] + 1)
    _log(f"Total fitness evals   : {total_evals}")
    _log("=" * 72 + "\n")


def _parse_cli():
    p = argparse.ArgumentParser(
        description="Joint per-band launch power PSO on a network via RWA")
    p.add_argument("--db", type=str, default=None)
    p.add_argument("--collection", type=str, default=None)
    p.add_argument("--topology", type=str, default=None)
    p.add_argument("--route_function", type=str, default=None)
    p.add_argument("--band", type=str, default=None,
                   choices=list(cfm.BAND_CONFIGS.keys()))
    p.add_argument("--span_length_km", type=float, default=None)
    p.add_argument("--sweep_lo_dBm", type=float, default=None)
    p.add_argument("--sweep_hi_dBm", type=float, default=None)
    p.add_argument("--output_dir", type=str, default=None)
    # PSO knobs
    p.add_argument("--n_particles", type=int, default=None)
    p.add_argument("--max_iter", type=int, default=None)
    p.add_argument("--w_max", type=float, default=None)
    p.add_argument("--w_min", type=float, default=None)
    p.add_argument("--c1", type=float, default=None)
    p.add_argument("--c2", type=float, default=None)
    p.add_argument("--v_max_frac", type=float, default=None)
    p.add_argument("--stagnation_patience", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--snr_metric", type=str, default=None,
                   choices=SNR_METRIC_CHOICES)
    p.add_argument("--no_particle_verbose", action="store_true",
                   help="Only print per-iteration summaries, not each particle")
    p.add_argument("--no_incremental_csv", action="store_true",
                   help="Write CSV only after PSO finishes")
    return p.parse_args()


if __name__ == "__main__":
    # ================================================================
    #  USER CONFIGURATION  (edit here, then run this file)
    # ================================================================
    DB = "Topology_Data"
    COLLECTION = "real"
    TOPOLOGY_NAME = "LondonDCI"
    ROUTE_FUNCTION = "kSP-FF"
    BAND_SELECTION = "OESCL"

    SPAN_LENGTH_KM = 80.0
    SWEEP_LO_DBM = -5.0     # per-band launch power lower bound
    SWEEP_HI_DBM = 2.0      # per-band launch power upper bound
    OUTPUT_DIR = None       # None -> <this script directory>/data

    # ---- PSO hyperparameters ----
    N_PARTICLES = 20        # swarm size (typical 15-30 for D=5)
    MAX_ITER = 30           # iteration cap; early-stop usually kicks in first
    W_MAX = 0.9             # initial inertia (favours exploration)
    W_MIN = 0.4             # final inertia (favours exploitation)
    C1 = 1.5                # cognitive coefficient
    C2 = 1.5                # social coefficient
    V_MAX_FRAC = 0.2        # |v_max| = 0.2 * (hi - lo); 1.6 dB per step here
    STAGNATION_PATIENCE = 8 # early-stop after this many no-improve iters
    SEED = 42               # set None for non-reproducible runs
    SNR_METRIC = "mean_linear_dB"
    PARTICLE_VERBOSE = True   # one line per particle eval (recommended for long runs)
    INCREMENTAL_CSV = True    # append CSV row after each PSO iteration

    cli = _parse_cli()
    _log("Starting network_pso.py ...")
    run(
        db=cli.db or DB,
        collection=cli.collection or COLLECTION,
        topology_name=cli.topology or TOPOLOGY_NAME,
        route_function=cli.route_function or ROUTE_FUNCTION,
        band_selection=cli.band or BAND_SELECTION,
        span_length_km=(cli.span_length_km if cli.span_length_km is not None
                        else SPAN_LENGTH_KM),
        sweep_lo_dbm=(cli.sweep_lo_dBm if cli.sweep_lo_dBm is not None
                      else SWEEP_LO_DBM),
        sweep_hi_dbm=(cli.sweep_hi_dBm if cli.sweep_hi_dBm is not None
                      else SWEEP_HI_DBM),
        output_dir=cli.output_dir or OUTPUT_DIR,
        n_particles=cli.n_particles or N_PARTICLES,
        max_iter=cli.max_iter or MAX_ITER,
        w_max=cli.w_max if cli.w_max is not None else W_MAX,
        w_min=cli.w_min if cli.w_min is not None else W_MIN,
        c1=cli.c1 if cli.c1 is not None else C1,
        c2=cli.c2 if cli.c2 is not None else C2,
        v_max_frac=(cli.v_max_frac if cli.v_max_frac is not None
                    else V_MAX_FRAC),
        stagnation_patience=(cli.stagnation_patience
                             if cli.stagnation_patience is not None
                             else STAGNATION_PATIENCE),
        seed=cli.seed if cli.seed is not None else SEED,
        snr_metric=cli.snr_metric or SNR_METRIC,
        particle_verbose=(PARTICLE_VERBOSE and not cli.no_particle_verbose),
        incremental_csv=(INCREMENTAL_CSV and not cli.no_incremental_csv),
    )