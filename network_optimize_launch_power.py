"""
Per-band launch power 1-D sweep on a real network (e.g. NSFNET) with RWA.

Reads a topology and its band-aware RWA from MongoDB and evaluates the total
network throughput with the CFM NLI model of ``compute_throughput_cfm_new``
(per-link NSR with RWA occupancy, capacity summed over lightpaths).

For each band we sweep its per-channel launch power while every other band
stays fixed at ``DEFAULT_POWER_DBM`` and the RWA occupancy is unchanged
("default_power" mode -- the network analogue that matches a real multi-band
deployment).  No joint optimisation: the optimum per band is the argmax of its
1-D sweep.

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

_SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
_EXTERNAL_DIR = _SCRIPT_DIR.parents[2] / "external"
_ONG_SRC = _EXTERNAL_DIR / "ong-python-toolbox" / "src"
for _p in (str(_SCRIPT_DIR), str(_EXTERNAL_DIR), str(_ONG_SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import compute_throughput_cfm_new as cfm  # noqa: E402  (path setup above)
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
# MongoDB topology + RWA loading
# ---------------------------------------------------------------------------
def load_topologies(db, collection, topology_name, route_function, band_selection):
    """Return ``[(doc_id, graph, rwa), ...]`` for ``topology_name``."""
    rwa_key = cfm._build_band_aware_rwa_key(route_function, band_selection)
    graph_list = nt.Database.read_topology_dataset_list(
        db, collection, find_dic={"name": topology_name}, node_data=True
    )
    print(f"Found {len(graph_list)} topologies for '{topology_name}'")

    out = []
    for graph, _id in graph_list:
        doc = list(nt.Database.read_data(
            db, collection, find_dic={"_id": _id}, max_count=1))
        if not doc:
            print(f"[WARN] No document for _id={_id}; skipped")
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
    """Per-topology RWA structures (built once, reused for every power)."""
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
# Network capacity + per-band path SNR for a given launch power dict
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
# Per-band 1-D sweep (default_power mode)
# ---------------------------------------------------------------------------
def sweep_band(target_band, default_powers, active_bands, grid, precomp_list,
               w_to_band, power_range_dBm, steps, snr_metric: str):
    powers = np.linspace(power_range_dBm[0], power_range_dBm[1], steps)
    results = []
    for p in powers:
        bp = dict(default_powers)
        bp[target_band] = float(p)
        t0 = time.perf_counter()
        cap, snr_bands = eval_network_metrics(
            bp, active_bands, grid, precomp_list, w_to_band,
            snr_metric=snr_metric,
        )
        dt = time.perf_counter() - t0
        snr_tgt = snr_bands.get(target_band, float("nan"))
        results.append({
            "power_dBm": float(p),
            "capacity_Tbps": cap / 1e12,
            "snr_target_band_dB": snr_tgt,
            "snr_all": snr_bands,
            "time_s": dt,
            "snr_metric": snr_metric,
        })
        print(
            f"    {target_band}-band: P={p:+6.2f} dBm  "
            f"cap={cap/1e12:.4f} Tbps  "
            f"SNR({target_band},{snr_metric})={snr_tgt:.2f} dB  "
            f"({dt:.1f}s)"
        )
    best = max(results, key=lambda r: r["capacity_Tbps"])
    return powers, results, best


def plot_sweeps(sweep_data, active_bands, out_path):
    try:
        from matplotlib import pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not available; skipping plot.")
        return
    n = len(active_bands)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)
    for ax, band in zip(axes[0], active_bands):
        powers, results, best = sweep_data[band]
        caps = [r["capacity_Tbps"] for r in results]
        ax.plot(powers, caps, "s-", color="tab:orange")
        ax.axvline(best["power_dBm"], color="red", ls="--", alpha=0.7,
                   label=f"best={best['power_dBm']:+.1f} dBm")
        ax.set_xlabel("Launch Power [dBm]")
        ax.set_ylabel("Network Capacity [Tbps]")
        ax.set_title(f"{band}-band (others @ default P)")
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot: {out_path}")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def run(*, db, collection, topology_name, route_function, band_selection,
        span_length_km, default_power_dbm, sweep_lo_dbm, sweep_hi_dbm,
        sweep_steps, output_dir, snr_metric: str = "mean_linear_dB"):
    if band_selection not in cfm.BAND_CONFIGS:
        raise ValueError(
            f"band_selection must be one of {list(cfm.BAND_CONFIGS.keys())}, "
            f"got {band_selection!r}"
        )
    if snr_metric not in SNR_METRIC_CHOICES:
        raise ValueError(f"snr_metric must be one of {SNR_METRIC_CHOICES}")
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

    print("=" * 65)
    print("Network per-band launch power sweep (default_power mode)")
    print("=" * 65)
    print(f"Topology           : {topology_name} ({len(topos)} doc(s))")
    print(f"RWA key            : {rwa_key}")
    print(f"Bands              : {band_selection} -> {active_bands}")
    print(f"Active channels    : {num_active}")
    print(f"Span length        : {span_length_km} km")
    print(f"default_power_dBm  : {default_power_dbm} dBm")
    print(f"Sweep range        : [{sweep_lo_dbm}, {sweep_hi_dbm}] dBm, "
          f"{int(sweep_steps)} steps")
    print(f"Output dir         : {out_dir}")
    print(f"snr_metric         : {snr_metric} (mean over lit path NSR per band)")
    print("Non-swept bands    : fixed @ default_power; RWA occupancy unchanged")
    print("=" * 65)

    default_powers = {b: float(default_power_dbm) for b in active_bands}
    sweep_data = {}
    csv_rows = []

    for band in active_bands:
        print(f"\n  --- Sweeping {band}-band ---")
        powers, results, best = sweep_band(
            band, default_powers, active_bands, grid, precomp_list,
            w_to_band,
            power_range_dBm=(sweep_lo_dbm, sweep_hi_dbm), steps=int(sweep_steps),
            snr_metric=snr_metric,
        )
        sweep_data[band] = (powers, results, best)
        print(f"  Best for {band}-band: P={best['power_dBm']:+.2f} dBm, "
              f"cap={best['capacity_Tbps']:.4f} Tbps, "
              f"SNR({snr_metric})={best['snr_target_band_dB']:.2f} dB")
        for r in results:
            csv_rows.append({
                "topology": topology_name,
                "rwa_key": rwa_key,
                "band_selection": band_selection,
                "swept_band": band,
                "power_dBm": r["power_dBm"],
                "capacity_Tbps": r["capacity_Tbps"],
                "snr_target_dB": r["snr_target_band_dB"],
                "default_power_dBm": float(default_power_dbm),
                "span_length_km": span_length_km,
                "snr_metric": snr_metric,
            })

    tag = f"{topology_name}_{band_selection}_{route_function}_{snr_metric}"
    csv_path = out_dir / f"network_launch_power_sweep_{tag}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        w.writeheader()
        w.writerows(csv_rows)
    print(f"\nSaved sweep CSV: {csv_path}")

    plot_sweeps(sweep_data, active_bands,
                out_dir / f"network_launch_power_sweep_{tag}.png")

    print("\n" + "=" * 65)
    print("OPTIMAL PER-BAND LAUNCH POWERS (per-band 1-D sweep argmax)")
    print("=" * 65)
    for b in active_bands:
        print(f"  '{b}': {{'p_launch_opt': {sweep_data[b][2]['power_dBm']:+.2f}}},")
    print("=" * 65)
    print("Each band optimised independently; others held at default_power. "
          "Done.\n")


def _parse_cli():
    p = argparse.ArgumentParser(
        description="Per-band launch power sweep on a network (NSFNET) via RWA")
    p.add_argument("--db", type=str, default=None)
    p.add_argument("--collection", type=str, default=None)
    p.add_argument("--topology", type=str, default=None)
    p.add_argument("--route_function", type=str, default=None)
    p.add_argument("--band", type=str, default=None,
                   choices=list(cfm.BAND_CONFIGS.keys()))
    p.add_argument("--span_length_km", type=float, default=None)
    p.add_argument("--default_power_dBm", type=float, default=None)
    p.add_argument("--sweep_lo_dBm", type=float, default=None)
    p.add_argument("--sweep_hi_dBm", type=float, default=None)
    p.add_argument("--sweep_steps", type=int, default=None)
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
    DEFAULT_POWER_DBM = 0.0
    SWEEP_LO_DBM = -4.0
    SWEEP_HI_DBM = 4.0
    SWEEP_STEPS = 21
    OUTPUT_DIR = None  # None -> <this script directory>/data
    # "mean_linear_dB" or "mean_dB" (over lit lightpath end-to-end NSR per band)
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
        default_power_dbm=(cli.default_power_dBm
                           if cli.default_power_dBm is not None
                           else DEFAULT_POWER_DBM),
        sweep_lo_dbm=(cli.sweep_lo_dBm if cli.sweep_lo_dBm is not None
                      else SWEEP_LO_DBM),
        sweep_hi_dbm=(cli.sweep_hi_dBm if cli.sweep_hi_dBm is not None
                      else SWEEP_HI_DBM),
        sweep_steps=(cli.sweep_steps if cli.sweep_steps is not None
                     else SWEEP_STEPS),
        output_dir=cli.output_dir or OUTPUT_DIR,
        snr_metric=cli.snr_metric or SNR_METRIC,
    )
