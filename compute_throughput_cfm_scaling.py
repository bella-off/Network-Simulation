"""
Compute CFM throughput on distance-scaled topologies.

Mirrors ``compute_throughput_cfm.py``: multi-band grid, ISRS + GN NLI per link
with occupancy mask, ASE from power evolution, transceiver SNR penalty, and
Shannon rate with ``2 * CH_BW_HZ * log2(1 + SNR)``.

Only difference: edge ``weight`` (nominal span counts, one span = ``span_length_km``)
are multiplied by a scale factor **as a float** before throughput is computed.
Fractions below 1 represent shorter effective fiber (e.g. 0.2 × 80 km); they are **not**
rounded up to a full span. Results are written to CSV (no MongoDB update).
"""
from __future__ import annotations

import csv
import pathlib
import sys
import time
from copy import deepcopy

# ---------------------------------------------------------------------------
# Path setup (must come before ong / gpu_init imports)
# ---------------------------------------------------------------------------
_SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
_EXTERNAL_DIR = _SCRIPT_DIR.parents[2] / "external"
_ONG_SRC = _EXTERNAL_DIR / "ong-python-toolbox" / "src"

for p in (_EXTERNAL_DIR, str(_ONG_SRC)):
    p = str(p)
    if p not in sys.path:
        sys.path.insert(0, p)

from ong.utils import gpu_init

import jax
print(f"Success! GPU count: {jax.device_count()}")
print(f"Devices: {jax.devices()}")

import networkx as nx
import numpy as np
from jax import numpy as jnp
from scipy.constants import c

import NetworkToolkit as nt

# Same-module throughput pipeline as compute_throughput_cfm.py
import compute_throughput_cfm_new as cfm

# Legacy notebook/script band label -> keys in cfm.BAND_CONFIGS
_BAND_ALIASES = {
    "SCLO": "OESCL",
}


def _resolve_band_selection(band: str) -> str:
    b = str(band).strip().upper()
    return _BAND_ALIASES.get(b, b)


def _build_band_aware_rwa_key(route_function: str, band_selection: str) -> str:
    return f"{route_function} {str(band_selection).strip().upper()} RWA"


_MIN_EFFECTIVE_SPANS = 1e-15  # avoid exact zero on an edge (numerical / degenerate link)


def _scale_graph_weights(graph, scale: float):
    """Scale edge ``weight`` by ``scale`` as a **float** (effective span count).

    ``weight`` is interpreted like ``compute_throughput_cfm._build_rwa_occupancy``:
    nominal number of reference spans of length ``span_length_km``.  Multiplying by
    ``scale`` gives a fractional effective length for NSR (ASE ∝ spans, GN-NLI ∝ spans
    in the current lumped model) without forcing ``≥ 1`` integer spans.
    """
    scaled = deepcopy(graph)
    for u, v in scaled.edges():
        w = float(scaled[u][v].get("weight", 1))
        new_w = float(scale) * w
        if new_w <= 0.0:
            new_w = _MIN_EFFECTIVE_SPANS
        scaled[u][v]["weight"] = new_w
    return scaled


def compute_scaled_graph_throughput(
    graph,
    rwa,
    band_selection: str,
    span_length_km: float = 80.0,
    launch_power_dBm: float | dict = -2.0,
    save_snr_to_disk: bool = False,
    *,
    doc_id=None,
    route_function: str | None = None,
    scale: float | None = None,
):
    """Throughput [bps] using the same method as ``compute_throughput_cfm.compute_topology_throughput``.

    Parameters
    ----------
    graph : nx.Graph
        Topology with integer node labels (relabeled inside if needed).
    rwa : dict
        Channel index (str) -> list of node paths.
    band_selection : str
        Key in ``cfm.BAND_CONFIGS`` (e.g. C, SCL, OESCL). ``SCLO`` maps to ``OESCL``.
    save_snr_to_disk : bool
        If True, write per-link ``.npz`` and ``*_occupancy.npz`` under ``data/snr/``
        (same layout as ``compute_throughput_cfm.py``); requires ``doc_id``,
        ``route_function``, and ``scale`` for filenames.
    doc_id, route_function, scale
        Used only when ``save_snr_to_disk`` is True (``scale`` distinguishes scaling runs).
    """
    graph = nx.relabel.convert_node_labels_to_integers(graph, first_label=1)

    band_key = _resolve_band_selection(band_selection)
    if band_key not in cfm.BAND_CONFIGS:
        raise ValueError(
            f"Invalid band_selection '{band_selection}' -> '{band_key}'. "
            f"Options: {list(cfm.BAND_CONFIGS.keys())} (alias: SCLO -> OESCL)"
        )

    band_cfg = cfm.BAND_CONFIGS[band_key]
    active_bands = band_cfg["bands"]

    ch_lambda, channel_idx, ch_idx_oband_full, band_masks, ref_lambda = (
        cfm._build_multiband_channel_grid(active_bands)
    )
    num_active = int(np.sum(channel_idx))
    ch_idx_oband_active = ch_idx_oband_full[channel_idx]

    P_channel, nf, snr_trx = cfm._build_per_channel_params(
        ch_lambda, band_masks, active_bands, launch_power_dBm=launch_power_dBm,
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

    (edges, edge_to_row, span_count_per_edge,
     occupancy_matrix, rwa_edge_paths, wavelength_to_col) = cfm._build_rwa_occupancy(
        graph, rwa, num_active
    )
    active_slot_ix = np.flatnonzero(np.asarray(channel_idx))

    nsr_link_channel = jnp.zeros_like(
        jnp.asarray(occupancy_matrix), dtype=jnp.float64
    )
    fit_params_per_link: dict[int, np.ndarray] = {}
    eta_spm_per_link: dict[int, np.ndarray] = {}
    eta_xpm_per_link: dict[int, np.ndarray] = {}
    eta_fwm_per_link: dict[int, np.ndarray] = {}
    nsr_ase_per_link: dict[int, np.ndarray] = {}
    n_links = int(span_count_per_edge.shape[0])

    for i in range(n_links):
        link_nsr, link_fit_params, eta_spm, eta_xpm, eta_fwm, link_nsr_ase = cfm.calc_NSR_link(
            setup,
            float(span_count_per_edge[i]),
            occupancy_matrix[i, :, None],
            ch_idx_oband_active,
        )
        link_nsr = link_nsr.squeeze(-1)
        if save_snr_to_disk:
            fit_params_per_link[i] = np.asarray(link_fit_params.squeeze(-1))
            eta_spm_per_link[i] = np.asarray(eta_spm, dtype=np.float64).reshape(-1)
            eta_xpm_per_link[i] = np.asarray(eta_xpm, dtype=np.float64).reshape(-1)
            eta_fwm_per_link[i] = np.asarray(eta_fwm, dtype=np.float64).reshape(-1)
            nsr_ase_per_link[i] = np.asarray(link_nsr_ase.squeeze(-1), dtype=np.float64)
        nsr_link_channel = nsr_link_channel.at[i].set(link_nsr)

    if save_snr_to_disk:
        if doc_id is None or route_function is None or scale is None:
            raise ValueError(
                "save_snr_to_disk=True requires doc_id, route_function, and scale "
                "for output filenames."
            )
        snr_dir = _SCRIPT_DIR / "data" / "snr"
        snr_dir.mkdir(parents=True, exist_ok=True)
        ch_lambda_active = np.asarray(ch_lambda)[np.asarray(channel_idx)]
        ch_freq_active = c / ch_lambda_active
        nsr_np = np.asarray(nsr_link_channel)
        scale_tag = f"{float(scale):g}".replace(".", "p")
        _id_str = str(doc_id)
        rf = str(route_function).strip()

        for i in range(n_links):
            occ = occupancy_matrix[i] > 0
            ln = nsr_np[i]
            snr_dB = np.full_like(ln, np.nan)
            valid = occ & np.isfinite(ln) & (ln > 0)
            snr_dB[valid] = 10.0 * np.log10(1.0 / ln[valid])

            ln_ase = nsr_ase_per_link[i]
            snr_ase_dB = np.full_like(ln_ase, np.nan)
            valid_ase = occ & np.isfinite(ln_ase) & (ln_ase > 0)
            snr_ase_dB[valid_ase] = 10.0 * np.log10(1.0 / ln_ase[valid_ase])

            fp = fit_params_per_link[i]
            u, v = edges[i]
            fname = (
                f"{_id_str}_{band_key}_{rf}_scale{scale_tag}_link{i}_{u}-{v}.npz"
            )
            np.savez_compressed(
                snr_dir / fname,
                channel_index=np.arange(num_active),
                wavelength_m=ch_lambda_active,
                frequency_hz=ch_freq_active,
                occupancy=occ.astype(int),
                snr_dB=snr_dB,
                nsr_linear=ln,
                nsr_ase_linear=ln_ase,
                snr_ase_dB=snr_ase_dB,
                eta_spm_linear=eta_spm_per_link[i],
                eta_xpm_linear=eta_xpm_per_link[i],
                eta_fwm_linear=eta_fwm_per_link[i],
                edge=np.array([u, v]),
                spans=float(span_count_per_edge[i]),
                a=fp[:, 0],
                a_bar=fp[:, 1],
                Cr=fp[:, 2],
            )

        occ_fname = f"{_id_str}_{band_key}_{rf}_scale{scale_tag}_occupancy.npz"
        np.savez_compressed(
            snr_dir / occ_fname,
            occupancy_matrix=occupancy_matrix,
            edges=np.array(edges),
            spans=np.asarray(span_count_per_edge),
            channel_index=np.arange(num_active),
            wavelength_m=ch_lambda_active,
            frequency_hz=ch_freq_active,
        )
        print(f"  Saved per-link SNR/fit-params + occupancy to {snr_dir}")

    capacity_total = 0.0
    n_lightpaths = 0

    for w_str, path_infos in rwa_edge_paths.items():
        w = int(w_str)
        if w not in wavelength_to_col:
            continue
        col = wavelength_to_col[w]
        slot_row = int(active_slot_ix[w])
        ch_bw_hz = float(np.asarray(setup.ch_bandwidth_ij)[slot_row, 0])
        for info in path_infos:
            edge_path = info["edge_path"]
            path_nsr = 0.0
            for edge in edge_path:
                row = edge_to_row[edge]
                path_nsr += float(nsr_link_channel[row, col])

            if not np.isfinite(path_nsr) or path_nsr <= 0:
                continue
            snr = 1.0 / path_nsr
            capacity_total += 2 * ch_bw_hz * np.log2(1.0 + snr)
            n_lightpaths += 1

    return float(capacity_total), int(n_lightpaths)


def parse_scales(scales_csv: str):
    return [float(x.strip()) for x in scales_csv.split(",") if x.strip()]


def _normalize_topology_names(topology: str | list[str] | tuple[str, ...]) -> list[str]:
    """Accept a single topology name or a list/tuple; return non-empty stripped strings."""
    if isinstance(topology, (list, tuple)):
        names = [str(t).strip() for t in topology if str(t).strip()]
    else:
        names = [str(topology).strip()] if str(topology).strip() else []
    return names


if __name__ == "__main__":
    # ================================================================
    # USER CONFIGURATION
    # ================================================================
    DB = "Topology_Data"
    # COLLECTION = "topology-paper"
    COLLECTION = "real"
    #CORONET_CONUS_Topology_nodes JPN25 BT22 DTAG germany50 nobel-eu RegularDCI NSFNET cost266

    # One topology (str) or several (list/tuple), e.g. ["RegularDCI", "DTAG", "germany50"]
    TOPOLOGY = "NSFNET"
    # TOPOLOGY = ["BT22","RegularDCI", "DTAG", "NSFNET"]
    ROUTE_FUNCTION = "FF-kSP" #
    # If True: write per-link SNR .npz, *_occupancy.npz under data/snr/ and print confirmation.
    SAVE_SNR_TO_DISK = False
    # One band or several comma-separated, e.g. "C" or "C,CL" or "C,CL,OESCL"
    BAND = "C,CL,SCL,ESCL,OESCL"  # "C" | CL | SCL | ESCL | OESCL | O|E|S|L
    SCALES = "0.2,0.4,0.6,0.8,1.0"
    # SCALES = "1.0"
    # SCALES = "0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0"

    SPAN_LENGTH_KM = 80.0
    # Per-band launch power [dBm] in order O, E, S, C, L.
    # Use a single float (e.g. -2.0) for uniform launch across all bands,
    # or a list/dict to set per-band values.
    LAUNCH_POWER_DBM = [-1.63, -2.63, -2.25, -4.76, -4.54]

    band_list = [b.strip() for b in str(BAND).split(",") if b.strip()]
    if not band_list:
        raise ValueError("BAND must contain at least one band, e.g. 'C' or 'C,CL,OESCL'")
    for _b in band_list:
        _bk = _resolve_band_selection(_b)
        if _bk not in cfm.BAND_CONFIGS:
            raise ValueError(
                f"Invalid BAND entry: {_b} (resolved {_bk}). "
                f"Options: {list(cfm.BAND_CONFIGS.keys())}; SCLO -> OESCL"
            )

    # Normalise list/tuple -> dict keyed by band order O,E,S,C,L
    lp_dBm = LAUNCH_POWER_DBM
    if isinstance(lp_dBm, (list, tuple)):
        _ORDER = ["O", "E", "S", "C", "L"]
        if len(lp_dBm) != len(_ORDER):
            raise ValueError(f"LAUNCH_POWER_DBM list must have {len(_ORDER)} entries (O,E,S,C,L), got {len(lp_dBm)}")
        lp_dBm = {b: float(p) for b, p in zip(_ORDER, lp_dBm)}

    scales = parse_scales(SCALES)

    topology_names = _normalize_topology_names(TOPOLOGY)
    if not topology_names:
        raise ValueError("TOPOLOGY must be a non-empty string or a non-empty list/tuple of names")

    print("=" * 60)
    print("CFM Weight Scaling Throughput (aligned with compute_throughput_cfm.py)")
    print("=" * 60)
    print(f"Source DB/Collection : {DB}/{COLLECTION}")
    print(f"Topologies           : {topology_names}")
    print(f"Bands                : {band_list}")
    print(f"Channel BW           : {cfm.CH_BW_HZ/1e9:.0f} GHz")
    print(f"Launch power         : {lp_dBm} dBm")
    print(f"Scales               : {scales}")
    print(f"Save SNR to disk     : {SAVE_SNR_TO_DISK}")
    print("=" * 60)

    fieldnames = [
        "source_id",
        "topology",
        "band_user",
        "band_resolved",
        "route_function",
        "rwa_key",
        "scale",
        "cfm_capacity_bps",
        "cfm_capacity_tbps",
        "cfm_lightpaths",
        "elapsed_seconds",
        "span_length_km",
        "launch_power_dBm",
    ]

    for topo in topology_names:
        graph_list = nt.Database.read_topology_dataset_list(
            DB, COLLECTION, find_dic={"name": topo}, node_data=True
        )
        if not graph_list:
            print(f"[WARN] No topology {topo!r} in {DB}.{COLLECTION}; skip.")
            continue

        for band in band_list:
            band_key = _resolve_band_selection(band)
            band_cfg = cfm.BAND_CONFIGS[band_key]
            rwa_key = _build_band_aware_rwa_key(ROUTE_FUNCTION, band)
            output_csv = str(_SCRIPT_DIR / "data" / f"cfm_scaling_{topo}_{band}.csv")
            print()
            print("-" * 60)
            print(f"Topology: {topo!r}  Band: {band} -> {band_key} ({band_cfg['name']})")
            print(f"RWA Key: {rwa_key}  ->  {output_csv}")
            print("-" * 60)

            rows = []
            for graph, _id in graph_list:
                docs = list(nt.Database.read_data(DB, COLLECTION, find_dic={"_id": _id}, max_count=1))
                if not docs:
                    print(f"[WARN] skip {_id}: source doc not found")
                    continue
                doc = docs[0]
                if rwa_key not in doc:
                    raise KeyError(f"Missing RWA key '{rwa_key}' for topology={topo!r}, _id={_id}")
                rwa_raw = doc[rwa_key]
                rwa = nt.Tools.read_database_dict(rwa_raw)

                for scale in scales:
                    t0 = time.perf_counter()
                    graph_scaled = _scale_graph_weights(graph, scale)
                    cap, n_lps = compute_scaled_graph_throughput(
                        graph=graph_scaled,
                        rwa=rwa,
                        band_selection=band,
                        span_length_km=SPAN_LENGTH_KM,
                        launch_power_dBm=lp_dBm,
                        save_snr_to_disk=SAVE_SNR_TO_DISK,
                        doc_id=_id,
                        route_function=ROUTE_FUNCTION,
                        scale=scale,
                    )
                    dt = time.perf_counter() - t0
                    print(
                        f"  _id={_id} scale={scale:.2f} -> "
                        f"{cap/1e12:.4f} Tbps, lightpaths={n_lps}, {dt:.1f}s"
                    )
                    rows.append(
                        {
                            "source_id": str(_id),
                            "topology": topo,
                            "band_user": band,
                            "band_resolved": band_key,
                            "route_function": ROUTE_FUNCTION,
                            "rwa_key": rwa_key,
                            "scale": float(scale),
                            "cfm_capacity_bps": float(cap),
                            "cfm_capacity_tbps": float(cap / 1e12),
                            "cfm_lightpaths": int(n_lps),
                            "elapsed_seconds": float(dt),
                            "span_length_km": float(SPAN_LENGTH_KM),
                            "launch_power_dBm": (
                                {b: float(p) for b, p in lp_dBm.items()}
                                if isinstance(lp_dBm, dict) else float(lp_dBm)
                            ),
                        }
                    )

            if rows:
                with open(output_csv, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(rows)
                print(f"Saved {len(rows)} rows to CSV: {output_csv}")
            else:
                print(f"No rows for topology {topo!r} band {band!r}; CSV not written.")

    print("\nDone.")
