"""
JAX-batched CFM throughput scaling.

Compared with ``compute_throughput_cfm_scaling.py``, this version keeps the
same physics model but avoids rebuilding the scaled graph and re-parsing the
RWA for every scale. Instead it:

1. Builds the RWA occupancy once per topology.
2. Converts lightpaths to padded JAX index arrays once.
3. Batches the scale sweep so GPU work happens across ``scale x link`` instead
   of a pure Python nested loop.

Results are written to CSV only; MongoDB is read-only here.
"""
from __future__ import annotations

import csv
import ctypes
import os
import pathlib
import sys
import time

cuda_lib = "/apps/cuda/cuda-13.0/lib64/libcudart.so.13"
cupti_lib = "/apps/cuda/cuda-13.0/extras/CUPTI/lib64/libcupti.so.13"
cudnn_lib = "/apps/cuda/cudnn-linux-x86_64-9.14.0.64_cuda13/lib/libcudnn.so.9"

try:
    ctypes.CDLL(cuda_lib)
    ctypes.CDLL(cupti_lib)
    ctypes.CDLL(cudnn_lib)
except Exception as e:
    print(f"wrong check: {e}")

os.environ["XLA_FLAGS"] = (
    "--xla_gpu_cuda_data_dir=/apps/cuda/cuda-13.0"
    " --xla_gpu_deterministic_ops=true"
)

import jax
from jax import numpy as jnp

jax.config.update("jax_enable_x64", True)

print(f"Success! GPU count: {jax.device_count()}")
print(f"Devices: {jax.devices()}")

import networkx as nx
import numpy as np

# ---------------------------------------------------------------------------
# Path setup (must come before local imports)
# ---------------------------------------------------------------------------
_SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
_EXTERNAL_DIR = _SCRIPT_DIR.parents[2] / "external"
_ONG_SRC = _EXTERNAL_DIR / "ong-python-toolbox" / "src"

for p in (_EXTERNAL_DIR, str(_ONG_SRC)):
    p = str(p)
    if p not in sys.path:
        sys.path.insert(0, p)

import NetworkToolkit as nt
import compute_throughput_cfm as cfm


_BAND_ALIASES = {
    "SCLO": "OESCL",
}


def _resolve_band_selection(band: str) -> str:
    b = str(band).strip().upper()
    return _BAND_ALIASES.get(b, b)


def _build_band_aware_rwa_key(route_function: str, band_selection: str) -> str:
    return f"{route_function} {str(band_selection).strip().upper()} RWA"


def parse_scales(scales_csv: str):
    return [float(x.strip()) for x in scales_csv.split(",") if x.strip()]


def _build_scaled_span_counts(base_span_counts, scales):
    """Per-link effective span count = scale * base (float, no rounding up to 1)."""
    base = np.asarray(base_span_counts, dtype=np.float64)
    scales_arr = np.asarray(scales, dtype=np.float64)[:, None]
    scaled = scales_arr * base[None, :]
    return np.maximum(scaled, 1e-15)


def _prepare_lightpath_arrays(
    rwa_edge_paths,
    edge_to_row,
    num_active_channels,
    active_slot_ix,
    ch_bandwidth_ij,
):
    paths = []
    max_hops = 0

    for w_str, path_infos in rwa_edge_paths.items():
        w = int(w_str)
        if w >= num_active_channels:
            continue
        slot_row = int(active_slot_ix[w])
        ch_bw_hz = float(np.asarray(ch_bandwidth_ij)[slot_row, 0])

        for info in path_infos:
            edge_rows = [edge_to_row[edge] for edge in info["edge_path"] if edge in edge_to_row]
            if not edge_rows:
                continue
            max_hops = max(max_hops, len(edge_rows))
            paths.append((w, ch_bw_hz, edge_rows))

    if not paths:
        return None

    num_paths = len(paths)
    path_edge_rows = np.zeros((num_paths, max_hops), dtype=np.int32)
    path_edge_valid = np.zeros((num_paths, max_hops), dtype=bool)
    path_cols = np.zeros((num_paths,), dtype=np.int32)
    path_bandwidths_hz = np.zeros((num_paths,), dtype=np.float64)

    for i, (w, ch_bw_hz, edge_rows) in enumerate(paths):
        path_cols[i] = w
        path_bandwidths_hz[i] = ch_bw_hz
        path_edge_rows[i, :len(edge_rows)] = edge_rows
        path_edge_valid[i, :len(edge_rows)] = True

    return (
        path_edge_rows,
        path_edge_valid,
        path_cols,
        path_bandwidths_hz,
    )


def _compute_nsr_scales(setup, occupancy_matrix, ch_idx_oband_active, span_counts_scales):
    """Compute NSR for every (scale, link).

    This intentionally stays outside an outer ``jax.jit``.

    ``cfm.calc_NSR_link`` touches lazy ``setup`` properties from the ONG model,
    and some of those properties still call NumPy helpers such as ``polyfit``.
    Wrapping that whole stack in an outer JAX trace causes
    ``TracerArrayConversionError``. We therefore keep the per-link call in
    Python, but still batch the later lightpath aggregation in JAX.
    """
    occupancy_np = np.asarray(occupancy_matrix, dtype=np.float64)
    ch_idx_oband_active_np = np.asarray(ch_idx_oband_active, dtype=bool)

    nsr_scales = []
    for span_counts_scale in np.asarray(span_counts_scales, dtype=np.float64):
        nsr_rows = []
        for nspans, occ_row in zip(span_counts_scale, occupancy_np):
            link_nsr = cfm.calc_NSR_link(
                setup,
                float(nspans),
                occ_row[:, None],
                ch_idx_oband_active_np,
            ).squeeze(-1)
            nsr_rows.append(np.asarray(link_nsr, dtype=np.float64))
        nsr_scales.append(np.stack(nsr_rows, axis=0))

    return jnp.asarray(np.stack(nsr_scales, axis=0), dtype=jnp.float64)


def _make_batched_capacity_fn(
    path_edge_rows,
    path_edge_valid,
    path_cols,
    path_bandwidths_hz,
):
    path_edge_rows_jnp = jnp.asarray(path_edge_rows, dtype=jnp.int32)
    path_edge_valid_jnp = jnp.asarray(path_edge_valid, dtype=bool)
    path_cols_jnp = jnp.asarray(path_cols, dtype=jnp.int32)
    path_cols_2d = jnp.broadcast_to(path_cols_jnp[:, None], path_edge_rows_jnp.shape)
    path_bandwidths_jnp = jnp.asarray(path_bandwidths_hz, dtype=jnp.float64)

    def _one_scale(nsr_scale):
        gathered = nsr_scale[path_edge_rows_jnp, path_cols_2d]
        path_nsr = jnp.sum(
            jnp.where(path_edge_valid_jnp, gathered, 0.0),
            axis=1,
        )
        valid = jnp.isfinite(path_nsr) & (path_nsr > 0)
        snr = jnp.where(valid, 1.0 / path_nsr, 0.0)
        capacities = jnp.where(
            valid,
            2.0 * path_bandwidths_jnp * jnp.log2(1.0 + snr),
            0.0,
        )
        return jnp.sum(capacities), jnp.sum(valid.astype(jnp.int32))

    return jax.jit(jax.vmap(_one_scale, in_axes=0))


def compute_scaled_graph_throughput_batched(
    graph,
    rwa,
    scales,
    band_selection: str,
    span_length_km: float = 80.0,
    launch_power_dBm: float = -2.0,
):
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
    ch_idx_oband_active = np.asarray(ch_idx_oband_full[channel_idx], dtype=bool)

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

    (
        edges,
        edge_to_row,
        base_span_counts,
        occupancy_matrix,
        rwa_edge_paths,
        _wavelength_to_col,
    ) = cfm._build_rwa_occupancy(graph, rwa, num_active)
    del edges, _wavelength_to_col

    active_slot_ix = np.flatnonzero(np.asarray(channel_idx))
    lightpath_arrays = _prepare_lightpath_arrays(
        rwa_edge_paths,
        edge_to_row,
        num_active,
        active_slot_ix,
        setup.ch_bandwidth_ij,
    )

    if lightpath_arrays is None:
        zero_caps = np.zeros((len(scales),), dtype=np.float64)
        zero_lps = np.zeros((len(scales),), dtype=np.int32)
        return zero_caps, zero_lps

    span_counts_scales = _build_scaled_span_counts(base_span_counts, scales)
    batched_capacity_fn = _make_batched_capacity_fn(*lightpath_arrays)

    t0 = time.perf_counter()
    nsr_scales = _compute_nsr_scales(
        setup,
        occupancy_matrix,
        ch_idx_oband_active,
        span_counts_scales,
    )
    caps_bps, n_lightpaths = batched_capacity_fn(nsr_scales)
    jax.block_until_ready(caps_bps)
    compute_seconds = time.perf_counter() - t0

    return (
        np.asarray(caps_bps, dtype=np.float64),
        np.asarray(n_lightpaths, dtype=np.int32),
        compute_seconds,
    )


if __name__ == "__main__":
    # ================================================================
    # USER CONFIGURATION
    # ================================================================
    DB = "Topology_Data"
    COLLECTION = "real"

    TOPOLOGY = "cost266"
    ROUTE_FUNCTION = "kSP-FF"
    BAND = "C"
    SCALES = "0.2,0.4,0.6,0.8,1.0"

    OUTPUT_CSV = str(_SCRIPT_DIR / "data" / f"cfm_scaling_jax_{TOPOLOGY}_{BAND}.csv")
    SPAN_LENGTH_KM = 80.0
    LAUNCH_POWER_DBM = -2.0

    band_key = _resolve_band_selection(BAND)
    if band_key not in cfm.BAND_CONFIGS:
        raise ValueError(
            f"Invalid BAND: {BAND} (resolved {band_key}). "
            f"Options: {list(cfm.BAND_CONFIGS.keys())}; SCLO -> OESCL"
        )

    band_cfg = cfm.BAND_CONFIGS[band_key]
    rwa_key = _build_band_aware_rwa_key(ROUTE_FUNCTION, BAND)
    scales = parse_scales(SCALES)
    output_csv = OUTPUT_CSV

    print("=" * 60)
    print("CFM Weight Scaling Throughput (JAX batched)")
    print("=" * 60)
    print(f"Source DB/Collection : {DB}/{COLLECTION}")
    print(f"Output CSV           : {output_csv}")
    print(f"Topology             : {TOPOLOGY}")
    print(f"RWA Key              : {rwa_key}")
    print(f"Band                 : {BAND} -> {band_key} ({band_cfg['name']})")
    print(f"Active bands         : {band_cfg['bands']}")
    print(f"Channel BW           : {cfm.CH_BW_HZ/1e9:.0f} GHz")
    print(f"Launch power         : {LAUNCH_POWER_DBM} dBm")
    print(f"Scales               : {scales}")
    print("=" * 60)

    graph_list = nt.Database.read_topology_dataset_list(
        DB, COLLECTION, find_dic={"name": TOPOLOGY}, node_data=True
    )
    if not graph_list:
        raise RuntimeError(
            f"No topology '{TOPOLOGY}' found in {DB}.{COLLECTION}"
        )

    rows = []
    for graph, _id in graph_list:
        docs = list(nt.Database.read_data(DB, COLLECTION, find_dic={"_id": _id}, max_count=1))
        if not docs:
            print(f"[WARN] skip {_id}: source doc not found")
            continue

        doc = docs[0]
        if rwa_key not in doc:
            print(f"[WARN] skip {_id}: missing RWA key '{rwa_key}'")
            continue

        rwa = nt.Tools.read_database_dict(doc[rwa_key])

        caps_bps, n_lightpaths, batch_seconds = compute_scaled_graph_throughput_batched(
            graph=graph,
            rwa=rwa,
            scales=scales,
            band_selection=BAND,
            span_length_km=SPAN_LENGTH_KM,
            launch_power_dBm=LAUNCH_POWER_DBM,
        )

        print(
            f"  _id={_id}: batched {len(scales)} scales in {batch_seconds:.1f}s "
            "(first run includes JIT compile)"
        )

        per_scale_seconds = batch_seconds / max(len(scales), 1)
        for scale, cap, n_lps in zip(scales, caps_bps, n_lightpaths):
            print(
                f"    scale={scale:.2f} -> "
                f"{cap/1e12:.4f} Tbps, lightpaths={int(n_lps)}"
            )
            rows.append(
                {
                    "source_id": str(_id),
                    "topology": TOPOLOGY,
                    "band_user": BAND,
                    "band_resolved": band_key,
                    "route_function": ROUTE_FUNCTION,
                    "rwa_key": rwa_key,
                    "scale": float(scale),
                    "cfm_capacity_bps": float(cap),
                    "cfm_capacity_tbps": float(cap / 1e12),
                    "cfm_lightpaths": int(n_lps),
                    "elapsed_seconds": float(per_scale_seconds),
                    "batch_elapsed_seconds": float(batch_seconds),
                    "span_length_km": float(SPAN_LENGTH_KM),
                    "launch_power_dBm": float(LAUNCH_POWER_DBM),
                }
            )

    if rows:
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
            "batch_elapsed_seconds",
            "span_length_km",
            "launch_power_dBm",
        ]
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nSaved {len(rows)} rows to CSV: {output_csv}")
    else:
        print("\nNo rows produced; CSV not written.")

    print("\nDone.")
