"""
Full-JAX scaling script backed by ``cfm_jax.py``.

Key idea for distance/weight scaling:

- RWA occupancy does not change across scales.
- For a fixed link occupancy mask, the ASE and NLI terms in the current CFM
  implementation scale linearly with the number of spans.
- The transceiver penalty is span-count independent.

So instead of recomputing the full link model for every scale, we:

1. Build topology + occupancy once.
2. Compute one-span NSR for each occupied link once.
3. Split that NSR into:
   - per-span noise term
   - constant transceiver penalty
4. Broadcast the scale-dependent span counts across all wavelengths with JAX.
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
import cfm_jax as cfmj
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
    base = np.asarray(base_span_counts, dtype=np.int32)
    scales_arr = np.asarray(scales, dtype=np.float64)[:, None]
    scaled = np.floor(scales_arr * base[None, :]).astype(np.int32)
    return np.maximum(1, scaled)


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


def _compute_one_span_noise(prep, occupancy_matrix):
    """Compute the span-dependent noise term once per occupied link.

    For ``nspans=1``:
        NSR = (NLI_per_span + ASE_per_span) + TRX_penalty

    So we can recover:
        per_span_noise = NSR(1) - TRX_penalty
    """
    occupancy_np = np.asarray(occupancy_matrix, dtype=np.float64)
    trx_pen = 1.0 / cfm.idB(np.asarray(prep.snr_trx_active, dtype=np.float64))

    per_span_rows = []
    t0 = time.perf_counter()
    for occ_row in occupancy_np:
        one_span_nsr = cfmj.calc_nsr_link_purejax(
            prep,
            1,
            occ_row[:, None],
        ).squeeze(-1)
        one_span_nsr = np.asarray(one_span_nsr, dtype=np.float64)

        active = occ_row > 0
        row = np.zeros_like(one_span_nsr, dtype=np.float64)
        row[active] = np.maximum(one_span_nsr[active] - trx_pen[active], 0.0)
        per_span_rows.append(row)

    elapsed = time.perf_counter() - t0
    return np.stack(per_span_rows, axis=0), trx_pen, elapsed


def _make_batched_nsr_fn(occupancy_matrix, per_span_noise, trx_pen):
    occ_jnp = jnp.asarray(occupancy_matrix > 0, dtype=bool)
    per_span_noise_jnp = jnp.asarray(per_span_noise, dtype=jnp.float64)
    trx_pen_jnp = jnp.asarray(trx_pen, dtype=jnp.float64)

    @jax.jit
    def _batched_nsr(span_counts_scales):
        span_counts_3d = jnp.asarray(span_counts_scales, dtype=jnp.float64)[:, :, None]
        nsr_active = span_counts_3d * per_span_noise_jnp[None, :, :] + trx_pen_jnp[None, None, :]
        return jnp.where(occ_jnp[None, :, :], nsr_active, jnp.inf)

    return _batched_nsr


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


def compute_scaled_graph_throughput_fulljax(
    graph,
    rwa,
    scales,
    band_selection: str,
    span_length_km: float = 80.0,
    launch_power_dBm: float = -2.0,
):
    (
        setup,
        prep,
        _edges,
        edge_to_row,
        base_span_counts,
        occupancy_matrix,
        rwa_edge_paths,
        _wavelength_to_col,
        active_slot_ix,
    ) = cfmj._compute_topology_arrays(
        graph,
        rwa,
        band_selection,
        span_length_km=span_length_km,
        launch_power_dBm=launch_power_dBm,
    )

    num_active = occupancy_matrix.shape[1]
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
        return zero_caps, zero_lps, 0.0, 0.0

    per_span_noise, trx_pen, one_span_seconds = _compute_one_span_noise(
        prep,
        occupancy_matrix,
    )

    span_counts_scales = _build_scaled_span_counts(base_span_counts, scales)
    batched_nsr_fn = _make_batched_nsr_fn(
        occupancy_matrix,
        per_span_noise,
        trx_pen,
    )
    batched_capacity_fn = _make_batched_capacity_fn(*lightpath_arrays)

    t0 = time.perf_counter()
    nsr_scales = batched_nsr_fn(span_counts_scales)
    caps_bps, n_lightpaths = batched_capacity_fn(nsr_scales)
    jax.block_until_ready(caps_bps)
    batch_seconds = time.perf_counter() - t0

    return (
        np.asarray(caps_bps, dtype=np.float64),
        np.asarray(n_lightpaths, dtype=np.int32),
        float(one_span_seconds),
        float(batch_seconds),
    )


if __name__ == "__main__":
    DB = "Topology_Data"
    COLLECTION = "real"

    TOPOLOGY = "germany50"
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
    print("CFM Weight Scaling Throughput (full JAX scaling)")
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

        caps_bps, n_lightpaths, one_span_seconds, batch_seconds = (
            compute_scaled_graph_throughput_fulljax(
                graph=graph,
                rwa=rwa,
                scales=scales,
                band_selection=BAND,
                span_length_km=SPAN_LENGTH_KM,
                launch_power_dBm=LAUNCH_POWER_DBM,
            )
        )

        total_seconds = one_span_seconds + batch_seconds
        print(
            f"  _id={_id}: one-span prep {one_span_seconds:.1f}s, "
            f"batched scaling {batch_seconds:.1f}s, total {total_seconds:.1f}s"
        )

        per_scale_seconds = total_seconds / max(len(scales), 1)
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
                    "one_span_prep_seconds": float(one_span_seconds),
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
            "one_span_prep_seconds",
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
