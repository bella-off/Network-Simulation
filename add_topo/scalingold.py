"""
Compute CFM throughput on distance-scaled topologies.

This script mirrors the workflow in `compute_throughput_cfm.py`, but applies
edge-weight scaling to each topology before throughput computation.
"""
from __future__ import annotations

import os
import ctypes

# 1.
cuda_lib = "/apps/cuda/cuda-13.0/lib64/libcudart.so.13"
cupti_lib = "/apps/cuda/cuda-13.0/extras/CUPTI/lib64/libcupti.so.13"
cudnn_lib = "/apps/cuda/cudnn-linux-x86_64-9.14.0.64_cuda13/lib/libcudnn.so.9"

# 2.
try:
    ctypes.CDLL(cuda_lib)
    ctypes.CDLL(cupti_lib)
    ctypes.CDLL(cudnn_lib)
except Exception as e:
    print(f"wrong check: {e}")

os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/apps/cuda/cuda-13.0"

import jax
print(f"Success! GPU count: {jax.device_count()}")
print(f"Devices: {jax.devices()}")

import csv
import pathlib
import sys
import time
from copy import deepcopy

import networkx as nx
import numpy as np
from scipy import interpolate
from scipy.constants import c, h as h_planck, pi

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

from jax import numpy as jnp

import NetworkToolkit as nt
from ong.measurements import load_corning_smf_28_raman
import data as fibre_data

from cfm_nli import compute_edge_nli, calc_beta3, calc_beta4


BAND_CONFIGS = {
    "C": {
        "name": "C band",
        "wavelength_start_nm": 1530,
        "wavelength_width_nm": 40.049,
        "B_o_THz": 5.0,
        "RefLambda_nm": 1550,
        "Cr": 0,
        "channel_bandwidth_GHz": 50,
    },
    "CL": {
        "name": "CL band",
        "wavelength_start_nm": 1530,
        "wavelength_width_nm": 95,
        "B_o_THz": 11.8,
        "RefLambda_nm": 1577.5,
        "Cr": 0.028 / 1e3 / 1e12,
        "channel_bandwidth_GHz": 50,
    },
    "SCL": {
        "name": "SCL band",
        "wavelength_start_nm": 1460,
        "wavelength_width_nm": 165,
        "B_o_THz": 20.86,
        "RefLambda_nm": 1542.5,
        "Cr": 0.028 / 1e3 / 1e12,
        "channel_bandwidth_GHz": 50,
    },
    "SCLO": {
        "name": "SCLO band",
        "wavelength_start_nm": 1260,
        "wavelength_width_nm": 365,
        "B_o_THz": 46.0,
        "RefLambda_nm": 1442.5,
        "Cr": 0.028 / 1e3 / 1e12,
        "channel_bandwidth_GHz": 50,
    },
}

O_BAND_MIN = 1260e-9
O_BAND_MAX = 1360e-9


def _build_band_aware_rwa_key(route_function: str, band_selection: str) -> str:
    return f"{route_function} {str(band_selection).strip().upper()} RWA"


def _build_channel_grid(band_config):
    ch_bw = band_config["channel_bandwidth_GHz"] * 1e9
    B_o = band_config["B_o_THz"] * 1e12
    num_ch = int(np.floor(B_o / ch_bw))

    wl_start = band_config["wavelength_start_nm"] * 1e-9
    wl_width = band_config["wavelength_width_nm"] * 1e-9
    ch_spacing = wl_width / num_ch

    wavelengths = np.array([wl_start + i * ch_spacing for i in range(num_ch)])
    frequencies = c / wavelengths

    ref_lambda = band_config["RefLambda_nm"] * 1e-9
    ref_freq = c / ref_lambda
    ch_centre_hz = frequencies - ref_freq
    o_mask = (wavelengths >= O_BAND_MIN) & (wavelengths <= O_BAND_MAX)
    return wavelengths, ch_centre_hz, ch_bw, num_ch, ref_lambda, o_mask


def _interp_profile(profile, wavelengths):
    wl_prof, val_prof = profile
    f = interpolate.interp1d(
        wl_prof, val_prof, kind="linear", fill_value="extrapolate"
    )
    return f(wavelengths)


def _scale_graph_weights(graph, scale: float):
    scaled = deepcopy(graph)
    for u, v in scaled.edges():
        w = scaled[u][v].get("weight", 1)
        new_w = int(scale * w)
        scaled[u][v]["weight"] = max(1, new_w)
    return scaled


def compute_scaled_graph_throughput(
    graph,
    rwa,
    band_config,
    span_length_km=80,
    launch_power_dBm=0.0,
    noise_figure_dB=5.0,
    samples_per_km=2,
    fwm_mode="chunk",
    fwm_chunk_size=8,
):
    graph = nx.relabel.convert_node_labels_to_integers(graph, first_label=1)

    wavelengths, ch_centre_hz, ch_bw, num_ch, ref_lambda, o_mask = _build_channel_grid(
        band_config
    )

    att_profile = fibre_data.fit_attenuation("min")
    disp_profile = fibre_data.get_dispersion("min")
    raman_profile = load_corning_smf_28_raman()

    attenuation_per_ch = _interp_profile(att_profile, wavelengths)
    gamma_per_ch = np.interp(wavelengths, [1310e-9, 1550e-9], [2e-3, 1.2e-3])
    Aeff_per_ch = np.interp(wavelengths, [1310e-9, 1550e-9], [66.476e-12, 86.59e-12])

    span_length_m = span_length_km * 1e3
    NF_lin = 10 ** (noise_figure_dB / 10)
    ch_power_W = 10 ** ((launch_power_dBm - 30) / 10) * np.ones(num_ch)
    ch_bw_arr = ch_bw * np.ones(num_ch)

    disp_at_ref = np.interp(ref_lambda, disp_profile[0], disp_profile[1])
    disp_wl, disp_val = disp_profile
    disp_slope = np.gradient(disp_val, disp_wl)
    S_at_ref = np.interp(ref_lambda, disp_wl, disp_slope)
    disp_curv = np.gradient(disp_slope, disp_wl)
    Shat_at_ref = np.interp(ref_lambda, disp_wl, disp_curv)

    beta2 = -disp_at_ref * ref_lambda ** 2 / (2 * pi * c)
    beta3 = calc_beta3(ref_lambda, S_at_ref, beta2)
    beta4 = calc_beta4(ref_lambda, Shat_at_ref, beta2, beta3)
    raman_gain_slope = band_config["Cr"]

    edge_spans = {}
    for s, d in graph.edges():
        n_spans = int(graph[s][d].get("weight", 1))
        edge_spans[(s, d)] = max(1, n_spans)
        edge_spans[(d, s)] = max(1, n_spans)

    unique_span_counts = set(edge_spans.values())
    nsr_cache = {}

    for n_sp in unique_span_counts:
        print(f"    Computing NLI for num_spans={n_sp} ...")
        eta_spm, eta_xpm, eta_fwm = compute_edge_nli(
            ch_centre_hz=jnp.array(ch_centre_hz),
            ch_bandwidth_hz=jnp.array(ch_bw_arr),
            ch_power_W=jnp.array(ch_power_W),
            attenuation_dBm=jnp.array(attenuation_per_ch),
            gamma=jnp.array(gamma_per_ch),
            Aeff=jnp.array(Aeff_per_ch),
            raman_profile=raman_profile,
            ref_lambda=ref_lambda,
            span_length_m=span_length_m,
            num_spans=n_sp,
            beta2=beta2,
            beta3=beta3,
            beta4=beta4,
            raman_gain_slope=raman_gain_slope,
            o_band_mask=jnp.array(o_mask),
            samples_per_km=samples_per_km,
            fwm_mode=fwm_mode,
            fwm_chunk_size=fwm_chunk_size,
        )

        P_nli = (np.array(eta_spm) + np.array(eta_xpm) + np.array(eta_fwm)) * ch_power_W ** 3
        gain_per_span_dB = np.array(attenuation_per_ch) * span_length_m
        G_per_span = 10 ** (gain_per_span_dB / 10)
        freq_abs = c / wavelengths
        P_ase = n_sp * NF_lin * h_planck * freq_abs * ch_bw * (G_per_span - 1)
        nsr_cache[n_sp] = (P_ase + P_nli) / ch_power_W

    capacity_total = 0.0
    n_lightpaths = 0
    for ch_key, paths in rwa.items():
        ch_idx = int(ch_key)
        if ch_idx >= num_ch:
            continue
        for path in paths:
            path_nodes = [int(n) for n in path]
            nsr_total = 0.0
            for i in range(len(path_nodes) - 1):
                s, d = path_nodes[i], path_nodes[i + 1]
                nsr_total += nsr_cache[edge_spans[(s, d)]][ch_idx]
            if nsr_total <= 0:
                continue
            snr = 1.0 / nsr_total
            capacity_total += 2 * ch_bw * np.log2(1 + snr)
            n_lightpaths += 1

    return float(capacity_total), int(n_lightpaths)


def parse_scales(scales_csv: str):
    return [float(x.strip()) for x in scales_csv.split(",") if x.strip()]


if __name__ == "__main__":
    # ================================================================
    # USER CONFIGURATION
    # ================================================================
    DB = "Topology_Data"
    COLLECTION = "topology-paper"
    TOPOLOGY = "NSFNET"
    ROUTE_FUNCTION = "kSP-FF"
    BAND = "SCLO"  # C | CL | SCL | SCLO
    SCALES = "0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0"
    OUTPUT_CSV = str(_SCRIPT_DIR / f"cfm_scaling_{TOPOLOGY}_{BAND}.csv")
    SPAN_LENGTH_KM = 80.0
    LAUNCH_POWER_DBM = 0.0
    NOISE_FIGURE_DB = 5.0
    SAMPLES_PER_KM = 2
    FWM_MODE = "chunk"  # off | full | chunk
    FWM_CHUNK_SIZE = 8

    if BAND not in BAND_CONFIGS:
        raise ValueError(f"Invalid BAND: {BAND}. Options: {list(BAND_CONFIGS.keys())}")

    band_config = BAND_CONFIGS[BAND]
    rwa_key = _build_band_aware_rwa_key(ROUTE_FUNCTION, BAND)
    scales = parse_scales(SCALES)
    output_csv = OUTPUT_CSV

    print("=" * 60)
    print("CFM Weight Scaling Throughput")
    print("=" * 60)
    print(f"Source DB/Collection : {DB}/{COLLECTION}")
    print(f"Output CSV           : {output_csv}")
    print(f"Topology             : {TOPOLOGY}")
    print(f"RWA Key              : {rwa_key}")
    print(f"Band                 : {BAND}")
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
            raise KeyError(f"Missing RWA key '{rwa_key}' for _id={_id}")
        rwa_raw = doc[rwa_key]
        rwa = nt.Tools.read_database_dict(rwa_raw)

        for scale in scales:
            t0 = time.perf_counter()
            graph_scaled = _scale_graph_weights(graph, scale)
            cap, n_lps = compute_scaled_graph_throughput(
                graph=graph_scaled,
                rwa=rwa,
                band_config=band_config,
                span_length_km=SPAN_LENGTH_KM,
                launch_power_dBm=LAUNCH_POWER_DBM,
                noise_figure_dB=NOISE_FIGURE_DB,
                samples_per_km=SAMPLES_PER_KM,
                fwm_mode=FWM_MODE,
                fwm_chunk_size=FWM_CHUNK_SIZE,
            )
            dt = time.perf_counter() - t0
            print(
                f"  _id={_id} scale={scale:.2f} -> "
                f"{cap/1e12:.4f} Tbps, lightpaths={n_lps}, {dt:.1f}s"
            )
            rows.append(
                {
                    "source_id": str(_id),
                    "topology": TOPOLOGY,
                    "band": BAND,
                    "route_function": ROUTE_FUNCTION,
                    "rwa_key": rwa_key,
                    "scale": float(scale),
                    "cfm_capacity_bps": float(cap),
                    "cfm_capacity_tbps": float(cap / 1e12),
                    "cfm_lightpaths": int(n_lps),
                    "elapsed_seconds": float(dt),
                    "span_length_km": float(SPAN_LENGTH_KM),
                    "launch_power_dBm": float(LAUNCH_POWER_DBM),
                    "noise_figure_dB": float(NOISE_FIGURE_DB),
                    "samples_per_km": int(SAMPLES_PER_KM),
                    "fwm_mode": FWM_MODE,
                    "fwm_chunk_size": int(FWM_CHUNK_SIZE),
                }
            )

    if rows:
        fieldnames = [
            "source_id",
            "topology",
            "band",
            "route_function",
            "rwa_key",
            "scale",
            "cfm_capacity_bps",
            "cfm_capacity_tbps",
            "cfm_lightpaths",
            "elapsed_seconds",
            "span_length_km",
            "launch_power_dBm",
            "noise_figure_dB",
            "samples_per_km",
            "fwm_mode",
            "fwm_chunk_size",
        ]
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nSaved {len(rows)} rows to CSV: {output_csv}")
    else:
        print("\nNo rows produced; CSV not written.")

    print("\nDone.")
