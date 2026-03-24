"""
Compute throughput using the CFM-based NLI model.

Reads RWA results from MongoDB (produced by ilp_connections.py, k_sp_ff.py,
or ff_k_sp.py), then recomputes NLI using the closed-form integral model
from cfm_o_jlt.ipynb with full ISRS ODE power evolution.  Results are
written back to MongoDB under a ``"{route_function} Capacity-CFM"`` key.

Usage
-----
    Option 1: Edit the configuration section at the bottom of this file
              and run ``python compute_throughput_cfm.py``

    Option 2: Override via command line:
        python compute_throughput_cfm.py \\
            --route_function kSP-FF \\
            --topology NSFNET \\
            --collection topology-paper \\
            --band C
"""
from __future__ import annotations

import argparse
import datetime
import pathlib
import sys
import time

import networkx as nx
import numpy as np
from scipy.constants import c, h as h_planck, pi
from scipy import interpolate

# ---------------------------------------------------------------------------
# Path setup  (must come before local imports)
# ---------------------------------------------------------------------------
_SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
_EXTERNAL_DIR = _SCRIPT_DIR.parents[2] / "external"
_ONG_SRC = _EXTERNAL_DIR / "ong-python-toolbox" / "src"

for p in (_EXTERNAL_DIR, str(_ONG_SRC)):
    p = str(p)
    if p not in sys.path:
        sys.path.insert(0, p)

import jax
from jax import numpy as jnp

import NetworkToolkit as nt
from ong.measurements import load_corning_smf_28_raman
from ong.models import raman_solver, FibreSpanSetupAdvanced
import data as fibre_data

from cfm_nli import compute_edge_nli, calc_beta3, calc_beta4

# ---------------------------------------------------------------------------
# Band configurations (mirrors the driver scripts)
# ---------------------------------------------------------------------------
BAND_CONFIGS = {
    "C": {
        "name": "C band",
        "wavelength_start_nm": 1530,
        "wavelength_width_nm": 40.049,
        "B_o_THz": 5.0,
        "RefLambda_nm": 1550,
        "Cr": 0,
        "channel_bandwidth_GHz": 50,
        "description": "C band: 1530-1570 nm",
    },
    "CL": {
        "name": "CL band",
        "wavelength_start_nm": 1530,
        "wavelength_width_nm": 95,
        "B_o_THz": 11.8,
        "RefLambda_nm": 1577.5,
        "Cr": 0.028 / 1e3 / 1e12,
        "channel_bandwidth_GHz": 50,
        "description": "C+L band: 1530-1625 nm",
    },
    "SCL": {
        "name": "SCL band",
        "wavelength_start_nm": 1460,
        "wavelength_width_nm": 165,
        "B_o_THz": 20.86,
        "RefLambda_nm": 1542.5,
        "Cr": 0.028 / 1e3 / 1e12,
        "channel_bandwidth_GHz": 50,
        "description": "SCL band: 1460-1625 nm",
    },
    "SCLO": {
        "name": "SCLO band",
        "wavelength_start_nm": 1260,
        "wavelength_width_nm": 365,
        "B_o_THz": 46.0,
        "RefLambda_nm": 1442.5,
        "Cr": 0.028 / 1e3 / 1e12,
        "channel_bandwidth_GHz": 50,
        "description": "SCLO band: 1260-1625 nm",
    },
}

# O-band wavelength boundaries [m]
O_BAND_MIN = 1260e-9
O_BAND_MAX = 1360e-9


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _build_band_aware_rwa_key(route_function, band_selection):
    """Return an RWA key like 'kSP-FF SCL RWA'."""
    band_label = str(band_selection).strip().upper()
    return f"{route_function} {band_label} RWA"


def _build_channel_grid(band_config):
    """Return channel-centre wavelengths [m], frequencies [Hz], and mask."""
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
    ch_centre_hz = frequencies - ref_freq  # relative to ref

    o_mask = (wavelengths >= O_BAND_MIN) & (wavelengths <= O_BAND_MAX)

    return wavelengths, ch_centre_hz, ch_bw, num_ch, ref_lambda, o_mask


def _interp_profile(profile, wavelengths):
    """Interpolate a (wl, value) profile onto *wavelengths*."""
    wl_prof, val_prof = profile
    f = interpolate.interp1d(
        wl_prof, val_prof, kind="linear", fill_value="extrapolate"
    )
    return f(wavelengths)


# ---------------------------------------------------------------------------
# Per-topology throughput computation
# ---------------------------------------------------------------------------
def compute_topology_throughput(
    db,
    collection,
    _id,
    band_config,
    band_selection,
    route_function,
    span_length_km=80,
    launch_power_dBm=0.0,
    noise_figure_dB=5.0,
    samples_per_km=2,
):
    """Compute throughput for one topology document stored in MongoDB.

    Parameters
    ----------
    db, collection, _id : str
        MongoDB coordinates.
    band_config : dict
        Band configuration dictionary.
    band_selection : str
        Band label used in band-aware RWA key names.
    route_function : str
        RWA key prefix, e.g. ``"ILP-connections"``, ``"kSP-FF"``, ``"FF-kSP"``.
    span_length_km : float
        Span length in km.
    launch_power_dBm : float
        Uniform per-channel launch power [dBm].
    noise_figure_dB : float
        EDFA noise figure [dB].
    samples_per_km : int
        ISRS ODE spatial sampling density.

    Returns
    -------
    float
        Total network throughput [bps].
    """
    # --- Read topology and RWA from MongoDB ----------------------------
    results = list(
        nt.Database.read_data(db, collection, find_dic={"_id": _id}, max_count=1)
    )
    if not results:
        print(f"[WARN] No document found for _id={_id}")
        return 0.0

    doc = results[0]
    rwa_key = _build_band_aware_rwa_key(route_function, band_selection)
    if rwa_key not in doc:
        legacy_rwa_key = f"{route_function} RWA"
        if legacy_rwa_key in doc:
            print(
                f"[WARN] Band-aware key '{rwa_key}' missing, fallback to legacy '{legacy_rwa_key}' for _id={_id}"
            )
            rwa_key = legacy_rwa_key
        else:
            print(
                f"[WARN] Key '{rwa_key}' (or legacy '{legacy_rwa_key}') not found for _id={_id}"
            )
            return 0.0

    graph_list = nt.Database.read_topology_dataset_list(
        db, collection, find_dic={"_id": _id}, node_data=True
    )
    if not graph_list:
        print(f"[WARN] Could not read topology for _id={_id}")
        return 0.0

    graph = graph_list[0][0]
    graph = nx.relabel.convert_node_labels_to_integers(graph, first_label=1)

    rwa_raw = doc[rwa_key]
    rwa = nt.Tools.read_database_dict(rwa_raw)

    # --- Channel grid --------------------------------------------------
    wavelengths, ch_centre_hz, ch_bw, num_ch, ref_lambda, o_mask = (
        _build_channel_grid(band_config)
    )

    # --- Fiber profiles ------------------------------------------------
    att_profile = fibre_data.fit_attenuation("min")
    disp_profile = fibre_data.get_dispersion("min")
    raman_profile = load_corning_smf_28_raman()

    attenuation_per_ch = _interp_profile(att_profile, wavelengths)  # dB/m
    gamma_per_ch = np.interp(
        wavelengths, [1310e-9, 1550e-9], [2e-3, 1.2e-3]
    )  # 1/W/m (linear interp between known values)
    Aeff_per_ch = np.interp(
        wavelengths, [1310e-9, 1550e-9], [66.476e-12, 86.59e-12]
    )  # m^2

    span_length_m = span_length_km * 1e3
    NF_lin = 10 ** (noise_figure_dB / 10)
    ch_power_W = 10 ** ((launch_power_dBm - 30) / 10) * np.ones(num_ch)
    ch_bw_arr = ch_bw * np.ones(num_ch)

    # Dispersion coefficients at ref_lambda
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

    # --- Assign physical wavelengths to channel indices ----------------
    ch_spacing_nm = band_config["wavelength_width_nm"] / num_ch
    wl_start_nm = band_config["wavelength_start_nm"]
    channel_idx_to_wavelength = {
        i: (wl_start_nm + i * ch_spacing_nm) * 1e-9 for i in range(num_ch)
    }

    # --- Pre-compute NLI per distinct span count -----------------------
    edge_spans = {}
    for s, d in graph.edges():
        n_spans = int(graph[s][d].get("weight", 1))
        if n_spans == 0:
            n_spans = 1
        edge_spans[(s, d)] = n_spans
        edge_spans[(d, s)] = n_spans

    unique_span_counts = set(edge_spans.values())

    nsr_cache = {}
    for n_sp in unique_span_counts:
        print(f"  Computing NLI for num_spans={n_sp} ...")
        t0 = time.perf_counter()

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
        )

        # NLI power per channel
        P_nli = (np.array(eta_spm) + np.array(eta_xpm) + np.array(eta_fwm)) * ch_power_W ** 3

        # ASE noise per channel (per link = num_spans amplifiers)
        gain_per_span_dB = np.array(attenuation_per_ch) * span_length_m
        G_per_span = 10 ** (gain_per_span_dB / 10)
        freq_abs = c / wavelengths
        P_ase = n_sp * NF_lin * h_planck * freq_abs * ch_bw * (G_per_span - 1)

        # NSR per channel per link
        nsr = (P_ase + P_nli) / ch_power_W
        nsr_cache[n_sp] = nsr

        dt = time.perf_counter() - t0
        print(f"    Done in {dt:.1f}s  |  mean SNR = {np.mean(10*np.log10(1/nsr)):.1f} dB")

    # --- Walk RWA lightpaths and compute capacity ----------------------
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
                n_sp = edge_spans.get((s, d), 1)
                nsr_total += nsr_cache[n_sp][ch_idx]

            if nsr_total <= 0:
                continue
            snr = 1.0 / nsr_total
            capacity = 2 * ch_bw * np.log2(1 + snr)
            capacity_total += capacity
            n_lightpaths += 1

    print(
        f"  Topology {_id}: {n_lightpaths} lightpaths, "
        f"capacity = {capacity_total/1e12:.4f} Tbps"
    )

    # --- Write results back to MongoDB ---------------------------------
    nt.Database.update_data_with_id(
        db,
        collection,
        _id,
        newvals={
            "$set": {
                f"{route_function} Capacity-CFM": float(capacity_total),
                f"{route_function} CFM-lightpaths": int(n_lightpaths),
                f"{route_function} CFM-timestamp": datetime.datetime.utcnow(),
                f"{route_function} CFM-band": band_config["name"],
                f"{route_function} CFM-span_length_km": float(span_length_km),
                f"{route_function} CFM-launch_power_dBm": float(launch_power_dBm),
                f"{route_function} CFM-noise_figure_dB": float(noise_figure_dB),
            }
        },
    )

    return capacity_total


# ---------------------------------------------------------------------------
# Ray-parallel wrapper
# ---------------------------------------------------------------------------
def run_parallel(
    db,
    collection,
    topology_name,
    route_function,
    band_selection,
    span_length_km=80,
    launch_power_dBm=0.0,
    noise_figure_dB=5.0,
    samples_per_km=2,
    hostname="128.40.42.13",
    port=6379,
):
    """Launch Ray tasks, one per topology document."""
    import ray

    if hostname is not None:
        ray.init(address=f"{hostname}:{port}")
    else:
        ray.init()

    band_config = BAND_CONFIGS[band_selection]

    graph_list = nt.Database.read_topology_dataset_list(
        db, collection, find_dic={"name": topology_name}, node_data=True
    )
    print(f"Found {len(graph_list)} topologies for '{topology_name}'")

    @ray.remote(num_cpus=1)
    def _remote_worker(_id):
        return compute_topology_throughput(
            db=db,
            collection=collection,
            _id=_id,
            band_config=band_config,
            band_selection=band_selection,
            route_function=route_function,
            span_length_km=span_length_km,
            launch_power_dBm=launch_power_dBm,
            noise_figure_dB=noise_figure_dB,
            samples_per_km=samples_per_km,
        )

    futures = [_remote_worker.remote(_id) for _, _id in graph_list]
    results = ray.get(futures)

    ray.shutdown()
    print(f"\nAll done. {len(results)} topologies processed.")
    for i, (_, _id) in enumerate(graph_list):
        print(f"  {_id}: {results[i]/1e12:.4f} Tbps")

    return results


# ---------------------------------------------------------------------------
# Sequential (non-Ray) execution for testing
# ---------------------------------------------------------------------------
def run_sequential(
    db,
    collection,
    topology_name,
    route_function,
    band_selection,
    span_length_km=80,
    launch_power_dBm=0.0,
    noise_figure_dB=5.0,
    samples_per_km=2,
):
    """Process topologies one by one (no Ray needed)."""
    band_config = BAND_CONFIGS[band_selection]

    graph_list = nt.Database.read_topology_dataset_list(
        db, collection, find_dic={"name": topology_name}, node_data=True
    )
    print(f"Found {len(graph_list)} topologies for '{topology_name}'")

    results = []
    for graph, _id in graph_list:
        cap = compute_topology_throughput(
            db=db,
            collection=collection,
            _id=_id,
            band_config=band_config,
            band_selection=band_selection,
            route_function=route_function,
            span_length_km=span_length_km,
            launch_power_dBm=launch_power_dBm,
            noise_figure_dB=noise_figure_dB,
            samples_per_km=samples_per_km,
        )
        results.append(cap)

    print(f"\nAll done. {len(results)} topologies processed.")
    for i, (_, _id) in enumerate(graph_list):
        print(f"  {_id}: {results[i]/1e12:.4f} Tbps")

    return results


# ---------------------------------------------------------------------------
# CLI (command-line overrides; defaults come from the config section below)
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Compute throughput using CFM NLI model"
    )
    parser.add_argument(
        "--route_function",
        type=str,
        default=None,
        help="RWA key prefix (ILP-connections | kSP-FF | FF-kSP)",
    )
    parser.add_argument(
        "--topology", type=str, default=None, help="Topology name in MongoDB"
    )
    parser.add_argument(
        "--collection", type=str, default=None, help="MongoDB collection"
    )
    parser.add_argument(
        "--db", type=str, default=None, help="MongoDB database name"
    )
    parser.add_argument(
        "--band",
        type=str,
        default=None,
        choices=list(BAND_CONFIGS.keys()),
        help="Band selection",
    )
    parser.add_argument(
        "--span_length_km", type=float, default=None, help="Span length in km"
    )
    parser.add_argument(
        "--launch_power_dBm", type=float, default=None, help="Launch power per channel"
    )
    parser.add_argument(
        "--noise_figure_dB", type=float, default=None, help="EDFA noise figure"
    )
    parser.add_argument(
        "--samples_per_km", type=int, default=None, help="ISRS ODE spatial sampling"
    )
    parser.add_argument(
        "--parallel", action="store_true", help="Use Ray for parallel execution"
    )
    parser.add_argument("--hostname", type=str, default=None)
    parser.add_argument("--port", type=int, default=None)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # ================================================================
    #  USER CONFIGURATION  — edit these values to change what to run
    # ================================================================

    # Available bands: "C", "CL", "SCL", "SCLO"
    BAND_SELECTION = "SCL"

    # Available route functions: "ILP-connections", "kSP-FF", "FF-kSP"
    ROUTE_FUNCTION = "kSP-FF"

    # Topology name in MongoDB
    TOPOLOGY_NAME = "NSFNET"   # Options: "NSFNET", "DTAG", "CONUS", etc.

    # MongoDB location of the topology / RWA data
    # DTAG is in "real"; generated topologies are typically in "topology-paper"
    COLLECTION = "real" if TOPOLOGY_NAME == "DTAG" else "topology-paper"
    DB = "Topology_Data"

    # Physical parameters
    SPAN_LENGTH_KM = 80          # Fiber span length [km]
    LAUNCH_POWER_DBM = 0.0       # Uniform per-channel launch power [dBm]
    NOISE_FIGURE_DB = 5.0        # EDFA noise figure [dB]
    SAMPLES_PER_KM = 2           # ISRS ODE spatial sampling density

    # Ray parallel execution
    USE_PARALLEL = False         # Set True to use Ray
    HOSTNAME = "128.40.42.13"    # Ray head node (None for local Ray)
    PORT = 6379

    # ================================================================
    #  Command-line overrides (if any flags are given, they win)
    # ================================================================
    cli = main()
    band       = cli.band            or BAND_SELECTION
    route_fn   = cli.route_function  or ROUTE_FUNCTION
    topo       = cli.topology        or TOPOLOGY_NAME
    collection = cli.collection      or COLLECTION
    db         = cli.db              or DB
    span_km    = cli.span_length_km  if cli.span_length_km  is not None else SPAN_LENGTH_KM
    lp_dBm     = cli.launch_power_dBm if cli.launch_power_dBm is not None else LAUNCH_POWER_DBM
    nf_dB      = cli.noise_figure_dB if cli.noise_figure_dB is not None else NOISE_FIGURE_DB
    spk        = cli.samples_per_km  if cli.samples_per_km  is not None else SAMPLES_PER_KM
    parallel   = cli.parallel        or USE_PARALLEL
    hostname   = cli.hostname        or HOSTNAME
    port       = cli.port            if cli.port is not None else PORT

    # Validate band
    if band not in BAND_CONFIGS:
        raise ValueError(f"Invalid band: {band}. Options: {list(BAND_CONFIGS.keys())}")

    band_cfg = BAND_CONFIGS[band]
    B_o = band_cfg["B_o_THz"] * 1e12
    ch_bw_hz = band_cfg["channel_bandwidth_GHz"] * 1e9
    num_ch = int(np.floor(B_o / ch_bw_hz))

    # ================================================================
    #  Print configuration summary
    # ================================================================
    print("=" * 60)
    print("CFM-based Throughput Computation")
    print("=" * 60)
    print(f"Route function   : {route_fn}")
    print(f"Topology         : {topo}")
    print(f"DB / Collection  : {db} / {collection}")
    print(f"Band             : {band_cfg['name']}")
    print(f"  Wavelength     : {band_cfg['wavelength_start_nm']:.0f} - "
          f"{band_cfg['wavelength_start_nm'] + band_cfg['wavelength_width_nm']:.0f} nm")
    print(f"  Bandwidth      : {band_cfg['B_o_THz']:.2f} THz")
    print(f"  Channels       : {num_ch}")
    print(f"  Ref Lambda     : {band_cfg['RefLambda_nm']:.1f} nm")
    print(f"Span length      : {span_km} km")
    print(f"Launch power     : {lp_dBm} dBm")
    print(f"Noise figure     : {nf_dB} dB")
    print(f"Parallel (Ray)   : {parallel}")
    print("=" * 60)

    # ================================================================
    #  Run
    # ================================================================
    if parallel:
        run_parallel(
            db=db,
            collection=collection,
            topology_name=topo,
            route_function=route_fn,
            band_selection=band,
            span_length_km=span_km,
            launch_power_dBm=lp_dBm,
            noise_figure_dB=nf_dB,
            samples_per_km=spk,
            hostname=hostname,
            port=port,
        )
    else:
        run_sequential(
            db=db,
            collection=collection,
            topology_name=topo,
            route_function=route_fn,
            band_selection=band,
            span_length_km=span_km,
            launch_power_dBm=lp_dBm,
            noise_figure_dB=nf_dB,
            samples_per_km=spk,
        )
