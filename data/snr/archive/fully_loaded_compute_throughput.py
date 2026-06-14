"""
CFM throughput with a *fully loaded* RWA derived from the stored sparse RWA.

Reads the same MongoDB topology + band-aware RWA field as ``compute_throughput_cfm.py``,
collects every end-to-end node path that appears on any wavelength, deduplicates them,
then assigns **that full set of paths to every active in-band channel** so that each
logical channel carries the same route multiset (full spectrum on every edge touched
by any original lightpath).

Results are written under distinct Mongo keys (``*-fully-loaded``) so the baseline
CFM throughput fields are not overwritten.

Run from this directory (same expectations as ``compute_throughput_cfm.py``):

    python fully_loaded_compute_throughput_cfm.py --topology NSFNET --band SCL
"""
from __future__ import annotations

import argparse
import datetime
import time

import compute_throughput_cfm as cfm
import networkx as nx
import numpy as np
from jax import numpy as jnp
from scipy.constants import c

nt = cfm.nt


def expand_rwa_fully_loaded(rwa_sparse: dict, num_active_channels: int) -> dict:
    """Turn sparse RWA into per-channel full assignment.

    For each wavelength slot ``w`` in ``0 .. num_active_channels-1``, the value is
    the list of **unique** node paths (deduped by tuple of ints) that appear anywhere
    in ``rwa_sparse``.  Thus every edge used by any original route carries every
    active channel in the occupancy matrix (fully loaded fibre on that edge set).

    Parameters
    ----------
    rwa_sparse : dict
        Same convention as Mongo / ``read_database_dict``: ``str(channel) -> list of node paths``.
    num_active_channels : int
        Active in-band channel count from the multi-band grid.

    Returns
    -------
    dict
        ``str(w) -> list[list[int]]`` for ``w`` in ``range(num_active_channels)``.
    """
    unique_paths: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()

    for w_str in sorted(rwa_sparse.keys(), key=lambda x: int(x)):
        paths = rwa_sparse[w_str]
        if not isinstance(paths, list):
            paths = [paths]
        for node_path in paths:
            if not node_path or len(node_path) < 2:
                continue
            key = tuple(int(n) for n in node_path)
            if key in seen:
                continue
            seen.add(key)
            unique_paths.append(list(key))

    return {str(w): [list(p) for p in unique_paths] for w in range(num_active_channels)}


def compute_topology_throughput_fully_loaded(
    db: str,
    collection: str,
    _id,
    band_selection: str,
    route_function: str,
    span_length_km: float = 80,
    launch_power_dBm: float = 2.0,
    save_snr_to_disk: bool = True,
):
    """Same pipeline as ``compute_throughput_cfm.compute_topology_throughput`` but RWA
    is replaced by ``expand_rwa_fully_loaded`` before occupancy and Shannon sums.
    """
    band_cfg = cfm.BAND_CONFIGS[band_selection]
    active_bands = band_cfg["bands"]

    results = list(
        nt.Database.read_data(db, collection, find_dic={"_id": _id}, max_count=1)
    )
    if not results:
        print(f"[WARN] No document found for _id={_id}")
        return 0.0

    doc = results[0]
    rwa_key = cfm._build_band_aware_rwa_key(route_function, band_selection)
    if rwa_key not in doc:
        raise KeyError(
            f"Band-aware key '{rwa_key}' not found for _id={_id}. "
            "Need sparse RWA in DB to infer path set for fully loaded expansion."
        )

    graph_list = nt.Database.read_topology_dataset_list(
        db, collection, find_dic={"_id": _id}, node_data=True
    )
    if not graph_list:
        print(f"[WARN] Could not read topology for _id={_id}")
        return 0.0

    graph = graph_list[0][0]
    graph = nx.relabel.convert_node_labels_to_integers(graph, first_label=1)

    rwa_sparse = nt.Tools.read_database_dict(doc[rwa_key])
    ch_lambda, channel_idx, ch_idx_oband_full, band_masks, ref_lambda = (
        cfm._build_multiband_channel_grid(active_bands)
    )
    num_active = int(np.sum(channel_idx))
    rwa = expand_rwa_fully_loaded(rwa_sparse, num_active)

    n_sparse_items = sum(
        len(p) if isinstance(p, list) else 1 for p in rwa_sparse.values()
    )
    n_unique_paths = len(rwa.get("0", [])) if num_active else 0
    print(
        f"  Fully loaded RWA: {n_unique_paths} unique path(s) × {num_active} channel(s) "
        f"(sparse RWA had {n_sparse_items} path list items across λ keys)."
    )

    ch_idx_oband_active = ch_idx_oband_full[channel_idx]

    print(
        f"  Channel grid: {cfm.NUM_CHANNELS_TOTAL} slots, {num_active} active channels, "
        f"ref_lambda={ref_lambda*1e9:.1f} nm"
    )

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

    (edges, edge_to_row, span_count_per_edge, occupancy_matrix, rwa_edge_paths,
     wavelength_to_col) = cfm._build_rwa_occupancy(graph, rwa, num_active)
    active_slot_ix = np.flatnonzero(np.asarray(channel_idx))

    nsr_link_channel = jnp.zeros_like(jnp.asarray(occupancy_matrix), dtype=jnp.float64)
    fit_params_per_link: dict = {}
    eta_spm_per_link: dict[int, np.ndarray] = {}
    eta_xpm_per_link: dict[int, np.ndarray] = {}
    eta_fwm_per_link: dict[int, np.ndarray] = {}
    nsr_ase_per_link: dict[int, np.ndarray] = {}

    n_links = int(span_count_per_edge.shape[0])
    print(f"    Undirected links in topology: {n_links} (Link indices 0..{n_links - 1})")

    for i in range(n_links):
        t0 = time.perf_counter()
        link_nsr, link_fit_params, eta_spm, eta_xpm, eta_fwm, link_nsr_ase = cfm.calc_NSR_link(
            setup,
            float(span_count_per_edge[i]),
            occupancy_matrix[i, :, None],
            ch_idx_oband_active,
        )
        link_nsr = link_nsr.squeeze(-1)
        fit_params_per_link[i] = np.asarray(link_fit_params.squeeze(-1))
        eta_spm_per_link[i] = np.asarray(eta_spm, dtype=np.float64).reshape(-1)
        eta_xpm_per_link[i] = np.asarray(eta_xpm, dtype=np.float64).reshape(-1)
        eta_fwm_per_link[i] = np.asarray(eta_fwm, dtype=np.float64).reshape(-1)
        nsr_ase_per_link[i] = np.asarray(link_nsr_ase.squeeze(-1), dtype=np.float64)
        nsr_link_channel = nsr_link_channel.at[i].set(link_nsr)
        dt = time.perf_counter() - t0
        occ = occupancy_matrix[i] > 0
        ln = np.asarray(link_nsr).ravel()
        use = occ & np.isfinite(ln) & (ln > 0)
        if not np.any(occ):
            snr_msg = "n/a (idle link)"
        elif np.any(use):
            mean_snr = float(np.mean(10.0 * np.log10(1.0 / ln[use])))
            snr_msg = f"{mean_snr:.1f} dB"
        else:
            snr_msg = "nan (no finite NSR on occupied λ)"
        print(f"    Link {i} ({edges[i]}, spans={span_count_per_edge[i]}): "
              f"{dt:.1f}s, mean SNR={snr_msg}")

    tag = "fully-loaded"
    if save_snr_to_disk:
        snr_dir = cfm._SCRIPT_DIR / "data" / "snr"
        snr_dir.mkdir(parents=True, exist_ok=True)
        ch_lambda_active = np.asarray(ch_lambda)[np.asarray(channel_idx)]
        ch_freq_active = c / ch_lambda_active
        nsr_np = np.asarray(nsr_link_channel)

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
            fname = f"{_id}_{band_selection}_{route_function}_{tag}_link{i}_{u}-{v}.npz"
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

        occ_fname = f"{_id}_{band_selection}_{route_function}_{tag}_occupancy.npz"
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
            rate_bps = 2 * ch_bw_hz * np.log2(1.0 + snr)
            capacity_total += rate_bps
            n_lightpaths += 1

    print(
        f"  Topology {_id} [{tag}]: {n_lightpaths} lightpaths, "
        f"capacity = {capacity_total/1e12:.4f} Tbps"
    )

    nt.Database.update_data_with_id(
        db,
        collection,
        _id,
        newvals={
            "$set": {
                f"{route_function} Capacity-CFM-fully-loaded": float(capacity_total),
                f"{route_function} CFM-fully-loaded-lightpaths": int(n_lightpaths),
                f"{route_function} CFM-fully-loaded-timestamp": datetime.datetime.utcnow(),
                f"{route_function} CFM-fully-loaded-band": band_cfg["name"],
                f"{route_function} CFM-fully-loaded-span_length_km": float(span_length_km),
                f"{route_function} CFM-fully-loaded-launch_power_dBm": float(launch_power_dBm),
            }
        },
    )

    return capacity_total


def run_parallel(
    db: str,
    collection: str,
    topology_name: str,
    route_function: str,
    band_selection: str,
    span_length_km: float = 80,
    launch_power_dBm: float = -2.0,
    save_snr_to_disk: bool = True,
    hostname: str = "128.40.42.13",
    port: int = 6379,
):
    import ray

    if hostname is not None:
        ray.init(address=f"{hostname}:{port}")
    else:
        ray.init()

    graph_list = nt.Database.read_topology_dataset_list(
        db, collection, find_dic={"name": topology_name}, node_data=True
    )
    print(f"Found {len(graph_list)} topologies for '{topology_name}'")

    @ray.remote(num_cpus=1)
    def _remote_worker(_id):
        return compute_topology_throughput_fully_loaded(
            db=db,
            collection=collection,
            _id=_id,
            band_selection=band_selection,
            route_function=route_function,
            span_length_km=span_length_km,
            launch_power_dBm=launch_power_dBm,
            save_snr_to_disk=save_snr_to_disk,
        )

    futures = [_remote_worker.remote(_id) for _, _id in graph_list]
    results = ray.get(futures)

    ray.shutdown()
    print(f"\nAll done. {len(results)} topologies processed.")
    for i, (_, _id) in enumerate(graph_list):
        print(f"  {_id}: {results[i]/1e12:.4f} Tbps")

    return results


def run_sequential(
    db: str,
    collection: str,
    topology_name: str,
    route_function: str,
    band_selection: str,
    span_length_km: float = 80,
    launch_power_dBm: float = -2.0,
    save_snr_to_disk: bool = True,
):
    graph_list = nt.Database.read_topology_dataset_list(
        db, collection, find_dic={"name": topology_name}, node_data=True
    )
    print(f"Found {len(graph_list)} topologies for '{topology_name}'")

    results = []
    for graph, _id in graph_list:
        cap = compute_topology_throughput_fully_loaded(
            db=db,
            collection=collection,
            _id=_id,
            band_selection=band_selection,
            route_function=route_function,
            span_length_km=span_length_km,
            launch_power_dBm=launch_power_dBm,
            save_snr_to_disk=save_snr_to_disk,
        )
        results.append(cap)

    print(f"\nAll done. {len(results)} topologies processed.")
    for i, (_, _id) in enumerate(graph_list):
        print(f"  {_id}: {results[i]/1e12:.4f} Tbps")

    return results


def _cli():
    p = argparse.ArgumentParser(
        description="CFM throughput with fully loaded RWA (all channels on all sparse paths)"
    )
    p.add_argument("--route_function", type=str, default=None)
    p.add_argument("--topology", type=str, default=None)
    p.add_argument("--collection", type=str, default=None)
    p.add_argument("--db", type=str, default=None)
    p.add_argument(
        "--band",
        type=str,
        default=None,
        choices=list(cfm.BAND_CONFIGS.keys()),
    )
    p.add_argument("--span_length_km", type=float, default=None)
    p.add_argument("--launch_power_dBm", type=float, default=None)
    p.add_argument("--parallel", action="store_true")
    p.add_argument("--hostname", type=str, default=None)
    p.add_argument("--port", type=int, default=None)
    return p.parse_args()


if __name__ == "__main__":
    SAVE_SNR_TO_DISK = True
    BAND_SELECTION = "OESCL"
    ROUTE_FUNCTION = "kSP-FF"
    TOPOLOGY_NAME = "NSFNET"
    DB = "Topology_Data"
    COLLECTION = "real"
    SPAN_LENGTH_KM = 80
    LAUNCH_POWER_DBM = 3.0
    USE_PARALLEL = False
    HOSTNAME = "128.40.40.67"
    PORT = 6379

    cli = _cli()
    band = cli.band or BAND_SELECTION
    route_fn = cli.route_function or ROUTE_FUNCTION
    topo = cli.topology or TOPOLOGY_NAME
    collection = cli.collection or COLLECTION
    db = cli.db or DB
    span_km = cli.span_length_km if cli.span_length_km is not None else SPAN_LENGTH_KM
    lp_dBm = cli.launch_power_dBm if cli.launch_power_dBm is not None else LAUNCH_POWER_DBM
    parallel = cli.parallel or USE_PARALLEL
    hostname = cli.hostname or HOSTNAME
    port = cli.port if cli.port is not None else PORT

    if band not in cfm.BAND_CONFIGS:
        raise ValueError(f"Invalid band: {band}")

    band_cfg = cfm.BAND_CONFIGS[band]
    _, channel_idx, _, _, ref_lambda_main = cfm._build_multiband_channel_grid(band_cfg["bands"])
    num_active = int(np.sum(channel_idx))

    print("=" * 60)
    print("CFM throughput — fully loaded RWA (all λ × union of DB paths)")
    print("=" * 60)
    print(f"Route function   : {route_fn}")
    print(f"Topology         : {topo}")
    print(f"DB / Collection  : {db} / {collection}")
    print(f"Band             : {band_cfg['name']}")
    print(f"Active channels  : {num_active}")
    print(f"Ref lambda (disp): {ref_lambda_main*1e9:.1f} nm")
    print(f"Span / launch    : {span_km} km, {lp_dBm} dBm")
    print(f"Parallel (Ray)   : {parallel}")
    print(f"Save SNR to disk : {SAVE_SNR_TO_DISK}")
    print("=" * 60)

    if parallel:
        run_parallel(
            db=db,
            collection=collection,
            topology_name=topo,
            route_function=route_fn,
            band_selection=band,
            span_length_km=span_km,
            launch_power_dBm=lp_dBm,
            save_snr_to_disk=SAVE_SNR_TO_DISK,
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
            save_snr_to_disk=SAVE_SNR_TO_DISK,
        )