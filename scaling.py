import NetworkToolkit as nt
import random
import collections
import matplotlib
import networkx as nx
import matplotlib.pyplot as plt
import ray
import numpy as np
import os
import re
import traceback
# import powerlaw as pl
import scipy
# import statsmodels.api as sm
# import tikzplotlib
import ast
from scipy.stats import ks_2samp, kstest, sem, t
from tqdm import tqdm
import datetime
from copy import deepcopy

# ============================================================
# Band Selection Configuration (must match ilp_connections.py)
# ============================================================
BAND_CONFIGS = {
    "C": {
        "name": "C band",
        "wavelength_start_nm": 1530,
        "wavelength_width_nm": 40.049,
        "B_o_THz": 5.0,
        "RefLambda_nm": 1550,
        "Cr": 0,
        "channel_bandwidth_GHz": 50,
        "description": "C band: 1530-1570 nm (40 nm)"
    },
    "CL": {
        "name": "C+L band",
        "wavelength_start_nm": 1530,
        "wavelength_width_nm": 95,
        "B_o_THz": 11.8,
        "RefLambda_nm": 1577.5,
        "Cr": 0.028 / 1e3 / 1e12,
        "channel_bandwidth_GHz": 50,
        "description": "C+L band: 1530-1625 nm (95 nm)"
    },
    "SCL": {
        "name": "SCL band (Super C+L)",
        "wavelength_start_nm": 1460,
        "wavelength_width_nm": 165,
        "B_o_THz": 20.86,
        "RefLambda_nm": 1542.5,
        "Cr": 0.028 / 1e3 / 1e12,
        "channel_bandwidth_GHz": 50,
        "description": "SCL band: 1460-1625 nm (165 nm)"
    },
    "SCLO": {
        "name": "SCLO band (Super C+L+O)",
        "wavelength_start_nm": 1260,
        "wavelength_width_nm": 365,
        "B_o_THz": 46.0,
        "RefLambda_nm": 1442.5,
        "Cr": 0.028 / 1e3 / 1e12,
        "channel_bandwidth_GHz": 50,
        "description": "SCLO band: 1260-1625 nm (365 nm)"
    }
}


def parralel_calculate_new_throughput(graph_list):
    pb = nt.Tools.ProgressBar(len(graph_list))
    actor = pb.actor
    tasks = [calculate_new_throughput.remote(item, actor=actor) for item in graph_list]
    pb.print_until_done()
    throughput_vals = ray.get(tasks)
    return throughput_vals

def calculate_edges(graph_list):
    edge_list = []
    for graph, _id, rwa in tqdm(graph_list):
        edge_list.append(graph.number_of_edges())
    return edge_list

def calculate_lightpaths_rwa(graph_list):
    lightpaths_list = []
    for graph, _id, rwa in tqdm(graph_list):
        lightpaths_list.append(sum([1  for wave in rwa for path in rwa[wave]]))

    return lightpaths_list

def get_box_plot_data(boxplot, ind=0):
    whiskers = [item.get_ydata() for item in boxplot['whiskers']]
    _median = boxplot['medians'][0].get_ydata()[0]
    _75 = whiskers[1][0]
    _25 = whiskers[0][0]
    _top = whiskers[1][1]
    _bottom = whiskers[0][1]
    _data = np.array([ind, _median, _75, _25, _top, _bottom])
    return _data

def calculate_new_throughput_single(item, band="C", span_length_km=80, actor=None):
    try:
        graph = item[0]
        rwa = {int(wave): [path for path in item[2][wave]] for wave in item[2]}

        # Get band configuration
        if band not in BAND_CONFIGS:
            raise ValueError(f"Invalid band: {band}. Available: {list(BAND_CONFIGS.keys())}")

        band_config = BAND_CONFIGS[band]
        channel_bandwidth = band_config['channel_bandwidth_GHz'] * 1e9  # Convert GHz to Hz
        B_o = band_config['B_o_THz'] * 1e12  # Convert THz to Hz

        # Create network with correct parameters
        network = nt.Network.OpticalNetwork(
            graph,
            B_o=B_o,
            channel_bandwidth=channel_bandwidth,
            fibre_num=1,
            span_length_km=span_length_km
        )

        # Configure physical layer wavelengths
        network.physical_layer.assign_physical_wavelengths(
            channels=network.channels,
            wavelength_start_nm=band_config['wavelength_start_nm'],
            wavelength_width_nm=band_config['wavelength_width_nm']
        )

        # Filter RWA to only include wavelengths within valid range
        max_wavelength = network.channels - 1
        rwa_filtered = {int(wave): paths for wave, paths in rwa.items() 
                        if 0 <= int(wave) <= max_wavelength}
        
        if len(rwa_filtered) == 0:
            print(f"ERROR: No valid wavelengths in RWA for band {band} (channels: {network.channels})")
            if rwa:
                print(f"  RWA wavelength range: {min(rwa.keys())} - {max(rwa.keys())}")
                print(f"  Valid range: 0-{max_wavelength}")
            return 0.0

        network.physical_layer.add_uniform_launch_power_to_links(network.channels)
        # network.physical_layer.add_LOGON_launch_power_to_links(network.channels)
        network.physical_layer.add_wavelengths_to_links(rwa_filtered)
        network.physical_layer.add_non_linear_NSR_to_links(
            channel_bandwidth=channel_bandwidth,
            channels_full=network.channels
        )
        throughput = network.physical_layer.get_lightpath_capacities_PLI(rwa_filtered)[0]
        if actor:
            actor.update.remote(1)
    except Exception as err:
        traceback.print_exc()
        return 0.0
    return throughput

@ray.remote
def calculate_new_throughput(item, band="C", span_length_km=80, actor=None):
    try:
        graph = item[0]
        rwa = {int(wave): [path for path in item[2][wave]] for wave in item[2]}

        # Get band configuration
        if band not in BAND_CONFIGS:
            raise ValueError(f"Invalid band: {band}. Available: {list(BAND_CONFIGS.keys())}")

        band_config = BAND_CONFIGS[band]
        channel_bandwidth = band_config['channel_bandwidth_GHz'] * 1e9  # Convert GHz to Hz
        B_o = band_config['B_o_THz'] * 1e12  # Convert THz to Hz

        # Create network with correct parameters
        network = nt.Network.OpticalNetwork(
            graph,
            B_o=B_o,
            channel_bandwidth=channel_bandwidth,
            fibre_num=1,
            span_length_km=span_length_km
        )

        # Configure physical layer wavelengths
        network.physical_layer.assign_physical_wavelengths(
            channels=network.channels,
            wavelength_start_nm=band_config['wavelength_start_nm'],
            wavelength_width_nm=band_config['wavelength_width_nm']
        )

        # Filter RWA to only include wavelengths within valid range
        max_wavelength = network.channels - 1
        
        # Debug: Print RWA and channel information
        if rwa:
            rwa_min = min(rwa.keys())
            rwa_max = max(rwa.keys())
            print(f"DEBUG calculate_new_throughput: Band={band}, Channels={network.channels}, Valid range: 0-{max_wavelength}")
            print(f"DEBUG: RWA wavelength range: {rwa_min} - {rwa_max}, Total wavelengths in RWA: {len(rwa)}")
            print(f"DEBUG: B_o={B_o/1e12:.2f} THz, channel_bandwidth={channel_bandwidth/1e9:.1f} GHz")
        
        # Filter RWA: only keep wavelengths within [0, network.channels)
        rwa_filtered = {}
        out_of_range_count = 0
        for wave, paths in rwa.items():
            wave_int = int(wave)
            if 0 <= wave_int <= max_wavelength:
                rwa_filtered[wave_int] = paths
            else:
                out_of_range_count += 1
        
        if len(rwa_filtered) == 0:
            print(f"ERROR: No valid wavelengths in RWA for band {band} (channels: {network.channels})")
            if rwa:
                print(f"  RWA wavelength range: {min(rwa.keys())} - {max(rwa.keys())}")
                print(f"  Valid range: 0-{max_wavelength}")
                print(f"  This suggests RWA was computed with a different band configuration!")
            return 0.0
        
        if out_of_range_count > 0:
            print(f"WARNING: Filtered {out_of_range_count} wavelengths out of range for band {band}")
            if rwa:
                print(f"  Valid range: 0-{max_wavelength}, RWA range: {min(rwa.keys())}-{max(rwa.keys())}")
                print(f"  Kept {len(rwa_filtered)} valid wavelengths out of {len(rwa)} total")

        network.physical_layer.add_uniform_launch_power_to_links(network.channels)
        # network.physical_layer.add_LOGON_launch_power_to_links(network.channels)
        network.physical_layer.add_wavelengths_to_links(rwa_filtered)
        network.physical_layer.add_non_linear_NSR_to_links(
            channel_bandwidth=channel_bandwidth,
            channels_full=network.channels
        )
        throughput = network.physical_layer.get_lightpath_capacities_PLI(rwa_filtered)[0]
        if actor:
            actor.update.remote(1)
    except Exception as err:
        traceback.print_exc()
        return 0.0
    return throughput

def update_networks(db, collection, graph_list, band="C"):
    """
    Update networks with throughput results.

    Parameters:
    -----------
    db : str
        Database name
    collection : str
        Collection name
    graph_list : list
        List of (graph, _id, throughput) tuples
    band : str
        Band selection (C, SCL, SCLO, etc.)
    """
    # Get band configuration for channel parameters
    if band not in BAND_CONFIGS:
        raise ValueError(f"Invalid band: {band}")

    band_config = BAND_CONFIGS[band]
    channel_bandwidth = band_config['channel_bandwidth_GHz'] * 1e9  # Convert GHz to Hz
    B_o = band_config['B_o_THz'] * 1e12  # Convert THz to Hz
    num_channels = int(np.floor(B_o / channel_bandwidth))

    for graph, _id, throughput in tqdm(graph_list):
        nt.Database.update_data_with_id(
            db, collection, _id,
            newvals={
                "$set": {
                    "ILP Capacity": float(throughput),
                    "Erratum capacity bandwidth": channel_bandwidth,
                    "Erratum capacity channels": num_channels,
                    "Erratum capacity timestamp": datetime.datetime.utcnow(),
                    "Erratum capacity band": band
                }
            }
        )

def write_graph(graph, rwa, db, coll, T_c, scale, graph_type=None, **kwargs):
    """
    Writes the graph to the specified database and collection.

    Parameters:
    - graph (Graph): The graph to write.
    - db (str): The name of the database.
    - coll (str): The name of the collection.
    - T_c (float): The value of T_c.
    - scale (float): scale of graphs.

    Returns:
    None
    """
    data = {
        "erratum timestamp": datetime.datetime.utcnow(),
        "T_c": T_c.tolist(),
        "original ILP capacity RWA": rwa,
        "scale": scale
    }

    # Merge data and kwargs, then pass to insert_graph
    all_data = {**data, **kwargs}

    # Debug: print what we're trying to write (first few times)
    if "scale" in all_data and all_data.get("scale", 0) in [0.1, 0.2, 0.3]:
        print(f"  DEBUG write_graph: db={db}, coll={coll}, scale={all_data.get('scale')}, "
              f"name={all_data.get('name')}, throughput={all_data.get('ilp_throughput', 'N/A')}")
        print(f"  DEBUG all_data keys: {list(all_data.keys())}")

    try:
        nt.Database.insert_graph(graph, db_name=db, collection_name=coll, **all_data)
        if "scale" in all_data and all_data.get("scale", 0) in [0.1, 0.2, 0.3]:
            print(f"  DEBUG: insert_graph call completed successfully")
    except Exception as e:
        print(f"Error in write_graph: {e}")
        print(f"  db={db}, coll={coll}, scale={all_data.get('scale')}, name={all_data.get('name')}")
        traceback.print_exc()
        raise

def scale_graphs(graph_list, scale):
    for item in graph_list:
        if type(item) == list or type(item) == tuple:
            graph = item[0]
            other_data = item[1:]
        else:
            graph = item
            other_data = None
        scaled_graph = deepcopy(graph)
        for u, v in graph.edges:
            scaled_graph[u][v]["weight"] = int(scale*scaled_graph[u][v]["weight"])
        if other_data:
            return (scaled_graph, other_data)
        else:
            return scaled_graph

def scale_graph(graph, _id, rwa, name, scale, actor=None):

    scaled_graph = deepcopy(graph)
    # print(type(scaled_graph))
    # print(len(list(graph.edges)), scale)
    for u, v in graph.edges:
        scaled_graph[u][v]["weight"] = int(scale*scaled_graph[u][v]["weight"])
    if actor:
        actor.update.remote(1)

    return scaled_graph, _id, rwa, name, np.round(scale, decimals=2)



top = nt.Topology.Topology()


if __name__ == "__main__":
    # hostname = "128.40.41.48"
    # port = 7112
    # ray.init(address='{}:{}'.format(hostname, port), _redis_password='5241590000000000',
    #             ignore_reinit_error=True)
    ray.init(ignore_reinit_error=True)

    # ============================================================
    # Configuration Parameters
    # ============================================================
    # Topology names to process
    topologies = ["DTAG"]  # Options: ["NSFNET"], ["DTAG"], ["CONUS"], or ["NSFNET", "DTAG", "CONUS"]

    # Bands to process
    # bands = ["C", "CL", "SCL", "SCLO"]  # Can modify to process specific bands
    bands = ["SCLO"]  # Must be a list, not a string!
    # Data source collections (map topology name to collection)
    collection_sources = {
        "NSFNET": "topology-paper",
        "CONUS": "topology-paper",
        "DTAG": "real",  # DTAG is stored in "real" collection
    }

    # Scaling parameters
    scales = np.arange(0.1, 1.01, 0.1)  # Distance scaling factors

    # Physical layer parameters
    span_length_km = 80  # Fiber span length in km (must match ilp_connections.py)

    # ============================================================
    # Process each topology and band combination
    # ============================================================
    for dic in topologies:
        for band in bands:
            print(f"\n{'='*60}")
            print(f"Processing: {dic} with {band} band")
            print(f"{'='*60}\n")

            # Validate band configuration
            if band not in BAND_CONFIGS:
                print(f"Warning: Invalid band '{band}'. Skipping...")
                continue

            band_config = BAND_CONFIGS[band]
            print(f"Band Configuration:")
            print(f"  Name: {band_config['name']}")
            print(f"  Description: {band_config['description']}")
            print(f"  Total Bandwidth: {band_config['B_o_THz']:.2f} THz")
            print(f"  Channel Bandwidth: {band_config['channel_bandwidth_GHz']:.1f} GHz")
            print(f"  Wavelength Range: {band_config['wavelength_start_nm']:.1f} - "
                  f"{band_config['wavelength_start_nm'] + band_config['wavelength_width_nm']:.1f} nm")
            print()

            # Determine source collection
            source_collection = collection_sources.get(dic, "topology-paper")

            # Build RWA field name
            rwa_field_name = f"ILP-connections RWA {band}"

            # Read data from database
            graph_list = []
            try:
                graph_list += nt.Database.read_topology_dataset_list(
                    "Topology_Data",
                    source_collection,
                    rwa_field_name,
                    "name",
                    find_dic={
                        "name": dic,
                        rwa_field_name: {"$exists": True}
                    },
                    node_data=True
                )
            except KeyError as e:
                print(f"Warning: No {rwa_field_name} found for {dic} in {source_collection}. Skipping...")
                print(f"Error: {e}")
                continue
            except Exception as e:
                print(f"Error reading {dic} with {band} band: {e}")
                traceback.print_exc()
                continue

            if len(graph_list) == 0:
                print(f"Warning: No graphs found with {rwa_field_name} for {dic}. Skipping...")
                continue

            print(f"Found {len(graph_list)} graphs with {rwa_field_name} for {dic}")

            # Process scaling for current topology and band
            pb = nt.Tools.ProgressBar(len(graph_list)*len(scales))
            actor = pb.actor
            scaled_graphs = [
                scale_graph(graph, _id, rwa, name, scale)
                for graph, _id, rwa, name in tqdm(graph_list, desc=f"Scaling {dic} {band}")
                for scale in scales
            ]

            print(f"number of tasks for {dic} {band}: {len(scaled_graphs)}")

            pb = nt.Tools.ProgressBar(len(graph_list)*len(scales))
            actor = pb.actor
            # Pass band and span_length_km parameters
            throughput_tasks = [
                calculate_new_throughput.remote(
                    (graph, _id, rwa),
                    band=band,
                    span_length_km=span_length_km,
                    actor=actor
                )
                for graph, _id, rwa, name, scale in tqdm(scaled_graphs, desc=f"throughput tasks {dic} {band}")
            ]

            pb.print_until_done()
            throughput_results = ray.get(throughput_tasks)
            print(f"Got {len(throughput_results)} throughput results")
            scaled_throughput = iter(throughput_results)

            # Build collection name: topology_{dic}_distance_scaling_{band}
            collection_name = f"topology_{dic}_distance_scaling_{band}"

            print(f"\nPreparing to write {len(scaled_graphs)} graphs to collection: {collection_name}")
            written_count = 0
            error_count = 0

            for idx, (graph, _id, rwa, name, scale) in enumerate(tqdm(scaled_graphs, desc=f"writing graphs {dic} {band}")):
                try:
                    throughput = next(scaled_throughput)
                    T_c = np.ones((len(graph),len(graph)))
                    np.fill_diagonal(T_c, 0)
                    T_c /= T_c.sum()

                    if idx < 3:  # Print first 3 writes for debugging
                        print(f"\nWriting graph {idx+1}/{len(scaled_graphs)}: scale={scale}, throughput={throughput:.2e}, name={name}")

                    write_graph(
                        graph, rwa, "Topology_Data", collection_name, T_c,
                        scale=scale,
                        name=name,
                        ilp_throughput=throughput,
                        band=band,
                        topology_name=dic
                    )
                    written_count += 1

                    if idx < 3:  # Print first 3 successful writes
                        print(f"  Successfully written graph {idx+1}")

                except StopIteration:
                    print(f"\nError: Not enough throughput values. Expected {len(scaled_graphs)}, got {written_count + error_count}.")
                    print(f"  Written: {written_count}, Errors: {error_count}")
                    break
                except Exception as e:
                    error_count += 1
                    print(f"\nError writing graph {idx+1} with scale {scale}: {e}")
                    traceback.print_exc()
                    if error_count <= 3:  # Print first 3 errors in detail
                        print(f"  Graph info: nodes={len(graph.nodes)}, edges={len(graph.edges)}, name={name}")
                    continue

            print(f"Completed processing {dic} with {band} band")
            print(f"Results saved to collection: {collection_name}")
            print(f"  Successfully written: {written_count} documents")
            if error_count > 0:
                print(f"  Errors encountered: {error_count} documents")

            # Verify data was written
            print(f"\nVerifying data in collection: {collection_name}")
            try:
                verify_data = list(nt.Database.read_data(
                    "Topology_Data",
                    collection_name,
                    find_dic={"name": dic},
                    max_count=5
                ))
                print(f"Found {len(verify_data)} documents in {collection_name}")
                if len(verify_data) > 0:
                    print(f"Sample document keys: {list(verify_data[0].keys())}")
                    if "ilp_throughput" in verify_data[0]:
                        print(f"Sample ilp_throughput: {verify_data[0]['ilp_throughput']}")
                    if "scale" in verify_data[0]:
                        print(f"Sample scale: {verify_data[0]['scale']}")
                else:
                    print("WARNING: No documents found after writing!")
            except Exception as e:
                print(f"Error verifying data: {e}")
                traceback.print_exc()

            print()

    print("="*60)
    print("All processing completed!")
    print("="*60)
    # scaled_throughput = ray.get(throughput_tasks)
    # scaled_throughput = iter(scaled_throughput)
    # for name in graph_data:
    #     for graph_type in graph_data[name]:
    #         for graph, _id, rwa in tqdm(graph_data[name][graph_type], desc="writing graphs"):
    #             for ind,scale in enumerate(scales):
    #                 throughput = next(scaled_throughput)
    #                 T_c = np.ones((len(graph),len(graph)))
    #                 np.fill_diagonal(T_c, 0)
    #                 T_c /= T_c.sum()
    #                 scaled_graph = scale_graph(graph, scale)
    #                 write_graph(scaled_graph, rwa, "Topology_Data", "topology_paper_erratum_scaling", T_c, scale=scale,
    #                            name=name, graph_type=graph_type, scale_index=ind, ilp_throughput=throughput)



