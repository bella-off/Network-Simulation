import os

import NetworkToolkit as nt
import numpy as np
from scipy.constants import c as C_LIGHT

if __name__ == "__main__":
    # ============================================================
    # Band Selection Configuration
    # ============================================================
    # Keys match ``compute_throughput_cfm.BAND_CONFIGS`` (O, E, S, C, L, CL, SCL, ESCL, OESCL).
    BAND_SELECTION = "O"  # Change this to select different bands

    _CR_RAMAN = 0.028 / 1e3 / 1e12  # same as legacy CL / SCL / multi-band entries


    def _rwa_phys_continuous_nm(lam_min_nm: float, lam_max_nm: float, cr: float):
        """Continuous λ window → heuristic RWA fields (50 GHz channels, B_o from Δf)."""
        lam_min_m = lam_min_nm * 1e-9
        lam_max_m = lam_max_nm * 1e-9
        b_o_hz = C_LIGHT * (1.0 / lam_min_m - 1.0 / lam_max_m)
        return {
            "wavelength_start_nm": lam_min_nm,
            "wavelength_width_nm": lam_max_nm - lam_min_nm,
            "B_o_THz": b_o_hz / 1e12,
            "RefLambda_nm": (lam_min_nm + lam_max_nm) / 2.0,
            "Cr": cr,
            "channel_bandwidth_GHz": 50.0,
        }


    # Same structure as compute_throughput_cfm.py: name, bands, description;
    # plus legacy keys (wavelength_*, B_o_THz, RefLambda_nm, Cr, channel_bandwidth_GHz)
    # required by ``NetworkSimulator.parralel_heuristic_throughput`` — RWA logic unchanged.
    BAND_CONFIGS = {
        "O": {
            "name": "O band",
            "bands": ["O"],
            "description": "O band only",
            **_rwa_phys_continuous_nm(1260.0, 1358.0, _CR_RAMAN),
        },
        "E": {
            "name": "E band",
            "bands": ["E"],
            "description": "E band only",
            **_rwa_phys_continuous_nm(1405.0, 1464.0, _CR_RAMAN),
        },
        "S": {
            "name": "S band",
            "bands": ["S"],
            "description": "S band only",
            **_rwa_phys_continuous_nm(1470.0, 1526.0, _CR_RAMAN),
        },
        "C": {
            "name": "C band",
            "bands": ["C"],
            "description": "C band only",
            # Legacy C-band numbers (unchanged from original k_sp_ff / ilp_connections)
            "wavelength_start_nm": 1530.0,
            "wavelength_width_nm": 40.049,
            "B_o_THz": 5.0,
            "RefLambda_nm": 1550.0,
            "Cr": 0.0,
            "channel_bandwidth_GHz": 50.0,
        },
        "L": {
            "name": "L band",
            "bands": ["L"],
            "description": "L band only",
            **_rwa_phys_continuous_nm(1573.0, 1625.0, _CR_RAMAN),
        },
        "CL": {
            "name": "CL band",
            "bands": ["C", "L"],
            "description": "C+L bands",
            "wavelength_start_nm": 1530.0,
            "wavelength_width_nm": 95.0,
            "B_o_THz": 11.8,
            "RefLambda_nm": 1577.5,
            "Cr": _CR_RAMAN,
            "channel_bandwidth_GHz": 50.0,
        },
        "SCL": {
            "name": "SCL band",
            "bands": ["S", "C", "L"],
            "description": "S+C+L bands",
            "wavelength_start_nm": 1460.0,
            "wavelength_width_nm": 165.0,
            "B_o_THz": 20.86,
            "RefLambda_nm": 1542.5,
            "Cr": _CR_RAMAN,
            "channel_bandwidth_GHz": 50.0,
        },
        "ESCL": {
            "name": "ESCL band",
            "bands": ["E", "S", "C", "L"],
            "description": "E+S+C+L bands",
            **_rwa_phys_continuous_nm(1405.0, 1625.0, _CR_RAMAN),
        },
        "OESCL": {
            "name": "OESCL band",
            "bands": ["O", "E", "S", "C", "L"],
            "description": "All bands (O+E+S+C+L)",
            # Same OESCL / 878-channel window as original k_sp_ff (RWA behaviour preserved)
            "wavelength_start_nm": 1312.64,
            "wavelength_width_nm": 312.36,
            "B_o_THz": 43.91,
            "RefLambda_nm": 1452.22,
            "Cr": 0.028 / 1e3 / 1e12,
            "channel_bandwidth_GHz": 50.0,
            "num_channels": 880,
        },
    }

    # Get selected band configuration
    if BAND_SELECTION not in BAND_CONFIGS:
        raise ValueError(f"Invalid band selection: {BAND_SELECTION}. "
                         f"Available options: {list(BAND_CONFIGS.keys())}")

    band_config = BAND_CONFIGS[BAND_SELECTION]

    # Calculate channel bandwidth from band configuration
    channel_bandwidth = band_config['channel_bandwidth_GHz'] * 1e9  # Convert GHz to Hz

    # Calculate number of channels based on B_o and channel_bandwidth
    B_o = band_config['B_o_THz'] * 1e12  # Convert THz to Hz
    if band_config.get("num_channels") is not None:
        num_channels = int(band_config["num_channels"])
    else:
        num_channels = int(np.floor(B_o / channel_bandwidth))

    print("=" * 60)
    print(f"Selected Band: {band_config['name']}")
    print(f"Description: {band_config['description']}")
    print(f"Wavelength Range: {band_config['wavelength_start_nm']:.1f} - "
          f"{band_config['wavelength_start_nm'] + band_config['wavelength_width_nm']:.1f} nm")
    print(f"Total Bandwidth: {band_config['B_o_THz']:.2f} THz")
    print(f"Channel Bandwidth: {band_config['channel_bandwidth_GHz']:.1f} GHz")
    print(f"Number of Channels: {num_channels}")
    print(f"Reference Wavelength: {band_config['RefLambda_nm']:.1f} nm")
    print("=" * 60)

    # ============================================================
    # Fiber Span Length Configuration
    # ============================================================
    span_length_km = 80  # Fiber span length in km (default: 80 km)
    print(f"Fiber Span Length: {span_length_km} km")
    print("=" * 60)
    collection = "real"

    # collection = "topology-paper"
    db = "Topology_Data"


    # Ray: remote head (geneva / london) or local.
    # If you see "Failed to register worker to Raylet ... End of file", the remote
    # cluster is down, unreachable, or Ray versions mismatch — use local Ray:
    #   K_SP_FF_LOCAL_RAY=1 python k_sp_ff.py
    hostname =  "128.40.40.67"
    port = 6379
    # _local = os.environ.get("K_SP_FF_LOCAL_RAY", "").strip().lower() in (
    #     "1", "true", "yes", "local",
    # )
    # if _local:
    #     hostname = None
    # else:
    #     # geneva  "128.40.42.10"   london  "128.40.42.13"
    #     hostname = os.environ.get("K_SP_FF_RAY_HOST", "128.40.40.67").strip() or None
    # port = int(os.environ.get("K_SP_FF_RAY_PORT", "6379"))

    # query = { "nodes" : 14, "ILP Capacity" : { "$exists" : True }, "FF-kSP RWA" : { "$exists" : False }}
    # query = { "nodes" : 10}

    query = None

    # nsfnet_graph = nt.Database.read_topology_dataset_list("Topology_Data", "real", "ILP RWA assignment",
    #                                                       find_dic={"name": "NSFNET"}, node_data=True)
    graph_list = nt.Database.read_topology_dataset_list(db, collection, find_dic={"name": "NSFNET"},
                                                        node_data=True)
    #  CORONET_CONUS_Topology_nodes
    # CORONET_CONUS_Topology_nodes DTAG germany50 nobel-eu RegularDCI JPN25  NSFNET cost266

    # graph_list = nt.Database.read_topology_dataset_list(db, collection,
    #                                                 find_dic=query,
    #                                                 node_data=True, max_count=2)
    # max_count=100000

    matrix_one = np.ones((len(graph_list[0][0].nodes), len(graph_list[0][0].nodes)))
    np.fill_diagonal(matrix_one, 0)
    # diagona all zero
    T_c = (matrix_one / (len(graph_list[0][0].nodes) * (len(graph_list[0][0].nodes) - 1))).tolist()
    # normalize
    graph_list = [(graph, _id, T_c) for graph, _id in graph_list]

    result = nt.NetworkSimulator.parralel_heuristic_throughput(graph_list, collection=collection, db=db,
                                                               workers=len(graph_list),
                                                               route_function="kSP-FF",
                                                               e=100, k=50, m_step=200,
                                                               channel_bandwidth=channel_bandwidth,
                                                               max_count=10,
                                                               m_start=0,
                                                               port=port,
                                                               hostname=hostname, fibre_num=1,
                                                               band_selection=BAND_SELECTION, band_config=band_config,
                                                               span_length_km=span_length_km,
                                                               throughput=True)
    route_function = "kSP-FF"
    for graph, _id, T_c in graph_list:
        # Print topology information (Topology Design)
        print(f"\nTopology ID: {_id}")
        print(f"Number of Nodes: {len(graph.nodes)}")
        print(f"Number of Edges: {len(graph.edges)}")
        print(f"Average Degree: {2 * len(graph.edges) / len(graph.nodes):.2f}")

        # Read results from database
        try:
            results = list(nt.Database.read_data(db, collection,
                                                 find_dic={"_id": _id},
                                                 max_count=1))
            if len(results) > 0:
                result_data = results[0]

                # Print throughput results
                capacity_key = f"{route_function} Capacity"
                connections_key = f"{route_function}-connections"
                time_key = f"{route_function} time"

                if capacity_key in result_data:
                    throughput = result_data[capacity_key]
                    print(f"\nThroughput: {throughput:.2e} bps ({throughput / 1e12:.4f} Tbps)")
                print(f"route_function: {route_function} ")

                if connections_key in result_data:
                    connections = result_data[connections_key]
                    print(f"Max Connections M: {connections}")

                if time_key in result_data:
                    time_taken = result_data[time_key]
                    print(f"Computation Time: {time_taken:.2f} s")

                # # Print other parameters  kSP-FF RWA
                # if f"{route_function} RWA" in result_data:
                #     RWA_results = result_data[f"{route_function} RWA"]
                #     print(f"RWA: {RWA_results}")

                if f"{route_function} channel number" in result_data:
                    channels = result_data[f"{route_function} channel number"]
                    print(f"Number of Channels: {channels}")

                if f"{route_function} channel bandwidth" in result_data:
                    bandwidth = result_data[f"{route_function} channel bandwidth"]
                    print(f"Channel Bandwidth: {bandwidth / 1e9:.1f} GHz")
                # ============================================================
                # Visualize RWA results (following Figure 3.2 style)
                # ============================================================
                rwa_key_vis = f"{route_function} {str(BAND_SELECTION).strip().upper()} RWA"
                if rwa_key_vis in result_data:
                    rwa_result = result_data[rwa_key_vis]
                    print(f"\nprinting RWA heatmap...")
                    # import json
                    #
                    # file_path = 'CL_simulation_results.json'
                    #
                    # with open(file_path, 'w', encoding='utf-8') as f:
                    #     json.dump(rwa_result, f,  ensure_ascii=False)
                    #
                    # print(f"saved: {file_path}")
                    try:
                        from plot_rwa import plot_rwa_heatmap

                        plot_rwa_heatmap(
                            graph,
                            rwa_result,
                            title=f"NSFNET with {route_function} Routing",
                            save_path=f"rwa_{route_function}_nsfnet.png"
                        )
                        print(f"RWA saved: rwa_{route_function}_nsfnet.png")
                    except ImportError:
                        print("Warning: Unable to import plot_rwa module, skipping visualization")
                    except Exception as e:
                        print(f"Error during visualization: {e}")
            else:
                print(f"\nWarning: No result data found for ID {_id}")
        except Exception as e:
            print(f"\nError reading results: {e}")

    print("=" * 60 + "\n")
