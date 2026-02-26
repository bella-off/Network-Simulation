import NetworkToolkit as nt
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

nt.Network
if __name__ == "__main__":
    # ============================================================
    # Band Selection Configuration
    # ============================================================
    # Available bands: "C", "C+L", "SCL", "SCLO"
    BAND_SELECTION = "C"  # Change this to select different bands
    span_length_km = 80  # Fiber span length in km (default: 80 km, changed to 10 km)

# Band configurations
    BAND_CONFIGS = {
        "C": {
            "name": "C band",
            "wavelength_start_nm": 1530,      # Starting wavelength in nm
            "wavelength_width_nm": 40.049,   # Band width in nm
            "B_o_THz": 5.0,                  # Total bandwidth in THz
            "RefLambda_nm": 1550,            # Reference wavelength in nm
            "Cr": 0,                         # ISRS coefficient (0 for C band)
            "channel_bandwidth_GHz": 50,     # Channel bandwidth in GHz (standard for C band)
            "description": "C band: 1530-1570 nm (40 nm)"
        },
        "C+L": {
            "name": "C+L band",
            "wavelength_start_nm": 1530,
            "wavelength_width_nm": 95,       # C+L band width
            "B_o_THz": 11.8,                 # C+L band total bandwidth
            "RefLambda_nm": 1577.5,          # Center of C+L band
            "Cr": 0.028 / 1e3 / 1e12,       # Enable ISRS for C+L
            "channel_bandwidth_GHz": 50,     # Channel bandwidth in GHz (standard for C+L band)
            "description": "C+L band: 1530-1625 nm (95 nm)"
        },
        "SCL": {
            "name": "SCL band (Super C+L)",
            "wavelength_start_nm": 1460,
            "wavelength_width_nm": 165,      # SCL band width
            "B_o_THz": 20.86,                # SCL band total bandwidth
            "RefLambda_nm": 1542.5,          # Center of SCL band
            "Cr": 0.028 / 1e3 / 1e12,       # Enable ISRS for SCL
            "channel_bandwidth_GHz": 50,     # Channel bandwidth in GHz (standard for SCL band)
            "description": "SCL band: 1460-1625 nm (165 nm)"
        },
        "SCLO": {
            "name": "SCLO band (Super C+L+O)",
            "wavelength_start_nm": 1260,     # Extended to O band
            "wavelength_width_nm": 365,      # SCLO band width (1260-1625 nm)
            "B_o_THz": 46.0,                 # SCLO band total bandwidth
            "RefLambda_nm": 1442.5,          # Center of SCLO band
            "Cr": 0.028 / 1e3 / 1e12,       # Enable ISRS for SCLO
            "channel_bandwidth_GHz": 50,     # Channel bandwidth in GHz (standard for SCLO band)
            "description": "SCLO band: 1260-1625 nm (365 nm)"
        }
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
    num_channels = int(np.floor(B_o / channel_bandwidth))
    
    print("="*60)
    print(f"Selected Band: {band_config['name']}")
    print(f"Description: {band_config['description']}")
    print(f"Wavelength Range: {band_config['wavelength_start_nm']:.1f} - "
          f"{band_config['wavelength_start_nm'] + band_config['wavelength_width_nm']:.1f} nm")
    print(f"Total Bandwidth: {band_config['B_o_THz']:.2f} THz")
    print(f"Channel Bandwidth: {band_config['channel_bandwidth_GHz']:.1f} GHz")
    print(f"Number of Channels: {num_channels}")
    print(f"Reference Wavelength: {band_config['RefLambda_nm']:.1f} nm")
    print("="*60)
    
    # ============================================================
    # Fiber Span Length Configuration
    # ============================================================
    print(f"Fiber Span Length: {span_length_km} km")
    print("="*60)
    
    hostname = "128.40.42.13"
    # hostname = None   # None to use the Ray
    port = 6379
    collection = "topology-paper"
    db = "Topology_Data"
    # port = 7112
    # query = { "nodes" : 14, "ILP Capacity" : { "$exists" : True }, "ILP-connections" : { "$exists" : False }}
    # graph_list = nt.Database.read_topology_dataset_list("Topology_Data", "topology-paper",
    #                                                 find_dic=query,
    #                                                 node_data=False, max_count=10000)
    #

    graph_list = nt.Database.read_topology_dataset_list(db, collection, find_dic={"name": "NSFNET"},
                                                        node_data=True)

    # ILP_throughput doesn't need T_c, it uses uniform bandwidth demand
    # graph_list should be (graph, _id) format, not (graph, _id, T_c)
    graph_list = [(graph, _id) for graph, _id in graph_list]

    # Run ILP throughput optimization
    # Note: ILP_throughput uses channel_bandwidth (not bandwidth), and doesn't accept insert, throughput, blocking_rate
    nt.NetworkSimulator.parralel_ILP_throughput(graph_list, db="Topology_Data", collection="topology-paper", 
                                 max_time=48*3600, workers=len(graph_list),
                                 threads=1, fibre_num=1, hostname=hostname, port=port,
                                 bandwidth=channel_bandwidth,  # This will be passed as channel_bandwidth to ILP_throughput
                                 k=1, e=0,
                                 node_file_start=0.01,
                                 capacity_constraint=True)

    # k=20



    # ============================================================
    # Print results
    # ============================================================
    print("\n" + "="*60)
    print("Results Summary")
    print("="*60)

    route_function = "ILP-throughput"

    for graph, _id in graph_list:
        # Print topology information
        print(f"\nTopology ID: {_id}")
        print(f"Number of Nodes: {len(graph.nodes)}")
        print(f"Number of Edges: {len(graph.edges)}")
        print(f"Average Degree: {2*len(graph.edges)/len(graph.nodes):.2f}")

        # Read results from database
        try:
            results = list(nt.Database.read_data(db, collection,
                                                 find_dic={"_id": _id},
                                                 max_count=1))
            if len(results) > 0:
                result_data = results[0]

                # Print throughput results
                capacity_key = f"{route_function} Capacity"
                connections_key = f"{route_function}"
                time_key = f"{route_function} time"

                if capacity_key in result_data:
                    throughput = result_data[capacity_key]
                    print(f"\nThroughput: {throughput:.2e} bps ({throughput/1e12:.4f} Tbps)")
                print(f"route_function: {route_function} ")

                if connections_key in result_data:
                    connections = result_data[connections_key]
                    print(f"Max Connections M: {connections}")

                if time_key in result_data:
                    time_taken = result_data[time_key]
                    print(f"Computation Time: {time_taken:.2f} seconds")

                # Print other parameters
                if f"{route_function} channels" in result_data:
                    channels = result_data[f"{route_function} channels"]
                    print(f"Number of Channels: {channels}")

                if f"{route_function} channel bandwidth" in result_data:
                    bandwidth = result_data[f"{route_function} channel bandwidth"]
                    print(f"Channel Bandwidth: {bandwidth/1e9:.1f} GHz")

                if f"{route_function} routing channels" in result_data:
                    routing_channels = result_data[f"{route_function} routing channels"]
                    print(f"Routing Channels: {routing_channels}")

                if f"{route_function} gap" in result_data:
                    gap = result_data[f"{route_function} gap"]
                    print(f"Gap: {gap:.4f}")

                if f"{route_function} status" in result_data:
                    status = result_data[f"{route_function} status"]
                    print(f"Status: {status}")

                # ============================================================
                # Visualize RWA results (following Figure 3.2 style)
                # ============================================================
                if f"{route_function} RWA" in result_data:
                    rwa_result = result_data[f"{route_function} RWA"]
                    print(f"\nprinting RWA heatmap...")

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

    print("="*60 + "\n")

print("=" * 60 + "\n")
