import os
import socket
# Set Gurobi environment variables BEFORE importing NetworkToolkit (which may import Gurobi)
# Try TokenServer first (may not need local license file)
os.environ['GRB_TOKEN_SERVER'] = 'lic-gurobi.ucl.ac.uk'
# Try to find license file, but TokenServer might be enough
hostname = socket.gethostname().split('.')[0]
possible_license_paths = [
    '/home/uceebd1/gurobi/gurobi.lic',  # User's license file location
    f'/home/uceeatz/gurobi-licences/{hostname}/gurobi.lic',
    '/home/uceeatz/gurobi.lic',
]
# Only set GRB_LICENSE_FILE if file exists, otherwise rely on TokenServer
for path in possible_license_paths:
    if os.path.exists(path):
        os.environ['GRB_LICENSE_FILE'] = path
        break
else:
    # If no license file found, only use TokenServer
    # Don't set GRB_LICENSE_FILE to a non-existent path
    pass

import NetworkToolkit as nt
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from tqdm import tqdm
from copy import deepcopy

# ============================================================
# Band Selection Configuration
# ============================================================
# Available bands: "C", "C+L", "SCL", "SCLO"
BAND_SELECTION = "SCL"  # Change this to select different bands

# Band configurations (same as ilp_connections.py)
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
    "C+L": {
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

def calculate_throughput_with_span_length(graph, rwa, channel_bandwidth, span_length_km, 
                                          channels, fibre_num=1, band_config=None, launch_power_dbm=0):
    """
    Calculate throughput for a given span length
    
    :param graph: Network graph
    :param rwa: Routing and wavelength assignment
    :param channel_bandwidth: Channel bandwidth in Hz
    :param span_length_km: Fiber span length in km
    :param channels: Number of channels
    :param fibre_num: Number of fibres
    :param band_config: Band configuration dictionary
    :param launch_power_dbm: Launch power in dBm (default: 0 dBm = 1 mW)
    :return: Throughput in bps
    """
    # Get band configuration
    if band_config is None:
        B_o = 5e12
        wavelength_start_nm = 1530
        wavelength_width_nm = 40.049
        RefLambda_nm = 1550
        Cr = 0
    else:
        B_o = band_config['B_o_THz'] * 1e12
        wavelength_start_nm = band_config['wavelength_start_nm']
        wavelength_width_nm = band_config['wavelength_width_nm']
        RefLambda_nm = band_config['RefLambda_nm']
        Cr = band_config['Cr']
    
    # Set launch power
    power_watt = 10 ** (launch_power_dbm / 10) * 0.001
    
    # Clear any existing launch_powers
    for edge in graph.edges():
        if "launch_powers" in graph[edge[0]][edge[1]]:
            del graph[edge[0]][edge[1]]["launch_powers"]
        if "launch_powers" in graph[edge[1]][edge[0]]:
            del graph[edge[1]][edge[0]]["launch_powers"]
    
    launch_powers = {}
    for edge in graph.edges():
        launch_powers[edge] = {"launch_powers": [power_watt] * channels}
        launch_powers[(edge[1], edge[0])] = {"launch_powers": [power_watt] * channels}
    nx.set_edge_attributes(graph, launch_powers)
    
    # Create network with specified span length
    network = nt.Network.OpticalNetwork(
        graph,
        B_o=B_o,
        channel_bandwidth=channel_bandwidth,
        routing_func="FF-kSP",
        fibre_num=fibre_num,
        span_length_km=span_length_km
    )
    
    # Configure physical layer for selected band
    network.physical_layer.assign_physical_wavelengths(
        channels=network.channels,
        wavelength_start_nm=wavelength_start_nm,
        wavelength_width_nm=wavelength_width_nm
    )
    network.physical_layer.RefLambda = RefLambda_nm * 1e-9
    network.physical_layer.Cr = Cr
    
    # Add wavelengths and calculate throughput
    network.physical_layer.add_wavelengths_to_links(rwa)
    network.physical_layer.add_non_linear_NSR_to_links()
    throughput = network.physical_layer.get_lightpath_capacities_PLI(rwa)[0]
    
    return throughput

def calculate_rwa_ff_ksp(graph, T_c, channel_bandwidth, e=100, k=5, m_step=200, 
                         max_count=10, m_start=0, fibre_num=1, band_config=None, span_length_km=80):
    """
    Calculate RWA using FF-kSP method
    """
    # Get band configuration
    if band_config is None:
        B_o = 5e12
        wavelength_start_nm = 1530
        wavelength_width_nm = 40.049
        RefLambda_nm = 1550
        Cr = 0
    else:
        B_o = band_config['B_o_THz'] * 1e12
        wavelength_start_nm = band_config['wavelength_start_nm']
        wavelength_width_nm = band_config['wavelength_width_nm']
        RefLambda_nm = band_config['RefLambda_nm']
        Cr = band_config['Cr']
    
    network = nt.Network.OpticalNetwork(
        graph, 
        B_o=B_o,
        channel_bandwidth=channel_bandwidth,
        routing_func="FF-kSP",
        fibre_num=fibre_num,
        span_length_km=span_length_km
    )
    
    # Configure physical layer for selected band
    network.physical_layer.assign_physical_wavelengths(
        channels=network.channels,
        wavelength_start_nm=wavelength_start_nm,
        wavelength_width_nm=wavelength_width_nm
    )
    network.physical_layer.RefLambda = RefLambda_nm * 1e-9
    network.physical_layer.Cr = Cr
    
    rwa_assignment = False
    M = m_start
    demand_matrix_old = np.zeros((len(graph), len(graph)))
    rwa_active = None
    success = False
    
    for i in range(max_count):
        while rwa_assignment != True:
            M += m_step
            demand_matrix_new = np.ceil(np.array(T_c) * M)
            alternate_demand = demand_matrix_new - demand_matrix_old
            
            if not success:
                connection_pairs = nt.Tools.mat_to_pairs_list(alternate_demand)
                rwa_assignment = network.route(alternate_demand, e=e, k=k, connection_pairs=connection_pairs)
            elif alternate_demand.sum() > 0:
                connection_pairs = nt.Tools.mat_to_pairs_list(alternate_demand)
                rwa_assignment = network.route(
                    alternate_demand, 
                    e=e, k=k,
                    rwa_assignment_previous=rwa_assignment,
                    connection_pairs=connection_pairs
                )
            
            if rwa_assignment != True:
                rwa_active = rwa_assignment
                demand_matrix_old = demand_matrix_new
                success = True
        
        if int(M) > 1:
            M -= m_step
        
        demand_matrix = np.ceil(np.array(T_c) * M)
        rwa_assignment = rwa_active
        
        if int(M) == 1 and rwa_assignment == False:
            break
        elif int(M) == 1 and rwa_assignment == True:
            print("Can't route base demand")
            break
        
        m_step /= 2
        m_step = np.ceil(m_step)
    
    assert rwa_assignment != True
    assert rwa_assignment is not None
    
    return rwa_assignment

def calculate_rwa_ksp_ff(graph, T_c, channel_bandwidth, e=100, k=5, m_step=200,
                         max_count=10, m_start=0, fibre_num=1, band_config=None, span_length_km=80):
    """
    Calculate RWA using kSP-FF method
    """
    # Get band configuration
    if band_config is None:
        B_o = 5e12
        wavelength_start_nm = 1530
        wavelength_width_nm = 40.049
        RefLambda_nm = 1550
        Cr = 0
    else:
        B_o = band_config['B_o_THz'] * 1e12
        wavelength_start_nm = band_config['wavelength_start_nm']
        wavelength_width_nm = band_config['wavelength_width_nm']
        RefLambda_nm = band_config['RefLambda_nm']
        Cr = band_config['Cr']
    
    network = nt.Network.OpticalNetwork(
        graph,
        B_o=B_o,
        channel_bandwidth=channel_bandwidth,
        routing_func="kSP-FF",
        fibre_num=fibre_num,
        span_length_km=span_length_km
    )
    
    # Configure physical layer for selected band
    network.physical_layer.assign_physical_wavelengths(
        channels=network.channels,
        wavelength_start_nm=wavelength_start_nm,
        wavelength_width_nm=wavelength_width_nm
    )
    network.physical_layer.RefLambda = RefLambda_nm * 1e-9
    network.physical_layer.Cr = Cr
    
    rwa_assignment = False
    M = m_start
    demand_matrix_old = np.zeros((len(graph), len(graph)))
    rwa_active = None
    success = False
    
    for i in range(max_count):
        while rwa_assignment != True:
            M += m_step
            demand_matrix_new = np.ceil(np.array(T_c) * M)
            alternate_demand = demand_matrix_new - demand_matrix_old
            
            if not success:
                connection_pairs = nt.Tools.mat_to_pairs_list(alternate_demand)
                rwa_assignment = network.route(alternate_demand, e=e, k=k, connection_pairs=connection_pairs)
            elif alternate_demand.sum() > 0:
                connection_pairs = nt.Tools.mat_to_pairs_list(alternate_demand)
                rwa_assignment = network.route(
                    alternate_demand,
                    e=e, k=k,
                    rwa_assignment_previous=rwa_assignment,
                    connection_pairs=connection_pairs
                )
            
            if rwa_assignment != True:
                rwa_active = rwa_assignment
                demand_matrix_old = demand_matrix_new
                success = True
        
        if int(M) > 1:
            M -= m_step
        
        demand_matrix = np.ceil(np.array(T_c) * M)
        rwa_assignment = rwa_active
        
        if int(M) == 1 and rwa_assignment == False:
            break
        elif int(M) == 1 and rwa_assignment == True:
            print("Can't route base demand")
            break
        
        m_step /= 2
        m_step = np.ceil(m_step)
    
    assert rwa_assignment != True
    assert rwa_assignment is not None
    
    return rwa_assignment

def calculate_rwa_ilp(graph, T_c, channel_bandwidth, e=20, k=1, max_time=48*3600,
                       threads=1, fibre_num=1, blocking_rate=0, laptop_run=False, 
                       token_server=None, band_config=None, span_length_km=80):
    """
    Calculate RWA using ILP method (same as ilp_connections.py)
    """
    import os
    import socket
    import networkx as nx
    
    # Set Gurobi license
    if laptop_run:
        os.environ['GRB_LICENSE_FILE'] = "/Users/robin/Documents/network-code/gurobi-licence/gurobi.lic"
        node_file_dir = "/Users/robin/Documents/network-code/nodefiles"
    else:
        license_path = "/home/uceebd1/gurobi/gurobi.lic"
        if os.path.exists(license_path):
            os.environ['GRB_LICENSE_FILE'] = license_path
        else:
            os.environ['GRB_LICENSE_FILE'] = "/home/uceeatz/gurobi.lic"
        node_file_dir = "/scratch/datasets/gurobi/nodefiles"
    
    # Set TokenServer
    if token_server:
        os.environ['GRB_TOKEN_SERVER'] = token_server
    else:
        os.environ['GRB_TOKEN_SERVER'] = 'lic-gurobi.ucl.ac.uk'
    
    # Force update environment
    os.putenv('GRB_TOKEN_SERVER', os.environ['GRB_TOKEN_SERVER'])
    if 'GRB_LICENSE_FILE' in os.environ:
        os.putenv('GRB_LICENSE_FILE', os.environ['GRB_LICENSE_FILE'])
    
    # Prepare graph
    graph = nx.relabel.convert_node_labels_to_integers(graph, first_label=1)
    assert type(graph) == nx.classes.graph.Graph
    assert nx.is_connected(graph) == True
    
    # Get band configuration
    if band_config is None:
        B_o = 5e12
        wavelength_start_nm = 1530
        wavelength_width_nm = 40.049
        RefLambda_nm = 1550
        Cr = 0
    else:
        B_o = band_config['B_o_THz'] * 1e12
        wavelength_start_nm = band_config['wavelength_start_nm']
        wavelength_width_nm = band_config['wavelength_width_nm']
        RefLambda_nm = band_config['RefLambda_nm']
        Cr = band_config['Cr']
    
    # Create network (same as ILP_connections line 197)
    network = nt.Network.OpticalNetwork(graph, B_o=B_o, channel_bandwidth=channel_bandwidth, fibre_num=1, span_length_km=span_length_km)
    
    # Set routing_channels (same as ILP_connections line 198)
    network.routing_channels = fibre_num * network.routing_channels
    
    # Configure physical layer for selected band
    network.physical_layer.assign_physical_wavelengths(
        channels=network.channels,
        wavelength_start_nm=wavelength_start_nm,
        wavelength_width_nm=wavelength_width_nm
    )
    network.physical_layer.RefLambda = RefLambda_nm * 1e-9
    network.physical_layer.Cr = Cr
    
    # Call maximise_connection_demand
    data = network.rwa.maximise_connection_demand(
        T_c=T_c,
        max_time=max_time,
        e=e,
        k=k,
        _id=0,
        threads=threads,
        node_file_dir=node_file_dir,
        node_file_start=0.01,
        emphasis=0,
        weighted=None,
        max_solutions=100,
        blocking_rate=blocking_rate,
        fibre_num=fibre_num,
        channels=network.channels
    )
    
    if data is None:
        raise ValueError("ILP solver returned None. Check Gurobi license and solver status.")
    
    return data["rwa"]

def sweep_distance(graph, T_c, channel_bandwidth, channels, route_function,
                   distance_range=(10, 80), step=10, fibre_num=1, 
                   band_config=None, launch_power_dbm=0,
                   e_heuristic=100, k_heuristic=1, m_step=200, max_count=10, m_start=0,
                   e_ilp=100, k_ilp=1, max_time_ilp=48*3600, threads_ilp=1, 
                   blocking_rate=0, laptop_run=False, token_server=None):
    """
    Sweep fiber span length and calculate throughput for each value
    
    :param graph: Network graph
    :param T_c: Traffic matrix
    :param channel_bandwidth: Channel bandwidth in Hz
    :param channels: Number of channels
    :param route_function: Route function name ("FF-kSP", "kSP-FF", or "ILP-connections")
    :param distance_range: Tuple of (min, max) span length in km
    :param step: Step size in km
    :param fibre_num: Number of fibres
    :param band_config: Band configuration dictionary
    :param launch_power_dbm: Launch power in dBm (default: 0 dBm)
    :param e_heuristic: Path length parameter for heuristic methods
    :param k_heuristic: Number of k-shortest paths for heuristic methods
    :param m_step: Step size for binary search (heuristic)
    :param max_count: Maximum iterations (heuristic)
    :param m_start: Starting M value (heuristic)
    :param e_ilp: Path length parameter for ILP
    :param k_ilp: Number of k-shortest paths for ILP
    :param max_time_ilp: Maximum time for ILP solver
    :param threads_ilp: Number of threads for ILP solver
    :param blocking_rate: Blocking rate for ILP
    :param laptop_run: Whether running on laptop
    :param token_server: TokenServer address
    :return: Dictionary with span lengths and throughputs
    """
    span_lengths_km = np.arange(distance_range[0], distance_range[1] + step, step)
    throughputs = []
    rwa_list = []
    
    print(f"Sweeping fiber span length from {distance_range[0]} to {distance_range[1]} km for {route_function}")
    print(f"Step size: {step} km")
    print(f"Launch power: {launch_power_dbm} dBm")
    
    for span_length_km in tqdm(span_lengths_km, desc=f"Calculating throughput ({route_function})"):
        # Create a deep copy of the graph for each iteration
        graph_copy = deepcopy(graph)
        
        try:
            # Calculate RWA for this span length
            if route_function == "FF-kSP":
                rwa = calculate_rwa_ff_ksp(
                    graph_copy, T_c, channel_bandwidth,
                    e=e_heuristic, k=k_heuristic, m_step=m_step,
                    max_count=max_count, m_start=m_start, fibre_num=fibre_num,
                    band_config=band_config, span_length_km=span_length_km
                )
            elif route_function == "kSP-FF":
                rwa = calculate_rwa_ksp_ff(
                    graph_copy, T_c, channel_bandwidth,
                    e=e_heuristic, k=k_heuristic, m_step=m_step,
                    max_count=max_count, m_start=m_start, fibre_num=fibre_num,
                    band_config=band_config, span_length_km=span_length_km
                )
            elif route_function == "ILP-connections":
                rwa = calculate_rwa_ilp(
                    graph_copy, T_c, channel_bandwidth,
                    e=e_ilp, k=k_ilp, max_time=max_time_ilp,
                    threads=threads_ilp, fibre_num=fibre_num, blocking_rate=blocking_rate,
                    laptop_run=laptop_run, token_server=token_server,
                    band_config=band_config, span_length_km=span_length_km
                )
            else:
                raise ValueError(f"Unknown route function: {route_function}")
            
            rwa_list.append(rwa)
            
            # Calculate throughput with this span length
            graph_copy2 = deepcopy(graph)
            throughput = calculate_throughput_with_span_length(
                graph_copy2, rwa, channel_bandwidth, span_length_km, channels, fibre_num,
                band_config=band_config, launch_power_dbm=launch_power_dbm
            )
            throughputs.append(throughput)
            
        except Exception as e:
            print(f"Error at {span_length_km} km: {e}")
            import traceback
            traceback.print_exc()
            throughputs.append(np.nan)
            rwa_list.append(None)
    
    return {
        'span_length_km': span_lengths_km.tolist(),
        'throughput_bps': throughputs,
        'throughput_tbps': [t / 1e12 if not np.isnan(t) else np.nan for t in throughputs],
        'route_function': route_function,
        'rwa_list': rwa_list
    }

def plot_distance_vs_throughput(results_list, save_path=None):
    """
    Plot throughput vs fiber span length for multiple route functions
    
    :param results_list: List of dictionaries from sweep_distance (one for each route function)
    :param save_path: Path to save the plot
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    colors = ['blue', 'green', 'red']
    markers = ['o', 's', '^']
    linestyles = ['-', '--', '-.']
    
    for idx, results in enumerate(results_list):
        route_function = results.get('route_function', f'Method {idx+1}')
        
        # Convert to numpy arrays for boolean indexing
        span_length_km = np.array(results['span_length_km'])
        throughput_tbps = np.array(results['throughput_tbps'])
        
        # Plot
        valid_mask = ~np.isnan(throughput_tbps)
        ax.plot(span_length_km[valid_mask],
                throughput_tbps[valid_mask],
                'o-', linewidth=2, markersize=8,
                color=colors[idx % len(colors)],
                marker=markers[idx % len(markers)],
                linestyle=linestyles[idx % len(linestyles)],
                label=route_function)
    
    # Plot settings
    ax.set_xlabel('Fiber Span Length [km]', fontsize=12)
    ax.set_ylabel('Throughput [Tbps]', fontsize=12)
    ax.set_title('Throughput vs Fiber Span Length', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

if __name__ == "__main__":
    import os
    
    # Set Gurobi environment variables
    laptop_run = False
    token_server = "lic-gurobi.ucl.ac.uk"
    
    if laptop_run:
        os.environ['GRB_LICENSE_FILE'] = "/Users/robin/Documents/network-code/gurobi-licence/gurobi.lic"
    else:
        license_path = "/home/uceebd1/gurobi/gurobi.lic"
        if os.path.exists(license_path):
            os.environ['GRB_LICENSE_FILE'] = license_path
        else:
            os.environ['GRB_LICENSE_FILE'] = "/home/uceeatz/gurobi.lic"
    
    if token_server:
        os.environ['GRB_TOKEN_SERVER'] = token_server
    
    # Configuration
    collection = "topology-paper"
    db = "Topology_Data"
    
    # Network parameters
    fibre_num = 1
    
    # FF-kSP and kSP-FF parameters
    e_heuristic = 100
    k_heuristic = 1
    m_step = 200
    max_count = 10
    m_start = 0
    
    # ILP parameters
    e_ilp = 100
    k_ilp = 1
    max_time_ilp = 48 * 3600  # 48 hours
    threads_ilp = 1
    blocking_rate = 0
    
    # Distance scaling parameters
    distance_range = (10, 80)  # 10 to 80 km
    step = 10  # step size in km
    launch_power_dbm = 0  # Fixed launch power: 0 dBm (1 mW)
    
    # Route functions to test
    route_functions = ["FF-kSP", "kSP-FF", "ILP-connections"]
    
    # Read graph from database
    print("Loading graph from database...")
    graph_list = nt.Database.read_topology_dataset_list(
        db, collection,
        find_dic={"name": "NSFNET"},
        node_data=True,
        max_count=1
    )
    
    if len(graph_list) == 0:
        print("No graph found!")
        exit(1)
    
    graph, _id = graph_list[0]
    
    # Create uniform traffic matrix
    matrix_one = np.ones((len(graph.nodes), len(graph.nodes)))
    np.fill_diagonal(matrix_one, 0)
    T_c = (matrix_one / (len(graph.nodes) * (len(graph.nodes) - 1))).tolist()
    
    print(f"\nTraffic Matrix: Uniform distribution")
    print(f"Channel Bandwidth: {channel_bandwidth/1e9:.1f} GHz")
    print(f"Fibre Number: {fibre_num}")
    print(f"Launch Power: {launch_power_dbm} dBm (fixed)")
    print(f"Distance Range: {distance_range[0]} - {distance_range[1]} km, step: {step} km")
    
    # Get number of channels
    temp_network = nt.Network.OpticalNetwork(
        graph,
        B_o=B_o,
        channel_bandwidth=channel_bandwidth,
        routing_func="FF-kSP",
        fibre_num=fibre_num,
        span_length_km=80  # Use default for channel count
    )
    channels = temp_network.channels
    print(f"Number of Channels: {channels}")
    
    # Store results for all route functions
    all_results = []
    
    # Process each route function
    for route_function in route_functions:
        print(f"\n{'='*60}")
        print(f"Processing {route_function}")
        print(f"{'='*60}")
        
        # Sweep distance and calculate throughput
        print(f"\nSweeping fiber span length for {route_function}...")
        results = sweep_distance(
            graph, T_c, channel_bandwidth, channels, route_function,
            distance_range=distance_range,
            step=step,
            fibre_num=fibre_num,
            band_config=band_config,
            launch_power_dbm=launch_power_dbm,
            e_heuristic=e_heuristic, k_heuristic=k_heuristic, 
            m_step=m_step, max_count=max_count, m_start=m_start,
            e_ilp=e_ilp, k_ilp=k_ilp, max_time_ilp=max_time_ilp,
            threads_ilp=threads_ilp, blocking_rate=blocking_rate,
            laptop_run=laptop_run, token_server=token_server
        )
        
        all_results.append(results)
        
        # Print results summary
        print(f"\n{route_function} Results Summary:")
        print("-" * 60)
        for i, (span_km, thr_tbps) in enumerate(zip(
                results['span_length_km'],
                results['throughput_tbps']
        )):
            if not np.isnan(thr_tbps):
                print(f"Span Length: {span_km:6.1f} km -> Throughput: {thr_tbps:8.4f} Tbps")
        
        # Find optimal span length (maximum throughput)
        valid_throughputs = np.array(results['throughput_tbps'])
        valid_mask = ~np.isnan(valid_throughputs)
        if np.any(valid_mask):
            max_idx = np.nanargmax(valid_throughputs)
            optimal_span_km = results['span_length_km'][max_idx]
            optimal_thr = results['throughput_tbps'][max_idx]
            print(f"\nOptimal Span Length: {optimal_span_km:.1f} km")
            print(f"Maximum Throughput: {optimal_thr:.4f} Tbps")
    
    # Plot all results together
    if len(all_results) > 0:
        plot_distance_vs_throughput(all_results, save_path="distance_scaling_comparison.png")
        
        # Save results to file
        import pandas as pd
        
        # Combine all results into a single DataFrame
        combined_data = []
        for results in all_results:
            route_func = results['route_function']
            for i in range(len(results['span_length_km'])):
                combined_data.append({
                    'route_function': route_func,
                    'span_length_km': results['span_length_km'][i],
                    'throughput_bps': results['throughput_bps'][i],
                    'throughput_tbps': results['throughput_tbps'][i]
                })
        
        df = pd.DataFrame(combined_data)
        df.to_csv("distance_scaling_results.csv", index=False)
        print("\nResults saved to distance_scaling_results.csv")
    else:
        print("No results to plot or save.")

