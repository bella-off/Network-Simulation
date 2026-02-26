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
BAND_SELECTION = "C"  # Change this to select different bands

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

def calculate_throughput_with_launch_power(graph, rwa, channel_bandwidth,
                                           launch_power_dbm, channels, fibre_num=1,
                                           band_config=None, span_length_km=80):
    """
    Calculate throughput for a given launch power
    This function follows the exact same logic as heuristic_throughput in NetworkSimulator.py

    :param graph: Network graph
    :param rwa: Routing and wavelength assignment
    :param channel_bandwidth: Channel bandwidth in Hz
    :param launch_power_dbm: Launch power in dBm
    :param channels: Number of channels (for reference, but network.channels will be used)
    :param fibre_num: Number of fibres
    :return: Throughput in bps
    """
    # IMPORTANT: We need to set launch power BEFORE creating the network
    # because get_non_linear_coefficient reads launch_powers from graph edges
    # First, we need to know the number of channels, which depends on the network
    # So we create a temporary network to get channels, then set launch power, then recreate network

    # Get band configuration
    if band_config is None:
        # Default C band configuration
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
    
    # Create temporary network to get channels count
    temp_network = nt.Network.OpticalNetwork(
        graph,
        B_o=B_o,
        channel_bandwidth=channel_bandwidth,
        routing_func="FF-kSP",
        fibre_num=fibre_num,
        span_length_km=span_length_km
    )
    num_channels = temp_network.channels
    
    # Configure physical layer for selected band
    temp_network.physical_layer.assign_physical_wavelengths(
        channels=temp_network.channels,
        wavelength_start_nm=wavelength_start_nm,
        wavelength_width_nm=wavelength_width_nm
    )
    temp_network.physical_layer.RefLambda = RefLambda_nm * 1e-9
    temp_network.physical_layer.Cr = Cr

    # Set launch power using the same method as heuristic_throughput
    # In heuristic_throughput line 1652: network.physical_layer.add_uniform_launch_power_to_links(network.channels)
    # But we need to modify it to use custom launch power instead of default 1 mW (0 dBm)
    # Default: (10 ** (-3)) = 0.001 W = 1 mW = 0 dBm
    power_watt = 10 ** (launch_power_dbm / 10) * 0.001
    power_mw = power_watt * 1000  # For debugging

    # Clear any existing launch_powers to ensure clean state
    for edge in graph.edges():
        if "launch_powers" in graph[edge[0]][edge[1]]:
            del graph[edge[0]][edge[1]]["launch_powers"]
        if "launch_powers" in graph[edge[1]][edge[0]]:
            del graph[edge[1]][edge[0]]["launch_powers"]

    launch_powers = {}
    # Use the channels count from network
    for edge in graph.edges():
        launch_powers[edge] = {"launch_powers": [power_watt] * num_channels}
        launch_powers[(edge[1], edge[0])] = {"launch_powers": [power_watt] * num_channels}
    nx.set_edge_attributes(graph, launch_powers)

    # Debug: Verify launch power was set correctly (only for first edge, first channel)
    if hasattr(calculate_throughput_with_launch_power, '_debug_count'):
        calculate_throughput_with_launch_power._debug_count += 1
    else:
        calculate_throughput_with_launch_power._debug_count = 1
    if calculate_throughput_with_launch_power._debug_count <= 3:
        first_edge = list(graph.edges())[0]
        actual_power = graph[first_edge[0]][first_edge[1]]["launch_powers"][0]
        print(f"Debug: Launch power set to {actual_power*1000:.3f} mW (expected {power_mw:.3f} mW) for {launch_power_dbm} dBm")

    # Now create the actual network with launch powers already set
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

    # Add wavelengths - exactly as in heuristic_throughput line 1671
    network.physical_layer.add_wavelengths_to_links(rwa)

    # Add non-linear NSR - exactly as in heuristic_throughput line 1672
    # IMPORTANT: In original code, NO parameters are passed, so it uses defaults:
    # channels_full=156 (default), channel_bandwidth=self.channel_bandwidth (which is network.channel_bandwidth)
    # This matches the exact behavior of heuristic_throughput
    network.physical_layer.add_non_linear_NSR_to_links()

    # Calculate throughput - exactly as in heuristic_throughput line 1673
    throughput = network.physical_layer.get_lightpath_capacities_PLI(rwa)[0]
    return throughput

def calculate_rwa_ff_ksp(graph, T_c, channel_bandwidth, e=100, k=5, m_step=200, 
                         max_count=10, m_start=0, fibre_num=1, band_config=None, span_length_km=80):
    """
    Calculate RWA using FF-kSP method (same as ff_k_sp.py)
    
    :param graph: Network graph
    :param T_c: Traffic matrix
    :param channel_bandwidth: Channel bandwidth in Hz
    :param e: Path length parameter
    :param k: Number of k-shortest paths
    :param m_step: Step size for binary search
    :param max_count: Maximum iterations
    :param m_start: Starting M value
    :param fibre_num: Number of fibres
    :return: RWA assignment dictionary
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
                # Convert demand matrix to connection pairs list
                connection_pairs = nt.Tools.mat_to_pairs_list(alternate_demand)
                rwa_assignment = network.route(alternate_demand, e=e, k=k, connection_pairs=connection_pairs)
            elif alternate_demand.sum() > 0:
                # Convert demand matrix to connection pairs list
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
    Calculate RWA using kSP-FF method (same as k_sp_ff.py)
    
    :param graph: Network graph
    :param T_c: Traffic matrix
    :param channel_bandwidth: Channel bandwidth in Hz
    :param e: Path length parameter
    :param k: Number of k-shortest paths
    :param m_step: Step size for binary search
    :param max_count: Maximum iterations
    :param m_start: Starting M value
    :param fibre_num: Number of fibres
    :return: RWA assignment dictionary
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
                # Convert demand matrix to connection pairs list
                connection_pairs = nt.Tools.mat_to_pairs_list(alternate_demand)
                rwa_assignment = network.route(alternate_demand, e=e, k=k, connection_pairs=connection_pairs)
            elif alternate_demand.sum() > 0:
                # Convert demand matrix to connection pairs list
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
    Uses the same approach as NetworkSimulator.ILP_connections
    
    :param graph: Network graph
    :param T_c: Traffic matrix
    :param channel_bandwidth: Channel bandwidth in Hz
    :param e: Path length parameter
    :param k: Number of k-shortest paths
    :param max_time: Maximum time for ILP solver
    :param threads: Number of threads for solver
    :param fibre_num: Number of fibres
    :param blocking_rate: Blocking rate
    :param laptop_run: Whether running on laptop (affects license path)
    :param token_server: TokenServer address (e.g., "lic-gurobi.ucl.ac.uk")
    :return: RWA assignment dictionary
    """
    import os
    import socket
    import networkx as nx
    
    # Set Gurobi license (EXACTLY same as NetworkSimulator.ILP_connections line 807-816)
    # MUST set BEFORE creating network or calling maximise_connection_demand
    if laptop_run:
        os.environ['GRB_LICENSE_FILE'] = "/Users/robin/Documents/network-code/gurobi-licence/gurobi.lic"
        node_file_dir = "/Users/robin/Documents/network-code/nodefiles"
    else:
        # Use user's license file location
        license_path = "/home/uceebd1/gurobi/gurobi.lic"
        if os.path.exists(license_path):
            os.environ['GRB_LICENSE_FILE'] = license_path
        else:
            # Fallback to default (even if file doesn't exist, TokenServer might work)
            os.environ['GRB_LICENSE_FILE'] = "/home/uceeatz/gurobi.lic"
        node_file_dir = "/scratch/datasets/gurobi/nodefiles"
    
    # Set TokenServer if provided (must be set before mip is imported)
    if token_server:
        os.environ['GRB_TOKEN_SERVER'] = token_server
    else:
        # Ensure TokenServer is set even if not provided
        os.environ['GRB_TOKEN_SERVER'] = 'lic-gurobi.ucl.ac.uk'
    
    # Force update environment (some systems cache env vars)
    os.putenv('GRB_TOKEN_SERVER', os.environ['GRB_TOKEN_SERVER'])
    if 'GRB_LICENSE_FILE' in os.environ:
        os.putenv('GRB_LICENSE_FILE', os.environ['GRB_LICENSE_FILE'])
    
    # Prepare graph (same as ILP_connections)
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
    # Note: ILP_connections uses fibre_num=1 here, then adjusts routing_channels later
    network = nt.Network.OpticalNetwork(graph, B_o=B_o, channel_bandwidth=channel_bandwidth, fibre_num=1, span_length_km=span_length_km)
    
    # Set routing_channels (same as ILP_connections line 198)
    network.routing_channels = fibre_num * network.routing_channels
    
    # Configure physical layer for selected band (same as ILP_connections line 202-210)
    network.physical_layer.assign_physical_wavelengths(
        channels=network.channels,
        wavelength_start_nm=wavelength_start_nm,
        wavelength_width_nm=wavelength_width_nm
    )
    # Set RefLambda and Cr for NLI calculation
    network.physical_layer.RefLambda = RefLambda_nm * 1e-9  # Convert nm to m
    network.physical_layer.Cr = Cr
    
    # Call maximise_connection_demand (same as ILP_connections line 215-223)
    # Environment variables must be set before this call
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
        fibre_num=fibre_num,  # ← 添加这个参数（关键！）
        channels=network.channels  # ← 添加这个参数（最关键！）
    )
    
    if data is None:
        raise ValueError("ILP solver returned None. Check Gurobi license and solver status.")
    
    return data["rwa"]

def sweep_launch_power(graph, rwa, channel_bandwidth, channels,
                       launch_power_range=(-6, 6), step=4, fibre_num=1, route_function="FF-kSP",
                       band_config=None, span_length_km=80):
    """
    Sweep launch power and calculate throughput for each value

    :param graph: Network graph
    :param rwa: Routing and wavelength assignment
    :param channel_bandwidth: Channel bandwidth in Hz
    :param channels: Number of channels
    :param launch_power_range: Tuple of (min, max) launch power in dBm
    :param step: Step size in dBm
    :param fibre_num: Number of fibres
    :param route_function: Route function name for labeling ("FF-kSP", "kSP-FF", or "ILP-connections")
    :return: Dictionary with launch powers and throughputs
    """
    launch_powers_dbm = np.arange(launch_power_range[0],
                                  launch_power_range[1] + step,
                                  step)
    throughputs = []

    print(f"Sweeping launch power from {launch_power_range[0]} to {launch_power_range[1]} dBm for {route_function}")

    for lp_dbm in tqdm(launch_powers_dbm, desc=f"Calculating throughput ({route_function})"):
        # Create a deep copy of the graph for each iteration to avoid state contamination
        graph_copy = deepcopy(graph)

        try:
            throughput = calculate_throughput_with_launch_power(
                graph_copy, rwa, channel_bandwidth, lp_dbm, channels, fibre_num,
                band_config=band_config, span_length_km=span_length_km
            )
            throughputs.append(throughput)
        except Exception as e:
            print(f"Error at {lp_dbm} dBm: {e}")
            throughputs.append(np.nan)

    return {
        'launch_power_dbm': launch_powers_dbm,
        'launch_power_mw': [10 ** (lp / 10) for lp in launch_powers_dbm],  # Convert to mW
        'throughput_bps': throughputs,
        'throughput_tbps': [t / 1e12 if not np.isnan(t) else np.nan for t in throughputs],
        'route_function': route_function
    }

def plot_launch_power_vs_throughput(results_list, save_path=None):
    """
    Plot throughput vs launch power for multiple route functions

    :param results_list: List of dictionaries from sweep_launch_power (one for each route function)
    :param save_path: Path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    colors = ['blue', 'green', 'red']
    markers = ['o', 's', '^']

    for idx, results in enumerate(results_list):
        route_function = results.get('route_function', f'Method {idx+1}')

        # Convert to numpy arrays for boolean indexing
        launch_power_dbm = np.array(results['launch_power_dbm'])
        launch_power_mw = np.array(results['launch_power_mw'])
        throughput_tbps = np.array(results['throughput_tbps'])

        # Plot 1: Throughput vs Launch Power (dBm)
        valid_mask = ~np.isnan(throughput_tbps)
        ax1.plot(launch_power_dbm[valid_mask],
                 throughput_tbps[valid_mask],
                 'o-', linewidth=2, markersize=6,
                 color=colors[idx % len(colors)],
                 marker=markers[idx % len(markers)],
                 label=route_function)

        # Plot 2: Throughput vs Launch Power (mW)
        ax2.plot(launch_power_mw[valid_mask],
                 throughput_tbps[valid_mask],
                 'o-', linewidth=2, markersize=6,
                 color=colors[idx % len(colors)],
                 marker=markers[idx % len(markers)],
                 label=route_function)

    # Plot 1 settings
    ax1.set_xlabel('Launch Power [dBm]', fontsize=12)
    ax1.set_ylabel('Throughput [Tbps]', fontsize=12)
    ax1.set_title('Throughput vs Launch Power', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Plot 2 settings
    ax2.set_xlabel('Launch Power [mW]', fontsize=12)
    ax2.set_ylabel('Throughput [Tbps]', fontsize=12)
    ax2.set_title('Throughput vs Launch Power', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')
    ax2.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

if __name__ == "__main__":
    import os
    
    # Set Gurobi environment variables BEFORE any imports that might use Gurobi
    # This must be done at the module level, before any Gurobi-related imports
    laptop_run = False
    token_server = "lic-gurobi.ucl.ac.uk"
    
    if laptop_run:
        os.environ['GRB_LICENSE_FILE'] = "/Users/robin/Documents/network-code/gurobi-licence/gurobi.lic"
    else:
        # Use user's license file location
        license_path = "/home/uceebd1/gurobi/gurobi.lic"
        if os.path.exists(license_path):
            os.environ['GRB_LICENSE_FILE'] = license_path
        else:
            # Fallback to default
            os.environ['GRB_LICENSE_FILE'] = "/home/uceeatz/gurobi.lic"
    
    if token_server:
        os.environ['GRB_TOKEN_SERVER'] = token_server
    
    # Configuration - Set parameters yourself
    collection = "topology-paper"
    db = "Topology_Data"
    
    # Network parameters (calculated from band selection)
    # channel_bandwidth is already calculated from band_config above
    fibre_num = 1
    
    # Fiber span length configuration
    span_length_km = 80  # Fiber span length in km (default: 80 km, changed to 10 km)
    
    # FF-kSP and kSP-FF parameters (same as ff_k_sp.py and k_sp_ff.py)
    e_heuristic = 100
    k_heuristic = 5
    m_step = 200
    max_count = 10
    m_start = 0
    
    # ILP parameters (same as ilp_connections.py)
    e_ilp = 100
    k_ilp = 1
    max_time_ilp = 48 * 3600  # 48 hours
    threads_ilp = 1
    blocking_rate = 0
    
    # Launch power sweep parameters
    launch_power_range = (-20, 0)  # -6 to 6 dBm
    step = 5  # step size in dBm
    
    # Route functions to test
    route_functions = ["FF-kSP", "kSP-FF"]
    # route_functions = ["FF-kSP", "kSP-FF", "ILP-connections"]

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
    graph = nx.relabel.convert_node_labels_to_integers(graph, first_label=1)
    
    print(f"Graph: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
    
    # Create uniform traffic matrix (same as all three reference files)
    matrix_one = np.ones((len(graph.nodes), len(graph.nodes)))
    np.fill_diagonal(matrix_one, 0)
    T_c = (matrix_one / (len(graph.nodes) * (len(graph.nodes) - 1))).tolist()
    
    print(f"\nTraffic Matrix: Uniform distribution")
    print(f"Channel Bandwidth: {channel_bandwidth/1e9:.1f} GHz")
    print(f"Fibre Number: {fibre_num}")
    
    # Store results for all route functions
    all_results = []
    
    # Process each route function
    for route_function in route_functions:
        print(f"\n{'='*60}")
        print(f"Processing {route_function}")
        print(f"{'='*60}")
        
        # Calculate RWA based on route function
        try:
            if route_function == "FF-kSP":
                print("Calculating RWA using FF-kSP method...")
                print(f"Parameters: e={e_heuristic}, k={k_heuristic}, m_step={m_step}, max_count={max_count}")
                rwa = calculate_rwa_ff_ksp(
                    graph, T_c, channel_bandwidth,
                    e=e_heuristic, k=k_heuristic, m_step=m_step,
                    max_count=max_count, m_start=m_start, fibre_num=fibre_num,
                    band_config=band_config, span_length_km=span_length_km
                )
                
            elif route_function == "kSP-FF":
                print("Calculating RWA using kSP-FF method...")
                print(f"Parameters: e={e_heuristic}, k={k_heuristic}, m_step={m_step}, max_count={max_count}")
                rwa = calculate_rwa_ksp_ff(
                    graph, T_c, channel_bandwidth,
                    e=e_heuristic, k=k_heuristic, m_step=m_step,
                    max_count=max_count, m_start=m_start, fibre_num=fibre_num,
                    band_config=band_config, span_length_km=span_length_km
                )
                
            elif route_function == "ILP-connections":
                print("Calculating RWA using ILP method...")
                print(f"Parameters: e={e_ilp}, k={k_ilp}, max_time={max_time_ilp}s, threads={threads_ilp}")
                rwa = calculate_rwa_ilp(
                    graph, T_c, channel_bandwidth,
                    e=e_ilp, k=k_ilp, max_time=max_time_ilp,
                    threads=threads_ilp, fibre_num=fibre_num, blocking_rate=blocking_rate,
                    laptop_run=laptop_run, token_server=token_server,
                    band_config=band_config, span_length_km=span_length_km
                )
            else:
                print(f"Unknown route function: {route_function}. Skipping...")
                continue
            
            # Get number of channels from network
            temp_network = nt.Network.OpticalNetwork(
                graph,
                channel_bandwidth=channel_bandwidth,
                routing_func="FF-kSP",
                fibre_num=fibre_num,
                span_length_km=span_length_km
            )
            channels = temp_network.channels
            
            print(f"RWA calculated successfully!")
            print(f"Channels: {channels}")
            print(f"RWA connections: {len(rwa)}")
            
        except Exception as e:
            print(f"Error calculating RWA for {route_function}: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        # Sweep launch power
        print(f"\nSweeping launch power for {route_function}...")
        results = sweep_launch_power(
            graph, rwa, channel_bandwidth, channels,
            launch_power_range=launch_power_range,
            step=step,
            fibre_num=fibre_num,
            route_function=route_function,
            band_config=band_config,
            span_length_km=span_length_km
        )
        
        all_results.append(results)
        
        # Print results summary
        print(f"\n{route_function} Results Summary:")
        print("-" * 60)
        for i, (lp_dbm, lp_mw, thr_tbps) in enumerate(zip(
                results['launch_power_dbm'],
                results['launch_power_mw'],
                results['throughput_tbps']
        )):
            if not np.isnan(thr_tbps):
                print(f"Launch Power: {lp_dbm:6.1f} dBm ({lp_mw:6.3f} mW) -> "
                      f"Throughput: {thr_tbps:8.4f} Tbps")
        
        # Find optimal launch power
        valid_throughputs = np.array(results['throughput_tbps'])
        valid_mask = ~np.isnan(valid_throughputs)
        if np.any(valid_mask):
            max_idx = np.nanargmax(valid_throughputs)
            optimal_lp_dbm = results['launch_power_dbm'][max_idx]
            optimal_thr = results['throughput_tbps'][max_idx]
            print(f"\nOptimal Launch Power: {optimal_lp_dbm:.1f} dBm")
            print(f"Maximum Throughput: {optimal_thr:.4f} Tbps")

    # Plot all results together
    if len(all_results) > 0:
        plot_launch_power_vs_throughput(all_results, save_path="launch_power_sweep_comparison.png")

        # Save results to file
        import pandas as pd

        # Combine all results into a single DataFrame
        combined_data = []
        for results in all_results:
            route_func = results['route_function']
            for i in range(len(results['launch_power_dbm'])):
                combined_data.append({
                    'route_function': route_func,
                    'launch_power_dbm': results['launch_power_dbm'][i],
                    'launch_power_mw': results['launch_power_mw'][i],
                    'throughput_bps': results['throughput_bps'][i],
                    'throughput_tbps': results['throughput_tbps'][i]
                })

        df = pd.DataFrame(combined_data)
        df.to_csv("launch_power_sweep_results_comparison.csv", index=False)
        print("\nResults saved to launch_power_sweep_results_comparison.csv")
    else:
        print("\nNo results to plot. Please ensure RWA calculation succeeded for at least one route function.")