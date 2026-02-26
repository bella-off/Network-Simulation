import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

def plot_launch_power_sweep_from_csv(csv_file='launch_power_sweep_results_comparison.csv', save_path=None):
    """
    Read launch_power_sweep.csv and plot throughput vs launch power for three schemes

    :param csv_file: Path to the CSV file
    :param save_path: Path to save the plot (optional)
    """
    # Read CSV file
    df = pd.read_csv(csv_file)

    # Clean throughput_tbps column (remove "Tbps" suffix if present)
    def clean_throughput(value):
        if isinstance(value, str):
            # Remove "Tbps" and any whitespace
            value = re.sub(r'\s*Tbps\s*', '', value)
            try:
                return float(value)
            except ValueError:
                return np.nan
        return float(value)

    df['throughput_tbps'] = df['throughput_tbps'].apply(clean_throughput)

    # Filter launch power range: -8 to 8 dBm
    df_filtered = df[(df['launch_power_dbm'] >= -8) & (df['launch_power_dbm'] <= 8)]

    # Get unique route functions
    route_functions = df_filtered['route_function'].unique()

    # Define colors and markers for each scheme
    color_map = {
        'FF-kSP': 'blue',
        'kSP-FF': 'green',
        'ILP-connections': 'red'
    }

    marker_map = {
        'FF-kSP': 'o',
        'kSP-FF': 's',
        'ILP-connections': '^'
    }

    # Create single figure for Tbps results only
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Plot for each route function
    for route_func in route_functions:
        route_data = df_filtered[df_filtered['route_function'] == route_func].copy()

        # Sort by launch_power_dbm for proper line plotting
        route_data = route_data.sort_values('launch_power_dbm')

        # Get data
        launch_power_dbm = route_data['launch_power_dbm'].values
        throughput_tbps = route_data['throughput_tbps'].values

        # Remove NaN values
        valid_mask = ~np.isnan(throughput_tbps)
        launch_power_dbm_clean = launch_power_dbm[valid_mask]
        throughput_tbps_clean = throughput_tbps[valid_mask]

        # Get color and marker
        color = color_map.get(route_func, 'black')
        marker = marker_map.get(route_func, 'o')

        # Plot: Throughput (Tbps) vs Launch Power (dBm)
        ax.plot(launch_power_dbm_clean,
                 throughput_tbps_clean,
                 marker=marker,
                linewidth=2.5,
                 markersize=8,
                 color=color,
                 label=route_func,
                alpha=0.8,
                markeredgewidth=1.5,
                markeredgecolor='white')

    # Configure plot
    ax.set_xlabel('Launch Power [dBm]', fontsize=14, fontweight='bold')
    ax.set_ylabel('Throughput [Tbps]', fontsize=14, fontweight='bold')
    ax.set_title('Throughput vs Launch Power', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=12, loc='best', framealpha=0.9)
    ax.set_xlim(-8.5, 8.5)  # Set x-axis limits

    # Add some statistics text box
    # stats_text = f"Launch Power Range: -8 to 8 dBm\n"
    # stats_text += f"Schemes: {', '.join(route_functions)}\n"
    # stats_text += f"Total Data Points: {len(df_filtered)}"

    # fig.text(0.5, 0.02, stats_text, ha='center', fontsize=10,
    #          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.savefig('launch_power_sweep_plot.png', dpi=300, bbox_inches='tight')
        print("Plot saved to launch_power_sweep_plot.png")

    plt.show()

    # Print summary statistics
    print("\n" + "="*60)
    print("Summary Statistics")
    print("="*60)
    for route_func in route_functions:
        route_data = df_filtered[df_filtered['route_function'] == route_func]
        print(f"\n{route_func}:")
        print(f"  Launch Power Range: {route_data['launch_power_dbm'].min():.1f} to {route_data['launch_power_dbm'].max():.1f} dBm")
        print(f"  Throughput Range: {route_data['throughput_tbps'].min():.2f} to {route_data['throughput_tbps'].max():.2f} Tbps")
        print(f"  Max Throughput: {route_data['throughput_tbps'].max():.2f} Tbps at {route_data.loc[route_data['throughput_tbps'].idxmax(), 'launch_power_dbm']:.1f} dBm")
        print(f"  Min Throughput: {route_data['throughput_tbps'].min():.2f} Tbps at {route_data.loc[route_data['throughput_tbps'].idxmin(), 'launch_power_dbm']:.1f} dBm")

if __name__ == "__main__":
    # You can specify the CSV file path here
    csv_file = "launch_power_sweep_results_comparison.csv"

    # Plot and save
    plot_launch_power_sweep_from_csv(csv_file, save_path="launch_power_sweep_comparison.png")