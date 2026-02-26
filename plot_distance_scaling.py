import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

def plot_distance_scaling_from_csv(csv_file='distance_scaling_results.csv', save_path=None):
    """
    Read distance_scaling_results.csv and plot throughput vs fiber span length for three schemes

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

    if 'throughput_tbps' in df.columns:
        df['throughput_tbps'] = df['throughput_tbps'].apply(clean_throughput)

    # Clean span_length_km column
    def clean_span_length(value):
        if isinstance(value, str):
            # Remove "km" and any whitespace
            value = re.sub(r'\s*km\s*', '', value)
            try:
                return float(value)
            except ValueError:
                return np.nan
        return float(value)

    if 'span_length_km' in df.columns:
        df['span_length_km'] = df['span_length_km'].apply(clean_span_length)

    # Get unique route functions
    route_functions = df['route_function'].unique()

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

    linestyle_map = {
        'FF-kSP': '-',
        'kSP-FF': '--',
        'ILP-connections': '-.'
    }

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Plot for each route function
    for route_func in route_functions:
        route_data = df[df['route_function'] == route_func].copy()

        # Sort by span_length_km for proper line plotting
        route_data = route_data.sort_values('span_length_km')

        # Get data
        span_length_km = route_data['span_length_km'].values
        throughput_tbps = route_data['throughput_tbps'].values

        # Remove NaN values
        valid_mask = ~np.isnan(throughput_tbps) & ~np.isnan(span_length_km)
        span_length_km_clean = span_length_km[valid_mask]
        throughput_tbps_clean = throughput_tbps[valid_mask]

        # Get color, marker, and linestyle
        color = color_map.get(route_func, 'black')
        marker = marker_map.get(route_func, 'o')
        linestyle = linestyle_map.get(route_func, '-')

        # Plot
        ax.plot(span_length_km_clean,
                throughput_tbps_clean,
                marker=marker,
                linewidth=2.5,
                markersize=8,
                color=color,
                linestyle=linestyle,
                label=route_func,
                alpha=0.8,
                markeredgewidth=1.5,
                markeredgecolor='white')

    # Configure plot
    ax.set_xlabel('Fiber Span Length [km]', fontsize=14, fontweight='bold')
    ax.set_ylabel('Throughput [Tbps]', fontsize=14, fontweight='bold')
    ax.set_title('Throughput vs Fiber Span Length', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--', which='both')
    ax.legend(fontsize=12, loc='best', framealpha=0.9)

    # Set x-axis limits if needed (optional, can be removed if you want auto-scaling)
    if 'span_length_km' in df.columns:
        min_span = df['span_length_km'].min()
        max_span = df['span_length_km'].max()
        # Add some padding
        ax.set_xlim(min_span - 2, max_span + 2)

    plt.tight_layout()

    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.savefig('distance_scaling_plot.png', dpi=300, bbox_inches='tight')
        print("Plot saved to distance_scaling_plot.png")

    plt.show()

    # Print summary statistics
    print("\n" + "="*60)
    print("Summary Statistics")
    print("="*60)
    for route_func in route_functions:
        route_data = df[df['route_function'] == route_func]
        valid_data = route_data.dropna(subset=['span_length_km', 'throughput_tbps'])
        
        if len(valid_data) > 0:
            print(f"\n{route_func}:")
            print(f"  Span Length Range: {valid_data['span_length_km'].min():.1f} to {valid_data['span_length_km'].max():.1f} km")
            print(f"  Throughput Range: {valid_data['throughput_tbps'].min():.4f} to {valid_data['throughput_tbps'].max():.4f} Tbps")
            
            # Find max throughput
            max_idx = valid_data['throughput_tbps'].idxmax()
            print(f"  Max Throughput: {valid_data.loc[max_idx, 'throughput_tbps']:.4f} Tbps at {valid_data.loc[max_idx, 'span_length_km']:.1f} km")
            
            # Find min throughput
            min_idx = valid_data['throughput_tbps'].idxmin()
            print(f"  Min Throughput: {valid_data.loc[min_idx, 'throughput_tbps']:.4f} Tbps at {valid_data.loc[min_idx, 'span_length_km']:.1f} km")
            
            # Calculate throughput change
            throughput_change = valid_data['throughput_tbps'].max() - valid_data['throughput_tbps'].min()
            throughput_change_pct = (throughput_change / valid_data['throughput_tbps'].max()) * 100
            print(f"  Throughput Change: {throughput_change:.4f} Tbps ({throughput_change_pct:.2f}%)")

if __name__ == "__main__":
    # You can specify the CSV file path here
    csv_file = "distance_scaling.csv"

    # Plot and save
    plot_distance_scaling_from_csv(csv_file, save_path="distance_scaling_comparison.png")

