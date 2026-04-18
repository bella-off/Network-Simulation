

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from networktoolbox.NetworkToolkit.Tools import nodes_to_edges, read_database_dict


def plot_rwa_heatmap(graph, rwa_dict, title="RWA Wavelength Allocation", 
                     save_path=None, max_channels=None, figsize=(12, 8)):
    """

    -----
    graph : networkx.Graph

    rwa_dict : dict
        RWA ：{'0': [[path 1], [path 2], ...], '1': [...], ...}
    title : str
    save_path : str, optional
    max_channels : int, optional
    figsize : tuple
    """
    if rwa_dict:
        first_key = list(rwa_dict.keys())[0]
        if isinstance(first_key, str):
            rwa_dict = read_database_dict(rwa_dict)
    
    # sort
    edges = sorted(list(graph.edges()))
    num_links = len(edges)
    link_id_map = {edge: i for i, edge in enumerate(edges)}
    
    # max channel number
    if max_channels is None:
        if rwa_dict:
            max_channels = max([int(k) for k in rwa_dict.keys()]) + 1
        else:
            max_channels = 312  #
    
    # initialization：row=path，Colum=channel
    matrix = np.zeros((num_links, max_channels), dtype=int)
    
    # matrix：1 assignment，0 no
    for wavelength, paths in rwa_dict.items():
        wavelength = int(wavelength)
        if wavelength >= max_channels:
            continue
        
        for path in paths:
            # path->path list
            path_edges = nodes_to_edges(path)
            
            for edge in path_edges:
                if edge in link_id_map:
                    link_id = link_id_map[edge]
                    matrix[link_id, wavelength] = 1
                elif (edge[1], edge[0]) in link_id_map:
                    link_id = link_id_map[(edge[1], edge[0])]
                    matrix[link_id, wavelength] = 1
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # vmin=0（white），vmax=1（red）
    im = ax.imshow(matrix, aspect='auto', cmap='Reds', interpolation='nearest',
                   vmin=0, vmax=1, origin='lower')
    

    ax.set_xlabel('Channel ID', fontsize=12)
    ax.set_ylabel('Link ID', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    

    ax.set_xlim(-0.5, max_channels - 0.5)
    ax.set_ylim(-0.5, num_links - 0.5)
    

    channel_interval = max(50, max_channels // 10)
    channel_ticks = np.arange(0, max_channels, channel_interval)
    ax.set_xticks(channel_ticks)
    ax.set_xticklabels(channel_ticks)
    
    link_interval = max(1, num_links // 20)
    link_ticks = np.arange(0, num_links, link_interval)
    ax.set_yticks(link_ticks)
    ax.set_yticklabels(link_ticks)
    
    ax.grid(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"figure saved: {save_path}")
    
    plt.show()
    
    return fig, ax, matrix

