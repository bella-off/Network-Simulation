"""
RWA 波长分配热图可视化函数
仿照图 3.2 的风格：Wavelength allocations of RWAs
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from networktoolbox.NetworkToolkit.Tools import nodes_to_edges, read_database_dict


def plot_rwa_heatmap(graph, rwa_dict, title="RWA Wavelength Allocation", 
                     save_path=None, max_channels=None, figsize=(12, 8)):
    """
    绘制 RWA 波长分配热图（仿照图 3.2 风格）
    
    参数:
    -----
    graph : networkx.Graph
        网络拓扑图
    rwa_dict : dict
        RWA 分配字典，格式：{'0': [[路径1], [路径2], ...], '1': [...], ...}
        或 {0: [[路径1], ...], 1: [...], ...}
    title : str
        图标题
    save_path : str, optional
        保存路径（如 "rwa_plot.png"）
    max_channels : int, optional
        最大信道数，如果不指定则自动推断
    figsize : tuple
        图像大小，默认 (12, 8)
    """
    # 转换字符串键为整数键（如果从数据库读取）
    if rwa_dict:
        first_key = list(rwa_dict.keys())[0]
        if isinstance(first_key, str):
            rwa_dict = read_database_dict(rwa_dict)
    
    # 获取所有链路并排序
    edges = sorted(list(graph.edges()))
    num_links = len(edges)
    link_id_map = {edge: i for i, edge in enumerate(edges)}
    
    # 确定最大信道数
    if max_channels is None:
        if rwa_dict:
            max_channels = max([int(k) for k in rwa_dict.keys()]) + 1
        else:
            max_channels = 312  # 默认值
    
    # 初始化矩阵：行=链路，列=信道
    matrix = np.zeros((num_links, max_channels), dtype=int)
    
    # 填充矩阵：1 表示已分配，0 表示未分配
    for wavelength, paths in rwa_dict.items():
        wavelength = int(wavelength)
        if wavelength >= max_channels:
            continue
        
        for path in paths:
            # 将路径（节点序列）转换为链路列表
            path_edges = nodes_to_edges(path)
            
            # 标记路径上的所有链路在该波长上已分配
            for edge in path_edges:
                # 检查边是否存在（考虑方向）
                if edge in link_id_map:
                    link_id = link_id_map[edge]
                    matrix[link_id, wavelength] = 1
                # 如果反向边存在（无向图情况）
                elif (edge[1], edge[0]) in link_id_map:
                    link_id = link_id_map[(edge[1], edge[0])]
                    matrix[link_id, wavelength] = 1
    
    # 创建图形
    fig, ax = plt.subplots(figsize=figsize)
    
    # 绘制热图：红色表示已分配，白色表示未分配
    # 使用 'Reds' 颜色映射，vmin=0（白色），vmax=1（红色）
    im = ax.imshow(matrix, aspect='auto', cmap='Reds', interpolation='nearest',
                   vmin=0, vmax=1, origin='lower')
    
    # 设置标签和标题
    ax.set_xlabel('Channel ID', fontsize=12)
    ax.set_ylabel('Link ID', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # 设置坐标轴范围
    ax.set_xlim(-0.5, max_channels - 0.5)
    ax.set_ylim(-0.5, num_links - 0.5)
    
    # 设置刻度
    # X 轴（信道 ID）：每 50 个或合理间隔显示一次
    channel_interval = max(50, max_channels // 10)
    channel_ticks = np.arange(0, max_channels, channel_interval)
    ax.set_xticks(channel_ticks)
    ax.set_xticklabels(channel_ticks)
    
    # Y 轴（链路 ID）：合理间隔显示
    link_interval = max(1, num_links // 20)
    link_ticks = np.arange(0, num_links, link_interval)
    ax.set_yticks(link_ticks)
    ax.set_yticklabels(link_ticks)
    
    # 不显示网格（与图 3.2 风格一致）
    ax.grid(False)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"figure saved: {save_path}")
    
    plt.show()
    
    return fig, ax, matrix

