import NetworkToolkit as nt
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict

def plot_path_snr_power(graph, rwa_result, network, save_path="path_snr_power_analysis.png"):
    """

   
    Returns:
    --------
    path_data : list
        包含每条 path 详细数据的列表
    """
    # 重新设置物理层（确保有最新的 NSR 数据）
    # 首先检查并验证波长编号范围
    if rwa_result:
        max_wavelength = max([w for w in rwa_result.keys()])
        min_wavelength = min([w for w in rwa_result.keys()])
        
        # 检查 wavelengths_physical 的长度是否足够
        if len(network.physical_layer.wavelengths_physical) < network.channels:
            print(f"Warning: wavelengths_physical length ({len(network.physical_layer.wavelengths_physical)}) < network.channels ({network.channels})")
            print(f"RWA wavelength range: {min_wavelength} to {max_wavelength}")
            print("This should have been fixed in plot_path_snr_power_from_database, but re-checking...")
        
        # 验证 RWA 中的波长编号是否在有效范围内
        if max_wavelength >= len(network.physical_layer.wavelengths_physical):
            print(f"Error: RWA contains wavelength {max_wavelength}, but only {len(network.physical_layer.wavelengths_physical)} wavelengths are defined.")
            print(f"Network channels: {network.channels}")
            print(f"wavelengths_physical length: {len(network.physical_layer.wavelengths_physical)}")
            raise ValueError(f"Wavelength index {max_wavelength} out of range [0, {len(network.physical_layer.wavelengths_physical)-1}]")
    
    network.physical_layer.add_uniform_launch_power_to_links(network.channels)
    network.physical_layer.add_wavelengths_to_links(rwa_result)
    
    # 在计算 NSR 之前，验证所有边上的波长编号都在有效范围内
    # 这是关键步骤：确保所有波长编号都在 wavelengths_physical 的有效索引范围内
    max_valid_wavelength = len(network.physical_layer.wavelengths_physical) - 1
    invalid_count = 0
    total_wavelengths_removed = 0
    
    for edge in network.graph.edges():
        if "wavelengths" in network.graph[edge[0]][edge[1]]:
            wavelengths_list = network.graph[edge[0]][edge[1]]["wavelengths"]
            original_count = len(wavelengths_list)
            
            # 过滤掉无效的波长（超出范围或负数）
            valid_wavelengths = [w for w in wavelengths_list 
                               if isinstance(w, (int, np.integer)) and 0 <= w <= max_valid_wavelength]
            
            if len(valid_wavelengths) < original_count:
                invalid_count += 1
                removed = original_count - len(valid_wavelengths)
                total_wavelengths_removed += removed
                if invalid_count <= 5:  # 只打印前5个错误
                    invalid_wavelengths = [w for w in wavelengths_list if w not in valid_wavelengths]
                    print(f"Warning: Edge {edge} has {removed} invalid wavelengths: {invalid_wavelengths[:10]}... (showing first 10)")
                
                # 更新边的波长列表
                network.graph[edge[0]][edge[1]]["wavelengths"] = valid_wavelengths
                network.graph[edge[1]][edge[0]]["wavelengths"] = valid_wavelengths
    
    if invalid_count > 0:
        print(f"\nWarning: {invalid_count} edges had invalid wavelengths")
        print(f"Total invalid wavelengths removed: {total_wavelengths_removed}")
        print(f"Valid wavelength range: 0 to {max_valid_wavelength}")
    
    # 再次验证：确保没有无效的波长
    for edge in network.graph.edges():
        if "wavelengths" in network.graph[edge[0]][edge[1]]:
            wavelengths_list = network.graph[edge[0]][edge[1]]["wavelengths"]
            for w in wavelengths_list:
                if w < 0 or w >= len(network.physical_layer.wavelengths_physical):
                    raise ValueError(f"Edge {edge} still has invalid wavelength {w} after filtering!")
    
    print(f"All wavelengths validated. Proceeding to calculate NSR...")
    
    network.physical_layer.add_non_linear_NSR_to_links(
        channel_bandwidth=network.channel_bandwidth,
        channels_full=network.channels
    )
    
    # 存储每条 path 的数据
    path_data = []
    
    # 遍历所有波长和路径
    for wavelength, paths in rwa_result.items():
        for path in paths:
            edges = network.physical_layer.nodes_to_edges(path)
            
            # 初始化累加值（仿照 get_lightpath_capacities_PLI 的方式）
            total_nsr = 0
            num_edges = 0
            
            # 存储每条边的信息，用于后续计算
            edge_info = []
            
            # 遍历路径上的每条边
            for edge in edges:
                if edge[0] in graph.nodes and edge[1] in graph.nodes:
                    # 获取该波长在边上的索引
                    if "wavelengths" in graph[edge[0]][edge[1]]:
                        wavelengths_list = graph[edge[0]][edge[1]]["wavelengths"]
                        if wavelength in wavelengths_list:
                            idx = wavelengths_list.index(wavelength)
                            num_edges += 1
                            
                            # 累加 NSR（仿照 get_lightpath_capacities_PLI 的方式）
                            if "NSR" in graph[edge[0]][edge[1]]:
                                nsr = graph[edge[0]][edge[1]]["NSR"][idx]
                                total_nsr += nsr
                                
                                # 存储每条边的信息
                                signal_power_edge = None
                                if "launch_powers" in graph[edge[0]][edge[1]]:
                                    signal_power_edge = graph[edge[0]][edge[1]]["launch_powers"][idx]
                                
                                edge_info.append({
                                    'nsr': nsr,
                                    'signal_power': signal_power_edge,
                                    'edge': edge
                                })
            
            # 计算 SNR（仿照 get_lightpath_capacities_PLI 的方式）
            snr = 1 / total_nsr if total_nsr > 0 else 0
            
            # 计算信号功率和噪声功率
            # 根据 NSR 的定义：NSR = (P_ase + P_NLI) / P_signal
            # 对于路径：total_NSR = sum(NSR_i) = sum((P_ase_i + P_NLI_i) / P_signal_i)
            # 如果假设每条边的信号功率相同（uniform launch power），则：
            # total_NSR = (1/P_signal) * sum(P_ase_i + P_NLI_i)
            # 所以：sum(P_ase_i + P_NLI_i) = total_NSR * P_signal
            
            # 获取单条边的信号功率（通常是 uniform 的）
            signal_power_per_edge = None
            if edge_info and edge_info[0]['signal_power'] is not None:
                signal_power_per_edge = edge_info[0]['signal_power']
            else:
                # 如果没有找到，使用默认值（1 mW = 0.001 W）
                signal_power_per_edge = 1e-3
            
            # 计算总噪声功率
            # 方法1：从累加的 NSR 和信号功率反推
            # total_noise_power = total_nsr * signal_power_per_edge
            # 但这种方法假设所有边的信号功率相同，可能不够准确
            
            # 方法2：从每条边的 NSR 和信号功率计算每条边的噪声功率，然后累加
            total_ase_power = 0
            total_nli_power = 0
            total_noise_power_from_edges = 0
            
            for edge_data in edge_info:
                if edge_data['signal_power'] is not None:
                    # 对于每条边：NSR = (P_ase + P_NLI) / P_signal
                    # 所以：P_ase + P_NLI = NSR * P_signal
                    noise_power_edge = edge_data['nsr'] * edge_data['signal_power']
                    total_noise_power_from_edges += noise_power_edge
                    
                    # 分别计算 ASE 和 NLI（需要从边的属性获取）
                    # 这里我们使用近似：从 NSR 反推
                    # 实际上，我们可以从边的 NSR 和信号功率计算总噪声功率
                    # 但要分离 ASE 和 NLI，需要单独计算 ASE
                    
                    # 计算该边的 ASE 功率
                    edge = edge_data['edge']
                    if edge[0] in graph.nodes and edge[1] in graph.nodes:
                        weight = graph[edge[0]][edge[1]].get("weight", 1)
                        wavelength_physical = network.physical_layer.wavelengths_physical[wavelength]
                        frequency = 3e8 / wavelength_physical
                        P_ase_edge = network.physical_layer.get_P_ase(
                            frequency, 
                            weight * network.physical_layer.span_length_km
                        )
                        total_ase_power += P_ase_edge
                        
                        # NLI = 总噪声 - ASE
                        P_nli_edge = noise_power_edge - P_ase_edge
                        total_nli_power += max(0, P_nli_edge)
            
            # 使用从边计算的总噪声功率
            total_noise_power = total_noise_power_from_edges if total_noise_power_from_edges > 0 else (total_ase_power + total_nli_power)
            
            # 验证：SNR 应该等于 signal_power / noise_power（在路径末端）
            # 但这里我们使用 NSR 累加的方式，这是标准做法
            # 如果信号功率在路径上衰减，路径末端的信号功率会更小
            # 但为了显示目的，我们使用发射功率
            
            # 存储数据
            path_data.append({
                'path': path,
                'wavelength': wavelength,
                'snr': snr,
                'snr_db': 10 * np.log10(snr) if snr > 0 else -np.inf,
                'signal_power': signal_power_per_edge,  # 单条边的发射功率
                'signal_power_db': 10 * np.log10(signal_power_per_edge * 1000) if signal_power_per_edge > 0 else -np.inf,  # 转换为 dBm
                'noise_power': total_noise_power,
                'noise_power_db': 10 * np.log10(total_noise_power * 1000) if total_noise_power > 0 else -np.inf,  # 转换为 dBm
                'ase_power': total_ase_power,
                'nli_power': total_nli_power,
                'path_length': len(path) - 1,  # 跳数
                'num_edges': num_edges,
                'path_str': f"{path[0]}→{path[-1]}",
                'path_full': '→'.join(map(str, path))  # 完整路径，例如 "1→3→5→2"
            })
    
    # 按路径排序
    path_data.sort(key=lambda x: (x['path'][0], x['path'][-1], x['wavelength']))
    
    # 创建图形
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    
    # 准备数据
    path_indices = range(len(path_data))
    snr_values = [d['snr_db'] for d in path_data]
    signal_power_values = [d['signal_power_db'] for d in path_data]
    noise_power_values = [d['noise_power_db'] for d in path_data]
    # 横坐标标签：显示完整路径和波长信息，例如 "1→3→5 (λ=10)"
    # 如果路径很长，只显示前几个节点和最后一个节点
    path_labels = []
    for d in path_data:
        path_full = d['path_full']
        if len(path_full) > 20:  # 如果路径字符串太长，简化显示
            path_nodes = d['path']
            if len(path_nodes) > 4:
                # 显示：起始节点→...→最后两个节点
                simplified = f"{path_nodes[0]}→...→{path_nodes[-2]}→{path_nodes[-1]}"
            else:
                simplified = path_full
            path_labels.append(f"{simplified} (λ={d['wavelength']})")
        else:
            path_labels.append(f"{path_full} (λ={d['wavelength']})")
    
    # 1. 绘制 SNR
    axes[0].bar(path_indices, snr_values, alpha=0.7, color='blue', edgecolor='black', linewidth=0.5)
    axes[0].set_xlabel('Path Index', fontsize=12)
    axes[0].set_ylabel('SNR [dB]', fontsize=12)
    axes[0].set_title('SNR for Each Lightpath', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3, linestyle='--')
    if len(path_data) > 0:
        axes[0].set_xticks(path_indices[::max(1, len(path_data)//20)])  # 只显示部分标签
        axes[0].set_xticklabels([path_labels[i] for i in path_indices[::max(1, len(path_data)//20)]], 
                                rotation=45, ha='right', fontsize=8)
    
    # 2. 绘制信号功率
    axes[1].bar(path_indices, signal_power_values, alpha=0.7, color='green', edgecolor='black', linewidth=0.5)
    axes[1].set_xlabel('Path Index', fontsize=12)
    axes[1].set_ylabel('Signal Power [dBm]', fontsize=12)
    axes[1].set_title('Signal Power for Each Lightpath', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3, linestyle='--')
    if len(path_data) > 0:
        axes[1].set_xticks(path_indices[::max(1, len(path_data)//20)])
        axes[1].set_xticklabels([path_labels[i] for i in path_indices[::max(1, len(path_data)//20)]], 
                                rotation=45, ha='right', fontsize=8)
    
    # 3. 绘制噪声功率（ASE + NLI，总噪声功率）
    axes[2].bar(path_indices, noise_power_values, alpha=0.7, color='red', edgecolor='black', linewidth=0.5)
    axes[2].set_xlabel('Path Index', fontsize=12)
    axes[2].set_ylabel('Noise Power [dBm]', fontsize=12)
    axes[2].set_title('Total Noise Power (ASE + NLI) for Each Lightpath', fontsize=14, fontweight='bold')
    axes[2].grid(True, alpha=0.3, linestyle='--')
    if len(path_data) > 0:
        axes[2].set_xticks(path_indices[::max(1, len(path_data)//20)])
        axes[2].set_xticklabels([path_labels[i] for i in path_indices[::max(1, len(path_data)//20)]], 
                                rotation=45, ha='right', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {save_path}")
    plt.close()
    
    # 打印统计信息
    print("\n" + "="*60)
    print("Path SNR and Power Statistics")
    print("="*60)
    print(f"Total number of paths: {len(path_data)}")
    
    valid_snr = [d['snr_db'] for d in path_data if d['snr_db'] > -np.inf]
    valid_signal = [d['signal_power_db'] for d in path_data if d['signal_power_db'] > -np.inf]
    valid_noise = [d['noise_power_db'] for d in path_data if d['noise_power_db'] > -np.inf]
    
    if valid_snr:
        print(f"Average SNR: {np.mean(valid_snr):.2f} dB")
        print(f"Min SNR: {min(valid_snr):.2f} dB")
        print(f"Max SNR: {max(valid_snr):.2f} dB")
    
    if valid_signal:
        print(f"Average Signal Power: {np.mean(valid_signal):.2f} dBm")
        print(f"Min Signal Power: {min(valid_signal):.2f} dBm")
        print(f"Max Signal Power: {max(valid_signal):.2f} dBm")
    
    if valid_noise:
        print(f"Average Noise Power: {np.mean(valid_noise):.2f} dBm")
        print(f"Min Noise Power: {min(valid_noise):.2f} dBm")
        print(f"Max Noise Power: {max(valid_noise):.2f} dBm")
    
    print("="*60 + "\n")
    
    return path_data


def plot_path_snr_power_from_database(db, collection, graph_id, route_function="kSP-FF", 
                                       band_config=None, span_length_km=80, save_path=None):
    """
    从数据库读取 RWA 结果并绘制 SNR 和功率
    
    Parameters:
    -----------
    db : str
        数据库名称
    collection : str
        集合名称
    graph_id : ObjectId or str
        图的 ID
    route_function : str
        路由函数名称（如 "kSP-FF", "FF-kSP", "ILP-connections"）
    band_config : dict
        频带配置（如果为 None，使用默认 C band）
    span_length_km : float
        光纤 span 长度（km）
    save_path : str
        保存图片的路径（如果为 None，自动生成）
    
    Returns:
    --------
    path_data : list
        包含每条 path 详细数据的列表
    """
    # 读取数据
    results = list(nt.Database.read_data(db, collection, find_dic={"_id": graph_id}, max_count=1))
    if len(results) == 0:
        raise ValueError(f"No data found for graph_id: {graph_id}")
    
    result_data = results[0]
    
    # 读取图
    graph_list = nt.Database.read_topology_dataset_list(db, collection, 
                                                        find_dic={"_id": graph_id}, 
                                                        node_data=True)
    if len(graph_list) == 0:
        raise ValueError(f"No graph found for graph_id: {graph_id}")
    
    graph, _id = graph_list[0]
    # Relabel graph node labels to start from 1 (if not already)
    graph = nx.relabel.convert_node_labels_to_integers(graph, first_label=1)
    
    # 读取 RWA 结果
    rwa_key = f"{route_function} RWA"
    if rwa_key not in result_data:
        raise ValueError(f"RWA result not found for {route_function}. Available keys: {list(result_data.keys())}")
    
    rwa_result = result_data[rwa_key]
    
    # 获取频带配置
    if band_config is None:
        # 默认 C band
        B_o = 5e12
        wavelength_start_nm = 1530
        wavelength_width_nm = 40.049
        RefLambda_nm = 1550
        Cr = 0
        channel_bandwidth = 50e9
    else:
        B_o = band_config['B_o_THz'] * 1e12
        wavelength_start_nm = band_config['wavelength_start_nm']
        wavelength_width_nm = band_config['wavelength_width_nm']
        RefLambda_nm = band_config['RefLambda_nm']
        Cr = band_config['Cr']
        channel_bandwidth = band_config['channel_bandwidth_GHz'] * 1e9
    
    # 初始化网络
    network = nt.Network.OpticalNetwork(
        graph, 
        B_o=B_o, 
        channel_bandwidth=channel_bandwidth, 
        fibre_num=1,
        span_length_km=span_length_km
    )
    
    # 配置物理层
    # 重要：重新分配物理波长，使用正确的频带参数
    # 因为 __init__ 中使用的是默认的 C band 参数
    network.physical_layer.assign_physical_wavelengths(
        channels=network.channels,
        wavelength_start_nm=wavelength_start_nm,
        wavelength_width_nm=wavelength_width_nm
    )
    network.physical_layer.RefLambda = RefLambda_nm * 1e-9
    network.physical_layer.Cr = Cr
    
    # 验证 wavelengths_physical 的长度
    print(f"\nNetwork configuration:")
    print(f"  B_o: {B_o/1e12:.2f} THz")
    print(f"  channel_bandwidth: {channel_bandwidth/1e9:.1f} GHz")
    print(f"  network.channels: {network.channels}")
    print(f"  wavelengths_physical length: {len(network.physical_layer.wavelengths_physical)}")
    
    if len(network.physical_layer.wavelengths_physical) != network.channels:
        raise ValueError(
            f"wavelengths_physical length ({len(network.physical_layer.wavelengths_physical)}) != network.channels ({network.channels}). "
            f"This should not happen after assign_physical_wavelengths is called."
        )
    
    print(f"Physical wavelengths configured: {len(network.physical_layer.wavelengths_physical)} channels")
    print(f"Wavelength range: {wavelength_start_nm:.1f} - {wavelength_start_nm + wavelength_width_nm:.1f} nm")
    
    # 转换 RWA 结果格式
    try:
        rwa_dict = nt.Tools.read_database_dict(rwa_result)
    except:
        rwa_dict = rwa_result
    
    # 验证 RWA 结果中的波长编号范围
    if rwa_dict:
        rwa_wavelengths = list(rwa_dict.keys())
        max_rwa_wavelength = max(rwa_wavelengths)
        min_rwa_wavelength = min(rwa_wavelengths)
        print(f"RWA wavelength range: {min_rwa_wavelength} to {max_rwa_wavelength}")
        print(f"Network channels: {network.channels}")
        print(f"wavelengths_physical length: {len(network.physical_layer.wavelengths_physical)}")
        
        if max_rwa_wavelength >= len(network.physical_layer.wavelengths_physical):
            raise ValueError(
                f"RWA contains wavelength {max_rwa_wavelength}, but only {len(network.physical_layer.wavelengths_physical)} "
                f"wavelengths are defined. This suggests a mismatch between the RWA calculation and the current band configuration."
            )
    
    # 生成保存路径
    if save_path is None:
        save_path = f"path_snr_power_{route_function}_{_id}.png"
    
    # 绘制
    print(f"Plotting SNR and power for {len(rwa_dict)} wavelengths...")
    path_data = plot_path_snr_power(graph, rwa_dict, network, save_path=save_path)
    
    return path_data


if __name__ == "__main__":
    # 示例用法
    db = "Topology_Data"
    collection = "topology-paper"
    
    # 从数据库读取一个图的 ID（你需要替换为实际的 ID）
    graph_list = nt.Database.read_topology_dataset_list(db, collection, 
                                                        find_dic={"name": "NSFNET"}, 
                                                        node_data=True, max_count=1)
    if len(graph_list) > 0:
        graph, graph_id = graph_list[0]
        
        # 定义频带配置（示例：C band）
        band_config = {
            "B_o_THz": 5.0,
            "wavelength_start_nm": 1530,
            "wavelength_width_nm": 40.049,
            "RefLambda_nm": 1550,
            "Cr": 0,
            "channel_bandwidth_GHz": 50
        }
        
        # 绘制
        path_data = plot_path_snr_power_from_database(
            db=db,
            collection=collection,
            graph_id=graph_id,
            route_function="kSP-FF",
            band_config=band_config,
            span_length_km=80,
            save_path="path_snr_power_kSP_FF.png"
        )
    else:
        print("No graph found in database")
