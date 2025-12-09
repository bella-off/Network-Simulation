"""
从数据库中读取已有的 RWA 结果并可视化
可以独立运行，不需要重新计算 RWA
"""

import NetworkToolkit as nt
import numpy as np
from plot_rwa import plot_rwa_heatmap


if __name__ == "__main__":
    # 配置
    db = "Topology_Data"
    collection = "topology-paper"  # 或 "real"，根据结果保存位置调整
    route_function = "FF-kSP"  # 或 "FF-kSP"
    
    # 读取 NSFNET 图
    print("读取 NSFNET 图...")
    graph_list = nt.Database.read_topology_dataset_list(
        "Topology_Data", "real",
        find_dic={"name": "NSFNET"},
        node_data=True,
        max_count=1
    )
    
    if len(graph_list) == 0:
        print("错误: 未找到 NSFNET 图")
        exit(1)
    
    graph, graph_id = graph_list[0][0], graph_list[0][1]
    print(f"找到图: {len(graph.nodes)} 个节点, {len(graph.edges)} 条边")
    
    # 从数据库读取 RWA 结果
    print(f"\n从数据库读取 {route_function} RWA 结果...")
    
    # 尝试从不同的集合读取
    collections_to_try = [collection, "real", "topology-paper"]
    rwa_result = None
    found_collection = None
    
    for coll in collections_to_try:
        try:
            results = list(nt.Database.read_data(
                db, coll,
                find_dic={"_id": graph_id},
                max_count=1
            ))
            
            if len(results) > 0:
                result_data = results[0]
                rwa_key = f"{route_function} RWA"
                
                if rwa_key in result_data:
                    rwa_result = result_data[rwa_key]
                    found_collection = coll
                    print(f"在集合 '{coll}' 中找到 RWA 结果")
                    break
        except Exception as e:
            print(f"从集合 '{coll}' 读取时出错: {e}")
            continue
    
    if rwa_result is None:
        print(f"\n错误: 未找到 {route_function} RWA 结果")
        print("请检查:")
        print(f"  1. 数据库: {db}")
        print(f"  2. 图 ID: {graph_id}")
        print(f"  3. RWA 键名: {route_function} RWA")
        exit(1)
    
    # 可视化
    print(f"\n开始绘制 RWA 热图...")
    print(f"找到的波长数: {len(rwa_result)}")
    
    plot_rwa_heatmap(
        graph,
        rwa_result,
        title=f"NSFNET with {route_function} Routing",
        save_path=f"rwa_{route_function}_nsfnet.png"
    )
    
    print(f"\n完成! rwa saved : rwa_{route_function}_nsfnet.png")

