import NetworkToolkit as nt
import numpy as np



if __name__ == "__main__":
    collection = "topology-paper"
    db = "Topology_Data"

    # geneva  hostname = "128.40.42.10"
    # london  hostname = "128.40.42.13"

    hostname = "128.40.42.13"

    port = 6379
    # hostname = None  # None to use the Ray
    # port = 6379  # local Ray port（will not be used actually）

    # query = { "nodes" : 14, "ILP Capacity" : { "$exists" : True }, "FF-kSP RWA" : { "$exists" : False }}
    # query = { "nodes" : 10}

    query = None

    # nsfnet_graph = nt.Database.read_topology_dataset_list("Topology_Data", "real", "ILP RWA assignment",
    #                                                       find_dic={"name": "NSFNET"}, node_data=True)
    graph_list = nt.Database.read_topology_dataset_list(db, collection, find_dic={"name": "NSFNET"},
                                                        node_data=True)

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
                                                               e=100, k=5, m_step=200, channel_bandwidth=50e9,
                                                               max_count=10,
                                                               m_start=0,
                                                               port=port,
                                                               hostname=hostname, fibre_num=1)
    route_function = "kSP-FF"
    for graph, _id, T_c in graph_list:
        # 打印拓扑信息 (Topology Design)
        print(f"\nTopology ID: {_id}")
        print(f"Number of Nodes: {len(graph.nodes)}")
        print(f"Number of Edges: {len(graph.edges)}")
        print(f"Average Degree: {2 * len(graph.edges) / len(graph.nodes):.2f}")

        # 从数据库读取结果
        try:
            results = list(nt.Database.read_data(db, collection,
                                                 find_dic={"_id": _id},
                                                 max_count=1))
            if len(results) > 0:
                result_data = results[0]

                # 打印吞吐量结果
                capacity_key = f"{route_function} Capacity"
                connections_key = f"{route_function}-connections"
                time_key = f"{route_function} time"

                if capacity_key in result_data:
                    throughput = result_data[capacity_key]
                    print(f"\nThroughput: {throughput:.2e} bps ({throughput / 1e12:.4f} Tbps)")

                if connections_key in result_data:
                    connections = result_data[connections_key]
                    print(f"Max Connections M: {connections}")

                if time_key in result_data:
                    time_taken = result_data[time_key]
                    print(f"Computation Time: {time_taken:.2f} s")

                # 打印其他参数  kSP-FF RWA
                if f"{route_function} RWA" in result_data:
                    RWA_results = result_data[f"{route_function} RWA"]
                    print(f"RWA: {RWA_results}")

                if f"{route_function} channel number" in result_data:
                    channels = result_data[f"{route_function} channel number"]
                    print(f"Number of Channels: {channels}")

                if f"{route_function} channel bandwidth" in result_data:
                    bandwidth = result_data[f"{route_function} channel bandwidth"]
                    print(f"Channel Bandwidth: {bandwidth / 1e9:.1f} GHz")

                # ============================================================
                # 可视化 RWA 结果（仿照图 3.2）
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
                        print("警告: 无法导入 plot_rwa 模块，跳过可视化")
                    except Exception as e:
                        print(f"可视化时出错: {e}")
            else:
                print(f"\nWarning: No result data found for ID {_id}")
        except Exception as e:
            print(f"\nError reading results: {e}")

    print("=" * 60 + "\n")
