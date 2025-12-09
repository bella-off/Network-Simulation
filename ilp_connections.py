import NetworkToolkit as nt
import numpy as np


nt.Network
if __name__ == "__main__":
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

    matrix_one = np.ones((len(graph_list[0][0].nodes), len(graph_list[0][0].nodes)))
    np.fill_diagonal(matrix_one, 0)
    T_c = (matrix_one / (len(graph_list[0][0].nodes) * (len(graph_list[0][0].nodes) - 1))).tolist()
    graph_list = [(graph, _id, T_c) for graph,_id in graph_list]

    nt.NetworkSimulator.parralel_ILP_connections(graph_list, db="Topology_Data",collection="topology-paper", max_time=48*3600, workers=len(graph_list),
                                 threads=1, fibre_num=1, hostname=hostname, port=port,
                                 insert=False, bandwidth=50e9, throughput=False, blocking_rate=0,
                                 k=3)

    # k=20