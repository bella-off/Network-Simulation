import sys
import NetworkToolkit as nt
from NetworkToolkit.NetworkSimulator import ILP_multi_fibre
import numpy as np
import ray
from tqdm import tqdm
import networkx as nx

def calculate_load_distribution_rwa(graph, rwa):
    graph = nx.convert_node_labels_to_integers(graph, first_label=1)
    load_dist = [0 for i in range(len(graph))]
    for wave in rwa:
        for path in rwa[wave]:
            for node in path:
                load_dist[node-1] += 1
    print(load_dist)
    load_dist = [load_dist[i]/sum(load_dist) for i in range(len(load_dist))]
    return load_dist

def calculate_load_distribution_p_sdk(graph, p_sdk, theta, T_c, k=20):
    graph = nx.convert_node_labels_to_integers(graph, first_label=1)
    load_dist = [0 for i in range(len(graph))]

    traffic_matrix = np.floor((theta*np.array(T_c)))
    # Get k shortest paths for all source-destination pairs using the 
    k_sp = nt.Routing.Tools.get_k_shortest_paths_MNH(graph, k=k, data_dict=True)
    for (s,d) in k_sp:
        for ind, path in enumerate(k_sp[(s,d)]):
            traffic_path = traffic_matrix[s-1][d-1]*p_sdk[s-1][d-1][ind]
            for node in path:
                load_dist[node-1] += traffic_path
    load_dist = [load_dist[i]/sum(load_dist) for i in range(len(load_dist))]
    return load_dist

def calculate_fully_loaded_load_dist(graph):
    graph = nx.convert_node_labels_to_integers(graph, first_label=1)
    sum_degree = sum([graph.degree[node] for node in graph.nodes])
    load_dist = [graph.degree[node]/sum_degree for node in graph.nodes]
    return load_dist

def update_db_load_dist(db, collection, data_list, key="load_distribution"):
    for _id, load_dist in data_list:
        nt.Database.update_data_with_id(db, collection, _id, newvals={"$set":{
            key: load_dist
        }})

if __name__ == "__main__":
    db = "Topology_Data"
    hostname = "128.40.41.48"
    collection = "topology-paper"
    query = { "nodes" : 14, "ILP Capacity" : { "$exists" : True }, "ILP-connections RWA":{"$exists":True}, "ILP-connections":{"$ne":0}}
    port = 7112
    # graph_list = nt.Database.read_topology_dataset_list(db, collection, "max_e p_sdk", "theta_star_e", "max_e p_sdk T_c", find_dic=query)
    # graph_list = nt.Database.read_topology_dataset_list(db, collection, "ILP-connections RWA", find_dic=query)
    graph_list = nt.Database.read_topology_dataset_list(db, collection, find_dic=query)

    # data_list = [(_id, calculate_load_distribution_rwa(graph, rwa)) for graph, _id, rwa in graph_list]    
    # data_list = [(_id, calculate_load_distribution_p_sdk(graph, p_sdk, theta, T_c, k=19)) for graph, _id, p_sdk, theta, T_c in graph_list]    
    data_list = [(_id, calculate_fully_loaded_load_dist(graph)) for graph, _id in graph_list]

    # update_db_load_dist(db, collection, data_list, key="ILP-connections load_distribution")
    # update_db_load_dist(db, collection, data_list, key="max_e load_distribution")
    update_db_load_dist(db, collection, data_list, key="fully_loaded load_distribution")