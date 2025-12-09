import sys
import NetworkToolkit as nt
import numpy as np
import ray
import networkx as nx
import utils
import traceback
from datetime import datetime
def node_in_path_indicator(graph,  k=20):
    k_sp = nt.Routing.Tools.get_k_shortest_paths_MNH(graph, k=k,e=1000, data_dict=True, weighted=None)
    I_nsdk = np.zeros((len(graph), len(graph), len(graph), k))
    for n in range(len(graph)):
        node = n+1
        for s_d_ind,(s,d) in enumerate(k_sp):
            for ind_p, path in enumerate(k_sp[(s,d)]):
                if node in path:
                    I_nsdk[n, s-1, d-1, ind_p] = 1
    return I_nsdk

def rwa_to_path_prob_dists(graph, rwa, k =20, e=1000):
    
    k_sp = nt.Routing.Tools.get_k_shortest_paths_MNH(graph, k=k, e=e, data_dict=True, weighted=None)
    path_probs = {(s,d):[0 for i in range(k)] for s,d in k_sp}
    for wave in rwa:
        for path in rwa[wave]:
            s,d = path[0], path[-1]
            path_ind = k_sp[(s,d)].index(path)
            path_probs[(s,d)][path_ind] += 1
            path_probs[(d,s)][path_ind] += 1
    for s,d in path_probs:
        normaliser = sum(path_probs[(s,d)])
        if normaliser == 0:
            continue
        for ind, item in enumerate(path_probs[(s,d)]):
            path_probs[(s,d)][ind] /= normaliser
            path_probs[(d,s)][ind] /= normaliser
    
    return path_probs

def calculate_node_load(graph, rwa):
    node_load = {node: 0 for node in graph.nodes}
    for wave in rwa:
        for path in rwa[wave]:
            for node in path:
                node_load[node] +=1
            
    return node_load

def calculate_theoretical_node_load(graph, T_c, path_probs, k, lambda_n):
    I_nsdk = node_in_path_indicator(graph, k=k)
    p_sdk = path_prob_dist_specific(graph, path_probs, k=k)
    transit_traffic = lambda n: 2*np.sum([sum([ sum([ T_c[s,d] * I_nsdk[n, s, d, k]*p_sdk[s,d,k] for k in range(k) if n != s and n != d
                                                     and d>s]) for d in range(len(graph))]) for s in range(len(graph))])
    nodal_traffic = lambda n: np.sum([T_c[n, d] for d in range(len(graph)) if d != n])
    node_load = {node: (graph.degree[node] * lambda_n)/(transit_traffic(node-1) + nodal_traffic(node-1)) for node in graph.nodes}
    return node_load
def path_probability_distribution(graph, p_list, k=20):
    p_sdk = np.array([[[p_list[k] for k in range(k)] for d in range(len(graph))] for s in range(len(graph))])
    return p_sdk

def path_prob_dist_specific(graph, path_probs, k=20):
    p_sdk = np.array([[[path_probs[(s+1,d+1)][k] if s!=d else 0 for k in range(k)] for d in range(len(graph))] for s in range(len(graph))])
    return p_sdk

def throughput_upper_bound_summed(graph, T_c, p_list, lambda_n=10, k=20, path_probs=None):
    I_nsdk = node_in_path_indicator(graph, k=k)
    if p_list is not None:
        p_sdk = path_probability_distribution(graph, p_list, k=k)
    elif path_probs is not None:
        p_sdk = path_prob_dist_specific(graph, path_probs, k=k)
    transit_traffic = lambda n: 2*np.sum([sum([ sum([ T_c[s,d] * I_nsdk[n, s, d, k]*p_sdk[s,d,k] for k in range(k) if n != s and n != d and d>s]) for d in range(len(graph))]) for s in range(len(graph))])
    nodal_traffic = lambda n: np.sum([sum([T_c[n, d]*I_nsdk[n, n, d, k]*p_sdk[n, d, k]  for k in range(k) if d != n]) for d in range(len(graph))])
    # print(graph.degree[1] * lambda_n)
    # print(nodal_traffic(0))
    # print(transit_traffic(0))
    theta_upper_bound = sum([graph.degree[node] * lambda_n for node in graph.nodes])/sum([(transit_traffic(node-1) + nodal_traffic(node-1)) for node in graph.nodes])
    return 2*theta_upper_bound


def edge_in_path_indicator(graph, k=20):
    k_sp = nt.Routing.Tools.get_k_shortest_paths_MNH(graph, k=k, e=1000, data_dict=True, weighted=None)
    I_esdk = np.zeros((len(graph),len(graph), len(graph), len(graph), k))
    for e1, e2 in graph.edges:
        for s_d_ind, (s,d) in enumerate(k_sp):
            for ind_p, path in enumerate(k_sp[(s,d)]):
                path_edges = nt.Tools.nodes_to_edges(path)
                if (e1, e2) in path_edges or (e2, e1) in path_edges:
                    I_esdk[e1-1, e2-1, s-1, d-1, ind_p]=1
                    I_esdk[e2-1, e1-1, d-1, s-1, ind_p]=1
    return I_esdk

@ray.remote
def throughput_upper_bound_bottle_neck(graph_list, lambda_n=10, k=20, db="Topology_Data", collection=None, pb_actor=None, min_edge_bound=False):
    for graph, _id, T_c, rwa, M in graph_list:
        try:
            graph = nx.convert_node_labels_to_integers(graph, first_label=1)
            T_c = np.ceil(np.array(T_c)*M)
            np.fill_diagonal(T_c, 0)
            path_probs = rwa_to_path_prob_dists(graph, rwa, k=k)
            I_nsdk = node_in_path_indicator(graph, k=k)
            k_sp = nt.Routing.Tools.get_k_shortest_paths_MNH(graph, k=k, e=1000, data_dict=True, weighted="weight")
            # print(rwa)
            # print(k_sp[(15,11)])
            # print(T_c)
            # print(I_nsdk)
            
            p_sdk = path_prob_dist_specific(graph, path_probs, k=k)
            
            transit_traffic = lambda n: 2*np.sum([sum([ sum([T_c[s,d] * I_nsdk[n, s, d, k]*p_sdk[s,d,k] for k in range(k) if n != s and n != d
                                                            and d>s]) for d in range(len(graph))]) for s in range(len(graph))])
            nodal_traffic = lambda n: np.sum([T_c[n, d] for d in range(len(graph)) if d != n])
            edge_traffic = lambda e1, e2: np.sum([sum([ sum([ T_c[s,d] * I_esdk[e1, e2, s, d, k]*p_sdk[s,d,k] for k in range(k) if d>s]) for d in range(len(graph))]) for s in range(len(graph))])
            theta_upper_bound_node = [(graph.degree[node] * lambda_n)/(transit_traffic(node-1) + nodal_traffic(node-1)) for node in graph.nodes]
            top = [(graph.degree[node] * lambda_n) for node in graph.nodes]
            transit = [transit_traffic(node-1) for node in graph.nodes]  
            
            nodal =np.array([nodal_traffic(node-1) for node in graph.nodes])
            bottom = np.array([nodal_traffic(node-1) + transit_traffic(node-1) for node in graph.nodes])
            # print("top: {}".format(top))
            # print("transit: {}".format(transit))
            # print("nodal: {}".format(nodal))
            # print("bottom: {}".format(bottom))
            # print("theta upper bound: {}".format(theta_upper_bound_node))
            # print(np.min(top/bottom))
            

            ilp_conns = sum([1 for wave in rwa for path in rwa[wave]])
            
            if min_edge_bound:
                I_esdk = edge_in_path_indicator(graph, k=k)
                theta_upper_bound_edge = [(lambda_n)/(edge_traffic(e1-1, e2-1)) for e1, e2 in graph.edges]
                
                # print("edge traffic: {}".format(edge_traf))
                # for s,d in graph.edges:
                #     print(s,d, (lambda_n)/(edge_traffic(s-1, d-1)))
                theta_upper_bound_edge = np.min(theta_upper_bound_edge)/2
                # print(theta_upper_bound_edge)
                assigned = (theta_upper_bound_edge*T_c).sum()
                nt.Database.update_data_with_id(db, collection, _id,
                                                        newvals={"$set": {"theta_star_min_ilp": theta_upper_bound_edge, 
                                                                        "theta_star_min_ilp LP assigned": assigned, 
                                                                        "ILP-connections LP assigned":ilp_conns}})
            else:
                theta_upper_bound = np.min(theta_upper_bound_node)/2
                
                assigned = (theta_upper_bound*T_c).sum()
                nt.Database.update_data_with_id(db, collection, _id,
                                                        newvals={"$set": {"theta_star_ilp": theta_upper_bound, 
                                                                        "theta_star_ilp LP assigned":assigned, 
                                                                        "ILP-connections LP assigned":ilp_conns,
                                                                        "theta_star_ilp timestamp": datetime.utcnow()}})

            
            if pb_actor:
                    pb_actor.update.remote(1)
        except Exception as e:
            traceback.print_exc()
        # print("theta upper_bound: {}".format(theta_upper_bound))
        # return theta_upper_bound
    
def throughput_upper_bound_bottle_neck_parralel(graph_list, lambda_n=10, k=20, db="Topology_Data", collection=None, min_edge_bound=False):
    ray.shutdown()
    ray.init()
    indeces = nt.Tools.create_start_stop_list(len(graph_list), len(graph_list))
    pb = nt.Tools.ProgressBar(len(graph_list))
    actor = pb.actor
    
    # graph_list = [(graph, _id, T_c, rwa) for graph, _id, T_c, rwa in graph_list]
    results = [throughput_upper_bound_bottle_neck.remote(graph_list=graph_list[indeces[ind]:indeces[ind + 1]], lambda_n=lambda_n,
                                                         db=db, collection=collection,pb_actor=actor, k=k, min_edge_bound=min_edge_bound) for ind in range(len(graph_list))]
    pb.print_until_done()
    results = ray.get(results)
    print(results)

def calc_T_c(rwa, graph):
    rwat = np.zeros((len(graph), len(graph)))
    for wave in rwa:
        for path in rwa[wave]:
            
            s, d = path[0], path[-1]
            rwat[s-1, d-1] += 1
            rwat[d-1, s-1] += 1 
    rwat = rwat
    return rwat

if __name__ == "__main__":
    db = "DRLRoutingDB"
    coll = "ilp_rwa_test"
    
    ray.shutdown()
    graph_list_lp = nt.Database.read_topology_dataset_list(db, coll, "T_c","ILP-connections RWA", "ILP-connections M",
                                                            find_dic={"ILP-connections RWA":{"$exists":True}}, max_count=100, parralel=False)
    
    utils.dump_data(graph_list_lp, "/rdata/ong/robin/thesis/throughput_bound/{}.pkl".format(coll))
    # graph_list_lp = utils.read_data("/rdata/ong/robin/thesis/throughput_bound/{}.pkl".format(coll))
    # graph_list_lp = [(graph, _id, T_c, rwa, M) for graph, _id, T_c, rwa, M, lp_ass in graph_list_lp]
    # graph = graph_list_lp[0][0]
    throughput_upper_bound_bottle_neck_parralel(graph_list_lp, lambda_n=100, k=5, db=db, collection=coll, min_edge_bound=True)
    throughput_upper_bound_bottle_neck_parralel(graph_list_lp, lambda_n=100, k=5, db=db, collection=coll, min_edge_bound=False)
    