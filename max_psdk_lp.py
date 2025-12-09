import sys
import NetworkToolkit as nt
from NetworkToolkit.NetworkSimulator import ILP_multi_fibre
import numpy as np
import ray
from tqdm import tqdm
import utils

if __name__ == "__main__":
    hostname = "128.40.41.48"
    db = "DRLRoutingDB"
    coll = "ilp_rwa"
    port = 7112

    # graph_list = nt.Database.read_topology_dataset_list(db, coll, "T_c", 
    #                                                         find_dic={"ILP-connections RWA":{"$exists":True}}, max_count=6000, parralel=False)
    
    # graph_list = [(graph, _id, np.array(T_c)) for graph, _id, T_c in graph_list]
    graph_list = utils.read_data("/rdata/ong/robin/thesis/throughput_bound/LP-{}.pkl".format(coll))
    # utils.dump_data(graph_list, "/rdata/ong/robin/thesis/throughput_bound/LP-{}.pkl".format(coll))
    nt.NetworkSimulator.parralel_max_psdk_edge_bound(graph_list, db=db, collection=coll, k=5,e=None, threads=1, workers=len(graph_list), 
                                                            local=True, bandwidth=50e9, channels=100, hostname=hostname, port=port)
    






        
        

    
