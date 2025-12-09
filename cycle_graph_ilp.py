import sys
import NetworkToolkit as nt
from NetworkToolkit.NetworkSimulator import ILP_multi_fibre


if __name__ == "__main__":
    hostname = "128.40.41.48"
    port = 7112
    K = [20, 40, 60, 80, 100]
    graph_list = nt.Database.read_topology_dataset_list("Topology_Data", "cycle_graphs", "T_c",
                                                            find_dic={"nodes":{"$lte":40}},
                                                            parralel=False)

    nt.NetworkSimulator.parralel_ILP_connections(graph_list, collection="cycle_graphs", k=5, hostname=hostname, threads=1, throughput=False, workers=len(graph_list),
                                                local=True, bandwidth=12.5e9)
    # nt.NetworkSimulator.maximum_throughput_routing(collection="cycle_graph",
    #                     hostname=hostname, port=port, desc="topology upgrade", fibres=1, bandwidth=50e9, 
    #                     throughput=False, max_time=3600*12, ILP_multi_fibre=False, k=5, insert=False, local=True,
    #                     graph_list=graph_list, ILP_threads=5, InS=True)