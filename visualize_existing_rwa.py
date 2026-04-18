

import NetworkToolkit as nt
import numpy as np
from plot_rwa import plot_rwa_heatmap


if __name__ == "__main__":
    db = "Topology_Data"
    collection = "topology-paper"
    route_function = "FF-kSP"  #
    
    graph_list = nt.Database.read_topology_dataset_list(
        "Topology_Data", "real",
        find_dic={"name": "NSFNET"},
        node_data=True,
        max_count=1
    )
    

    graph, graph_id = graph_list[0][0], graph_list[0][1]


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
                    break
        except Exception as e:
            continue
    
    if rwa_result is None:
        exit(1)
    

    
    plot_rwa_heatmap(
        graph,
        rwa_result,
        title=f"NSFNET with {route_function} Routing",
        save_path=f"rwa_{route_function}_nsfnet.png"
    )
    

