"""
查看 "Topology_Data" 数据库中 "real" 集合的所有数据集名称
"""

import NetworkToolkit as nt
import pymongo

# 方法 1：使用 read_data 读取所有文档的 name 字段，然后提取唯一值
def method1_read_all_names():
    """方法 1：读取所有文档并提取 name 字段的唯一值"""
    print("=" * 60)
    print("方法 1：读取所有文档并提取 name 字段")
    print("=" * 60)
    
    # 读取所有文档，只获取 name 字段
    results = nt.Database.read_data(
        "Topology_Data", 
        "real", 
        find_dic={},  # 空字典表示查询所有文档
        max_count=10000  # 设置一个足够大的值
    )
    
    # 提取所有 name 字段的值
    names = []
    for doc in results:
        if "name" in doc:
            names.append(doc["name"])
    
    # 获取唯一值并排序
    unique_names = sorted(set(names))
    
    print(f"\n找到 {len(unique_names)} 个唯一的数据集名称：")
    print("-" * 60)
    for i, name in enumerate(unique_names, 1):
        count = names.count(name)
        print(f"{i}. {name} (共 {count} 个文档)")
    
    return unique_names


# 方法 2：直接使用 MongoDB 的 distinct 方法
def method2_distinct():
    """方法 2：使用 MongoDB 的 distinct 命令获取唯一的 name 值"""
    print("\n" + "=" * 60)
    print("方法 2：使用 MongoDB distinct 命令")
    print("=" * 60)
    
    # 直接连接 MongoDB
    user = "robin_admin"
    pwd = "Focker_12UCL!"
    port = 27017
    
    client = pymongo.MongoClient('mongodb://localhost:{}'.format(port), username=user, password=pwd)
    db = client["Topology_Data"]
    collection = db["real"]
    
    # 使用 distinct 获取唯一的 name 值
    unique_names = sorted(collection.distinct("name"))
    
    print(f"\n找到 {len(unique_names)} 个唯一的数据集名称：")
    print("-" * 60)
    for i, name in enumerate(unique_names, 1):
        # 统计每个名称的文档数量
        count = collection.count_documents({"name": name})
        print(f"{i}. {name} (共 {count} 个文档)")
    
    client.close()
    return unique_names


# 方法 3：查看更详细的统计信息
def method3_detailed_info():
    """方法 3：查看每个数据集的详细信息"""
    print("\n" + "=" * 60)
    print("方法 3：详细统计信息")
    print("=" * 60)
    
    user = "robin_admin"
    pwd = "Focker_12UCL!"
    port = 27017
    
    client = pymongo.MongoClient('mongodb://localhost:{}'.format(port), username=user, password=pwd)
    db = client["Topology_Data"]
    collection = db["real"]
    
    # 获取所有唯一的 name 值
    unique_names = sorted(collection.distinct("name"))
    
    print(f"\n数据集详细统计：")
    print("=" * 60)
    
    for name in unique_names:
        docs = list(collection.find({"name": name}, {"nodes": 1, "edges": 1, "name": 1}))
        if len(docs) > 0:
            print(f"\n数据集名称: {name}")
            print(f"  文档数量: {len(docs)}")
            if "nodes" in docs[0]:
                nodes_list = [doc.get("nodes") for doc in docs if "nodes" in doc]
                if nodes_list:
                    print(f"  节点数范围: {min(nodes_list)} - {max(nodes_list)}")
            if "edges" in docs[0]:
                edges_list = [doc.get("edges") for doc in docs if "edges" in doc]
                if edges_list:
                    print(f"  边数范围: {min(edges_list)} - {max(edges_list)}")
    
    client.close()


if __name__ == "__main__":
    # 运行所有方法
    print("\n查看 'Topology_Data' 数据库中 'real' 集合的所有数据集")
    print("=" * 60)
    
    # 方法 1
    names1 = method1_read_all_names()
    
    # 方法 2（推荐，最快）
    names2 = method2_distinct()
    
    # 方法 3（详细统计）
    method3_detailed_info()
    
    print("\n" + "=" * 60)
    print("完成！")
    print("=" * 60)

