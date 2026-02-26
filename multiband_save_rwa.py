import NetworkToolkit as nt
from pymongo import MongoClient
import datetime

def copy_rwa_to_band_field(db_name, collection_name, band_selection, find_dic=None):
    """
    将 "ILP-connections RWA" 字段复制到带频段名的字段

    Parameters:
    -----------
    db_name : str
        数据库名称
    collection_name : str
        集合名称
    band_selection : str
        频段选择（"C", "CL", "SCL", "SCLO"）
    find_dic : dict, optional
        查询条件，默认查找所有有 "ILP-connections RWA" 的文档
    """
    if find_dic is None:
        find_dic = {"ILP-connections RWA": {"$exists": True}}
    else:
        find_dic["ILP-connections RWA"] = {"$exists": True}

    # 目标字段名
    target_field = f"ILP-connections RWA {band_selection}"
    print(target_field)
    # 读取数据
    print(f"Reading documents from {db_name}.{collection_name}...")
    results = list(nt.Database.read_data(
        db_name,
        collection_name,
        find_dic=find_dic,
        max_count=100000
    ))

    print(f"Found {len(results)} documents")

    # 复制字段
    updated_count = 0
    for result in results:
        _id = result["_id"]

        # 检查源字段是否存在
        if "ILP-connections RWA" not in result:
            print(f"Warning: Document {_id} does not have 'ILP-connections RWA' field. Skipping...")
            continue

        # 复制 RWA 字段
        rwa_data = result["ILP-connections RWA"]

        # 准备更新的数据
        update_data = {
            "$set": {
                target_field: rwa_data,
                f"{target_field} timestamp": datetime.datetime.utcnow(),
                # 同时复制其他相关字段
                f"{target_field} band": band_selection,
            }
        }

        # 如果存在其他相关字段，也复制
        fields_to_copy = [
            "ILP-connections",
            "ILP-connections Capacity",
            "ILP-connections gap",
            "ILP-connections status",
            "ILP-connections e",
            "ILP-connections k",
            "ILP-connections channel bandwidth",
            "ILP-connections channels",
            "ILP-connections routing channels",
            "ILP-connections fibre number",
            "ILP-connections blocking rate",
            "ILP-connections M",
        ]

        for field in fields_to_copy:
            if field in result:
                # 创建带频段名的字段名
                new_field_name = field.replace("ILP-connections", f"ILP-connections {band_selection}")
                update_data["$set"][new_field_name] = result[field]

        # 更新数据库
        try:
            nt.Database.update_data_with_id(
                db_name,
                collection_name,
                _id,
                newvals=update_data
            )
            updated_count += 1
            print(f"Updated document {_id} with {target_field}")
        except Exception as e:
            print(f"Error updating document {_id}: {e}")

    print(f"\n{'='*60}")
    print(f"Summary for {band_selection} band:")
    print(f"  Total documents found: {len(results)}")
    print(f"  Successfully updated: {updated_count}")
    print(f"  Target field: {target_field}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    # 配置
    db_name = "Topology_Data"
    # Configuration: Change topology_name to switch between different topologies
    topology_name = "DTAG"  # Options: "NSFNET", "DTAG", "CONUS", etc.
    collection_name = "real" if topology_name == "DTAG" else "topology-paper"  # DTAG is in "real", others in "topology-paper"
    # 定义要处理的频段  ["C", "CL", "SCL", "SCLO"]
    band = "SCLO"  # 可以修改为只处理特定频段
    # 查询条件
    find_dic = {"name": topology_name}

    print("="*60)
    print("RWA Field Copy Script")
    print("="*60)
    print(f"Database: {db_name}")
    print(f"Collection: {collection_name}")
    print(f"Query: {find_dic}")
    print("="*60)

    # 处理每个频段
        print(f"\nProcessing {band} band...")
        copy_rwa_to_band_field(
            db_name=db_name,
            collection_name=collection_name,
            band_selection=band,
            find_dic=find_dic.copy()  # 复制以避免修改原字典
        )

    print("="*60)
    print("All bands processed!")
    print("="*60)