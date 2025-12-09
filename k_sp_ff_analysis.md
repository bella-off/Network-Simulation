# k_sp_ff.py 详细分析文档

## 1. 文件概述

`k_sp_ff.py` 是一个用于计算光网络吞吐量上界的仿真脚本。它使用 **k-Shortest Paths with First Fit** (kSP-FF) 路由算法来为网络拓扑分配路由和波长，然后计算网络的最大吞吐量。

### 主要功能
- 从 MongoDB 数据库读取网络拓扑数据
- 使用并行计算框架 Ray 进行分布式仿真
- 使用 kSP-FF 启发式算法进行路由和波长分配 (RWA)
- 计算并存储每个拓扑的最大吞吐量结果

---

## 2. 导入的包和模块

### 2.1 `NetworkToolkit as nt`
- **类型**: 自定义包
- **位置**: `networktoolbox/NetworkToolkit/`
- **用途**: 提供网络仿真所需的所有类和函数
- **主要模块**:
  - `nt.Database`: 数据库操作模块
  - `nt.NetworkSimulator`: 网络仿真模块

### 2.2 `numpy as np`
- **类型**: 第三方库
- **用途**: 数值计算和数组操作
- **主要使用场景**:
  - 矩阵运算（需求矩阵计算）
  - 数组操作和数值处理

---

## 3. 代码逐行分析

### 3.1 配置部分（第 6-12 行）

```python
collection = "topology-paper"
db = "Topology_Data"
hostname = "128.40.42.10"
port = 6379
```

**详细说明**:
- `collection`: MongoDB 集合名称，存储拓扑数据
- `db`: MongoDB 数据库名称
- `hostname`: Ray 集群的主机地址（用于分布式计算）
- `port`: Ray 集群的端口号（Redis 端口）

---

### 3.2 数据库查询配置（第 14 行）

```python
query = { "nodes" : 14, "ILP Capacity" : { "$exists" : True }, "FF-kSP RWA" : { "$exists" : False }}
```

**MongoDB 查询条件**:
- `"nodes": 14`: 查找节点数为 14 的拓扑
- `"ILP Capacity": { "$exists": True }`: 必须存在 "ILP Capacity" 字段
- `"FF-kSP RWA": { "$exists": False }`: "FF-kSP RWA" 字段不存在（表示尚未计算过）

**目的**: 筛选出满足条件且尚未处理过的拓扑

---

### 3.3 读取拓扑数据（第 15-17 行）

```python
graph_list = nt.Database.read_topology_dataset_list(db, collection,
                                                find_dic=query,
                                                node_data=True, max_count=1)
```

**调用的函数**: `nt.Database.read_topology_dataset_list()`

#### 3.3.1 `read_topology_dataset_list()` 函数详解

**位置**: `networktoolbox/NetworkToolkit/Database.py`

**函数签名**:
```python
def read_topology_dataset_list(db_name, collection_name, *args, find_dic=None,
                               node_data=False,
                               max_count=1000000, parralel=False, ...):
```

**参数说明**:
- `db_name`: 数据库名称
- `collection_name`: 集合名称
- `find_dic`: MongoDB 查询字典
- `node_data`: 是否读取节点数据（布尔值）
- `max_count`: 最大返回数量（这里设为 1，只读取一个拓扑）

**工作流程**:
1. **连接 MongoDB**:
   ```python
   client = pymongo.MongoClient('mongodb://localhost:{}'.format(port), 
                                username=user, password=pwd)
   ```

2. **查询数据**:
   ```python
   results = read_data(db_name, collection_name, *args, 
                       find_dic=find_dic, max_count=max_count, skip=skip)
   ```
   - 调用 `read_data()` 函数执行 MongoDB 查询
   - 返回满足条件的文档（最多 `max_count` 个）

3. **转换为 NetworkX 图**:
   ```python
   graph_list = [(Tools.read_database_topology(result[topology_data_field_name], 
                                              node_data=result["node data"]),
                  result["_id"]) 
                 for result in results]
   ```
   - 使用 `Tools.read_database_topology()` 将 MongoDB 文档转换为 NetworkX 图对象
   - 返回 `(graph, _id)` 元组列表

**返回值**: `[(graph, _id), ...]`
- `graph`: NetworkX 图对象（表示网络拓扑）
- `_id`: MongoDB 文档的唯一标识符

#### 3.3.2 依赖的底层函数

**`read_data()` 函数**:
- **位置**: `networktoolbox/NetworkToolkit/Database.py`
- **功能**: 执行 MongoDB 查询，返回 PyMongo 游标对象
- **实现**:
  ```python
  db = client[db_name]
  data = db[collection_name]
  find_results = data.find(find_dic).limit(max_count)
  return find_results
  ```

**`Tools.read_database_topology()` 函数**:
- **位置**: `networktoolbox/NetworkToolkit/Tools.py`
- **功能**: 将 MongoDB 存储的图数据转换为 NetworkX 图对象
- **处理的数据格式**: 字典格式的邻接表表示
- **返回**: `nx.Graph()` 对象

---

### 3.4 计算流量矩阵 T_c（第 20-22 行）

```python
matrix_one = np.ones((len(graph_list[0][0].nodes), len(graph_list[0][0].nodes)))
np.fill_diagonal(matrix_one, 0)
T_c = (matrix_one / (len(graph_list[0][0].nodes) * (len(graph_list[0][0].nodes) - 1))).tolist()
```

**详细步骤**:

1. **创建全 1 矩阵**:
   ```python
   matrix_one = np.ones((N, N))  # N 为节点数
   ```
   - 创建一个 N×N 的全 1 矩阵（表示所有节点对之间的连接）

2. **对角线置 0**:
   ```python
   np.fill_diagonal(matrix_one, 0)
   ```
   - 将对角线元素设为 0（节点不与自己连接）

3. **归一化**:
   ```python
   T_c = matrix_one / (N * (N-1))
   ```
   - 除以 `N * (N-1)`（所有可能的节点对数量）
   - 得到归一化的流量需求矩阵

**结果**: `T_c` 是一个归一化的流量需求矩阵
- `T_c[i][j]`: 节点 i 到节点 j 的归一化流量需求
- 所有元素之和为 1（均匀流量分布）

**数学意义**: 
- 表示所有节点对之间都有相同的流量需求
- `T_c[i][j] = 1/(N*(N-1))` (当 i ≠ j 时)

---

### 3.5 准备图列表（第 23 行）

```python
graph_list = [(graph, _id, T_c) for graph, _id in graph_list]
```

**操作**: 将每个 `(graph, _id)` 元组扩展为 `(graph, _id, T_c)` 元组

**数据结构变化**:
- **之前**: `[(graph1, _id1), (graph2, _id2), ...]`
- **之后**: `[(graph1, _id1, T_c), (graph2, _id2, T_c), ...]`

**目的**: 为每个图添加流量需求矩阵，供后续仿真使用

---

### 3.6 执行并行仿真（第 25-32 行）

```python
result = nt.NetworkSimulator.parralel_heuristic_throughput(
    graph_list, collection=collection, db=db,
    workers=len(graph_list),
    route_function="kSP-FF",
    e=100, k=5, m_step=200, channel_bandwidth=50e9,
    max_count=10,
    m_start=2000,
    port=port,
    hostname=hostname, fibre_num=1
)
```

**调用的函数**: `nt.NetworkSimulator.parralel_heuristic_throughput()`

#### 3.6.1 参数详解

| 参数 | 类型 | 值 | 说明 |
|------|------|-----|------|
| `graph_list` | list | `[(graph, _id, T_c), ...]` | 要处理的图列表 |
| `collection` | str | `"topology-paper"` | MongoDB 集合名称 |
| `db` | str | `"Topology_Data"` | MongoDB 数据库名称 |
| `workers` | int | `len(graph_list)` | 并行工作进程数 |
| `route_function` | str | `"kSP-FF"` | 路由算法名称 |
| `e` | int | `100` | 边数（用于 k-最短路径计算） |
| `k` | int | `5` | k-最短路径的数量 |
| `m_step` | int | `200` | M 值的步长 |
| `channel_bandwidth` | float | `50e9` | 每个信道的带宽（50 GHz） |
| `max_count` | int | `10` | 最大迭代次数 |
| `m_start` | int | `2000` | M 的起始值 |
| `port` | int | `6379` | Ray 集群端口 |
| `hostname` | str | `"128.40.42.10"` | Ray 集群主机地址 |
| `fibre_num` | int | `1` | 光纤数量 |

**关键参数说明**:
- **`route_function="kSP-FF"`**: 使用 k-最短路径优先，然后 First Fit 波长分配
- **`e=100`**: 用于限制 k-最短路径计算的边数
- **`k=5`**: 为每个源-目的对计算 5 条最短路径
- **`m_step=200`**: 在二分搜索中，每次增加或减少的 M 值步长
- **`channel_bandwidth=50e9`**: 每个波长信道的带宽为 50 GHz（标准 DWDM 信道带宽）
- **`m_start=2000`**: 开始时的 M 值（流量需求的倍数）

---

#### 3.6.2 `parralel_heuristic_throughput()` 函数详解

**位置**: `networktoolbox/NetworkToolkit/NetworkSimulator.py`

**函数签名**:
```python
def parralel_heuristic_throughput(graph_list, collection=None, db="Topology_Data", 
                                  workers=50, route_function="FF-kSP",
                                  e=10, k=10, m_step=100,
                                  channel_bandwidth=16e9, max_count=10, 
                                  m_start=0, port=6379,
                                  hostname="128.40.43.93", fibre_num=1):
```

**工作流程**:

1. **初始化 Ray 集群**:
   ```python
   ray.shutdown()
   if hostname is not None:
       ray.init(address='{}:{}'.format(hostname, port), 
                _redis_password='5241590000000000',
                ignore_reinit_error=True)
   else:
       ray.init()
   ```
   - 连接到远程 Ray 集群（分布式计算）
   - 使用 Redis 作为后端存储

2. **分配任务**:
   ```python
   indeces = Tools.create_start_stop_list(len(graph_list), workers)
   ```
   - **`Tools.create_start_stop_list()`**:
     - **位置**: `networktoolbox/NetworkToolkit/Tools.py`
     - **功能**: 将图列表分割成多个子列表，分配给不同的 worker
     - **示例**: 如果有 10 个图和 2 个 worker，返回 `[0, 5, 10]`
       - Worker 0 处理图 0-4
       - Worker 1 处理图 5-9

3. **创建进度条**:
   ```python
   pb = Tools.ProgressBar(workers)
   actor = pb.actor
   ```
   - **`Tools.ProgressBar` 类**:
     - **位置**: `networktoolbox/NetworkToolkit/Tools.py`
     - **功能**: 创建 Ray Actor 来跟踪仿真进度
     - **内部实现**:
       - 使用 `ProgressBarActor` Ray Actor
       - 每个 worker 完成任务时调用 `actor.update.remote(1)`

4. **提交远程任务**:
   ```python
   results = [heuristic_throughput.remote(
       db=db, collection=collection,
       graph_list=graph_list[indeces[ind]:indeces[ind + 1]],
       route_function=route_function, e=e, k=k, m_step=m_step,
       channel_bandwidth=channel_bandwidth,
       max_count=max_count, m_start=m_start, 
       fibre_num=fibre_num, pb_actor=actor
   ) for ind in range(workers)]
   ```
   - **`heuristic_throughput.remote()`**: Ray 远程函数装饰器
   - 每个 worker 处理分配给它的图子列表
   - 所有任务并行执行

5. **等待完成**:
   ```python
   pb.print_until_done()  # 显示进度条直到完成
   results = ray.get(results)  # 收集所有结果
   ```

---

### 3.7 `heuristic_throughput()` 函数详解

**位置**: `networktoolbox/NetworkToolkit/NetworkSimulator.py`

**函数签名**:
```python
def heuristic_throughput(graph_list=None, collection=None, db="Topology_Data", 
                         e=10, k=10, route_function="FF-kSP",
                         m_step=100, max_count=10, channel_bandwidth=16e9, 
                         m_start=0, fibre_num=1, pb_actor=None):
```

**核心算法**: **二分搜索 + RWA 算法**

#### 3.7.1 算法流程

```
1. 初始化网络
   ↓
2. 二分搜索找到最大可路由的 M 值
   ↓
3. 对于每个图：
   a. 创建 OpticalNetwork 对象
   b. 循环 max_count 次：
      - 尝试路由需求矩阵 T_c * M
      - 如果成功：增加 M
      - 如果失败：减少 M
      - 缩小搜索步长
   ↓
4. 计算物理层性能（吞吐量）
   ↓
5. 更新数据库
```

#### 3.7.2 详细步骤

**步骤 1: 初始化** (第 1310-1327 行)
```python
for graph, _id, T_c in graph_list:
    # 转换节点标签为整数（从 1 开始）
    graph = nx.relabel.convert_node_labels_to_integers(graph, first_label=1)
    
    # 验证图是连通图
    assert nx.is_connected(graph) is True
    
    # 创建光学网络对象
    network = Network.OpticalNetwork(
        graph, 
        channel_bandwidth=channel_bandwidth, 
        routing_func=route_function,
        fibre_num=fibre_num
    )
    
    # 初始化变量
    rwa_assignment = False  # RWA 分配结果
    M = m_start  # M 的初始值
    demand_matrix_old = np.zeros((len(graph), len(graph)))  # 上次成功的需求矩阵
    rwa_active = None  # 当前有效的 RWA 分配
    success = False  # 是否成功路由过
    rwa_list = []  # 存储所有的 RWA 分配历史
```

**步骤 2: 创建 OpticalNetwork 对象**

**`Network.OpticalNetwork` 类**:
- **位置**: `networktoolbox/NetworkToolkit/Network.py`
- **初始化参数**:
  - `graph`: NetworkX 图对象
  - `channel_bandwidth`: 信道带宽
  - `routing_func`: 路由函数名称
  - `fibre_num`: 光纤数量

**关键属性**:
```python
self.graph = graph  # 网络拓扑图
self.channels = int(B_o / channel_bandwidth)  # 波长信道数量
self.rwa = Router.RWA(graph, self.channels, channel_bandwidth)  # RWA 对象
self.physical_layer = PhysicalLayer.PhysicalLayer(...)  # 物理层对象

# 根据 routing_func 选择路由函数
if routing_func == "kSP-FF":
    self.route = self.rwa.k_SP_FF_revised
```

**步骤 3: 二分搜索循环** (第 1330-1380 行)

```python
for i in range(max_count):  # 最多迭代 max_count 次
    while rwa_assignment != True:  # 继续尝试直到成功
        M += m_step  # 增加 M 值
        demand_matrix_new = np.ceil(np.array(T_c) * M)  # 计算新的需求矩阵
        
        if route_function == "FF-kSP" or "kSP-FF":
            if not success:
                # 第一次尝试：路由增量需求
                alternate_demand = demand_matrix_new - demand_matrix_old
                connection_pairs = Tools.mat_to_pairs_list(alternate_demand)
                rwa_assignment = network.route(
                    demand_matrix_new - demand_matrix_old, 
                    e=e, k=k, 
                    connection_pairs=connection_pairs
                )
            elif (demand_matrix_new - demand_matrix_old).sum() > 0:
                # 增量路由（更高效）
                alternate_demand = demand_matrix_new - demand_matrix_old
                connection_pairs = Tools.mat_to_pairs_list(alternate_demand)
                rwa_assignment = network.route(
                    demand_matrix_new - demand_matrix_old, 
                    e=e, k=k,
                    rwa_assignment_previous=rwa_assignment,
                    connection_pairs=connection_pairs
                )
            
            if rwa_assignment != True:  # 成功
                rwa_active = rwa_assignment
                demand_matrix_old = demand_matrix_new
                success = True
                rwa_list.append(rwa_assignment)
        
    # 二分搜索调整
    if int(M) > 1:
        M -= m_step  # 回溯一步
    
    m_step /= 2  # 缩小步长
    m_step = np.ceil(m_step)
```

**算法逻辑**:
1. **外层循环**: 最多执行 `max_count` 次，逐步细化搜索
2. **内层循环**: 不断尝试增加 M 值，直到路由失败
3. **二分搜索**: 
   - 如果成功：增加 M，继续尝试更大的值
   - 如果失败：回溯，缩小步长，继续搜索
4. **增量路由**: 
   - 只路由新增的连接请求（更高效）
   - 复用之前的 RWA 分配结果

**步骤 4: 调用路由函数 `network.route()`**

**`network.route()` 的实现**:
- 当 `route_function="kSP-FF"` 时，`network.route = network.rwa.k_SP_FF_revised`

**`k_SP_FF_revised()` 函数**:
- **位置**: `networktoolbox/NetworkToolkit/Routing/Heuristics.py`
- **算法**: k-最短路径 + First Fit 波长分配

**工作流程**:
```
1. 获取所有源-目的对的需求
   ↓
2. 为每个源-目的对计算 k 条最短路径
   ↓
3. 对每个连接请求：
   a. 遍历 k 条路径
   b. 使用 First Fit 算法分配波长
   c. 如果找到可用波长：分配成功
   d. 如果所有路径都无法分配：分配失败
   ↓
4. 返回 RWA 分配结果
```

**关键函数 `FF_return()`**:
- **位置**: `networktoolbox/NetworkToolkit/Routing/Heuristics.py` (第 1921 行)
- **功能**: First Fit 波长分配
- **算法**:
  ```python
  def FF_return(self, graph, channels, W, path):
      # W: 波长占用矩阵 [edges × channels]
      #     W[edge_index, wavelength] = 1 表示占用，0 表示空闲
      
      # 1. 创建路径向量 P：标记路径经过哪些边
      P = [0, 0, 1, 0, 1, ...]  # P[edge_index] = 1 表示路径经过该边
      
      # 2. 计算：W × P（逐元素相乘，然后按波长求和）
      #    对于每个波长 j，计算 sum(W[:, j] * P)
      #    如果结果 == 0，说明该波长在路径的所有边上都空闲
      a = np.einsum('ij,ij->j', W, P)
      
      # 3. 找到所有可用波长（a[j] == 0）
      indeces = np.where(a == 0)
      
      # 4. 返回第一个可用波长（最小的索引）
      #    如果没有可用波长（indeces 为空），返回 channels + 1（失败标记）
      try:
          min_wave = np.min(indeces)
      except:
          return channels + 1  # 失败
      return min_wave
  ```

**返回值**:
- **成功**: RWA 分配字典 `{wavelength: [paths], ...}`
- **失败**: `True`（表示无法分配）

**步骤 5: 计算物理层性能** (第 1385-1405 行)

```python
time_taken = time.perf_counter() - time_start

# 添加均匀发射功率到链路
network.physical_layer.add_uniform_launch_power_to_links(network.channels)

if fibre_num > 1:
    # 多光纤情况
    rwa_assignment = Tools.single_to_multi_fibre_rwa(...)
    throughput = 0
    for i in range(fibre_num):
        network.physical_layer.add_wavelengths_to_links(rwa_assignment[i])
        network.physical_layer.add_non_linear_NSR_to_links()
        throughput += network.physical_layer.get_lightpath_capacities_PLI(...)[0]
else:
    # 单光纤情况
    network.physical_layer.add_wavelengths_to_links(rwa_assignment)
    network.physical_layer.add_non_linear_NSR_to_links()
    throughput = network.physical_layer.get_lightpath_capacities_PLI(rwa_assignment)[0]
```

**物理层计算**:
1. **添加发射功率**: 为每个波长分配均匀的发射功率
2. **添加波长到链路**: 将 RWA 分配结果应用到物理层模型
3. **计算非线性噪声**: 考虑非线性效应（自相位调制、交叉相位调制等）
4. **计算吞吐量**: 基于功率限制迭代（PLI）模型计算每个光路的容量

**`PhysicalLayer` 类**:
- **位置**: `networktoolbox/NetworkToolkit/PhysicalLayer/`
- **功能**: 模拟光网络物理层特性
- **关键方法**:
  - `add_wavelengths_to_links()`: 将波长分配应用到链路
  - `add_non_linear_NSR_to_links()`: 计算非线性噪声
  - `get_lightpath_capacities_PLI()`: 计算光路容量（考虑功率限制）

**步骤 6: 更新数据库** (第 1407-1420 行)

```python
Database.update_data_with_id(
    db, collection, _id,
    newvals={"$set": {
        "{} RWA".format(route_function): rwa_write,  # RWA 分配结果
        "{} Throughput".format(route_function): throughput,  # 吞吐量
        "{} M".format(route_function): M,  # 最大 M 值
        "{} Time".format(route_function): time_taken,  # 计算时间
        "{} RWA List".format(route_function): rwa_list,  # RWA 历史
        ...
    }}
)
```

**`Database.update_data_with_id()` 函数**:
- **位置**: `networktoolbox/NetworkToolkit/Database.py`
- **功能**: 更新 MongoDB 文档
- **操作**: 使用 `$set` 操作符添加新字段

---

## 4. 数据流图

```
MongoDB 数据库
    ↓ (查询)
read_topology_dataset_list()
    ↓
graph_list: [(graph, _id), ...]
    ↓ (添加 T_c)
graph_list: [(graph, _id, T_c), ...]
    ↓ (并行分发)
parralel_heuristic_throughput()
    ↓ (Ray 分布式计算)
heuristic_throughput() [Worker 0]
heuristic_throughput() [Worker 1]
...
    ↓ (每个 worker)
OpticalNetwork 对象
    ↓ (二分搜索 + RWA)
network.route() → k_SP_FF_revised()
    ↓
RWA 分配结果
    ↓ (物理层计算)
PhysicalLayer 计算吞吐量
    ↓ (更新数据库)
Database.update_data_with_id()
    ↓
MongoDB 更新完成
```

---

## 5. 关键类和函数总结

### 5.1 数据库相关

| 类/函数 | 位置 | 功能 |
|---------|------|------|
| `Database.read_topology_dataset_list()` | `Database.py` | 从 MongoDB 读取拓扑列表 |
| `Database.read_data()` | `Database.py` | 执行 MongoDB 查询 |
| `Database.update_data_with_id()` | `Database.py` | 更新 MongoDB 文档 |
| `Tools.read_database_topology()` | `Tools.py` | 将数据库数据转换为 NetworkX 图 |

### 5.2 网络仿真相关

| 类/函数 | 位置 | 功能 |
|---------|------|------|
| `NetworkSimulator.parralel_heuristic_throughput()` | `NetworkSimulator.py` | 并行吞吐量仿真 |
| `NetworkSimulator.heuristic_throughput()` | `NetworkSimulator.py` | 单个拓扑的吞吐量计算 |
| `Network.OpticalNetwork` | `Network.py` | 光网络对象 |
| `Heuristics.k_SP_FF_revised()` | `Heuristics.py` | kSP-FF 路由算法 |
| `Heuristics.FF_return()` | `Heuristics.py` | First Fit 波长分配 |

### 5.3 工具函数

| 函数 | 位置 | 功能 |
|------|------|------|
| `Tools.create_start_stop_list()` | `Tools.py` | 分配任务给 workers |
| `Tools.ProgressBar` | `Tools.py` | 进度条显示 |
| `Tools.mat_to_pairs_list()` | `Tools.py` | 矩阵转连接对列表 |
| `Tools.write_database_dict()` | `Tools.py` | 转换数据格式用于存储 |

### 5.4 物理层相关

| 类/函数 | 位置 | 功能 |
|---------|------|------|
| `PhysicalLayer.PhysicalLayer` | `PhysicalLayer/` | 物理层模型 |
| `PhysicalLayer.add_wavelengths_to_links()` | `PhysicalLayer/` | 添加波长到链路 |
| `PhysicalLayer.add_non_linear_NSR_to_links()` | `PhysicalLayer/` | 计算非线性噪声 |
| `PhysicalLayer.get_lightpath_capacities_PLI()` | `PhysicalLayer/` | 计算光路容量 |

---

## 6. 算法详解

### 6.1 kSP-FF 路由算法

**kSP-FF (k-Shortest Paths with First Fit)** 是一种启发式路由和波长分配算法。

**算法步骤**:

1. **计算 k-最短路径**:
   - 为每个源-目的对计算 k 条最短路径
   - 使用 Yen's 算法或类似方法

2. **排序连接请求**:
   - 按照某种顺序处理连接请求（例如：按路径长度排序）

3. **波长分配（First Fit）**:
   - 对于每个连接请求：
     - 遍历 k 条路径
     - 对于每条路径，使用 First Fit 算法查找可用波长
     - 选择第一条有可用波长的路径

4. **First Fit 算法**:
   - 从波长 0 开始，依次检查每个波长
   - 检查该波长在路径的所有边上是否都空闲
   - 如果空闲，立即分配（不继续查找更好的波长）

**时间复杂度**:
- k-最短路径计算: O(k * E * log V) (E: 边数, V: 节点数)
- First Fit 波长分配: O(k * W * E) (W: 波长数)
- 总体: O(N² * k * W * E) (N²: 连接请求数)

### 6.2 二分搜索算法

**目标**: 找到最大可路由的 M 值

**算法**:
```
初始化: M = m_start, m_step = 初始步长
for i in range(max_count):
    尝试路由 T_c * M
    while 路由成功:
        M += m_step  # 增加 M
        尝试路由 T_c * M
    M -= m_step  # 回溯
    m_step /= 2  # 缩小步长
```

**终止条件**:
- 达到最大迭代次数 `max_count`
- 或步长缩小到 1 且无法路由

---

## 7. 依赖关系图

```
k_sp_ff.py
├── NetworkToolkit
│   ├── Database
│   │   ├── read_topology_dataset_list()
│   │   │   ├── read_data() → MongoDB
│   │   │   └── Tools.read_database_topology()
│   │   └── update_data_with_id() → MongoDB
│   │
│   ├── NetworkSimulator
│   │   ├── parralel_heuristic_throughput()
│   │   │   ├── ray.init() → Ray 集群
│   │   │   ├── Tools.create_start_stop_list()
│   │   │   ├── Tools.ProgressBar()
│   │   │   └── heuristic_throughput.remote()
│   │   │
│   │   └── heuristic_throughput()
│   │       ├── Network.OpticalNetwork()
│   │       │   ├── Router.RWA()
│   │       │   └── PhysicalLayer.PhysicalLayer()
│   │       │
│   │       ├── network.route() → k_SP_FF_revised()
│   │       │   ├── get_k_shortest_paths_MNH()
│   │       │   └── FF_return()
│   │       │
│   │       ├── PhysicalLayer.add_wavelengths_to_links()
│   │       ├── PhysicalLayer.add_non_linear_NSR_to_links()
│   │       └── PhysicalLayer.get_lightpath_capacities_PLI()
│   │
│   └── Tools
│       ├── read_database_topology()
│       ├── create_start_stop_list()
│       ├── ProgressBar()
│       ├── mat_to_pairs_list()
│       └── write_database_dict()
│
├── numpy (第三方库)
├── pymongo (第三方库) → MongoDB
└── ray (第三方库) → 分布式计算
```

---

## 8. 执行流程示例

假设有以下输入:
- 节点数: 14
- 图数量: 1
- M 起始值: 2000
- M 步长: 200

**执行流程**:

```
1. 读取 MongoDB:
   查询条件: {nodes: 14, ILP Capacity exists, FF-kSP RWA 不存在}
   → 找到 1 个拓扑

2. 计算 T_c:
   T_c = 1/(14*13) ≈ 0.0055 (均匀流量)

3. 初始化 Ray:
   连接到 128.40.42.10:6379
   创建 1 个 worker

4. Worker 0 开始处理:
   
   Iteration 1:
   - M = 2000
   - demand_matrix = T_c * 2000
   - 尝试路由: 成功
   - M = 2200, 尝试路由: 成功
   - M = 2400, 尝试路由: 成功
   - ...
   - M = 3000, 尝试路由: 失败
   - M = 2800 (回溯)
   - m_step = 100
   
   Iteration 2:
   - M = 2900, 尝试路由: 成功
   - M = 3000, 尝试路由: 失败
   - M = 2900 (回溯)
   - m_step = 50
   
   ...
   
   Iteration 10:
   - M = 2930, 尝试路由: 成功
   - M = 2940, 尝试路由: 失败
   - M = 2930 (最终结果)

5. 计算物理层吞吐量:
   - 应用 RWA 分配
   - 计算非线性噪声
   - 计算光路容量
   - 总吞吐量 = sum(所有光路容量)

6. 更新 MongoDB:
   {
     "FF-kSP RWA": {...},
     "FF-kSP Throughput": 1234.56,
     "FF-kSP M": 2930,
     "FF-kSP Time": 45.67
   }
```

---

## 9. 输出结果

**MongoDB 文档更新字段**:
- `"kSP-FF RWA"`: RWA 分配结果（字典格式）
- `"kSP-FF Throughput"`: 计算得到的吞吐量（比特/秒）
- `"kSP-FF M"`: 最大可路由的 M 值
- `"kSP-FF Time"`: 计算耗时（秒）
- `"kSP-FF RWA List"`: RWA 分配历史列表

---

## 10. 总结

`k_sp_ff.py` 是一个完整的光网络吞吐量仿真系统，它：

1. **从 MongoDB 读取拓扑数据**
2. **使用 Ray 进行分布式并行计算**
3. **使用 kSP-FF 启发式算法进行路由和波长分配**
4. **通过二分搜索找到最大可路由的流量需求**
5. **计算物理层性能（考虑非线性效应）**
6. **将结果写回 MongoDB**

该系统可以高效地处理大量网络拓扑，并自动计算每个拓扑的最大吞吐量。

