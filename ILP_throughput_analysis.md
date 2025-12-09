# ILP_throughput 函数详细分析

## 目录
1. [函数概述](#函数概述)
2. [函数签名与参数](#函数签名与参数)
3. [函数执行流程](#函数执行流程)
4. [核心函数分析](#核心函数分析)
5. [数据流分析](#数据流分析)
6. [关键算法与约束](#关键算法与约束)
7. [依赖关系图](#依赖关系图)

---

## 函数概述

`ILP_throughput` 是一个用于计算光网络最大吞吐量的函数，它使用整数线性规划（ILP）方法来优化路由和波长分配（RWA），以最大化网络的统一带宽需求。该函数是网络仿真工具包中的核心函数之一，用于评估光网络的性能上限。

### 主要功能
- 对给定的网络拓扑图，使用ILP求解器（Gurobi）计算最大吞吐量
- 考虑物理层约束（非线性噪声、信号功率等）
- 支持单光纤和多光纤网络配置
- 将结果存储到数据库中

---

## 函数签名与参数

### 函数位置
`thesis/ptd/networktoolbox/NetworkToolkit/NetworkSimulator.py` (行1469-1570)

### 函数签名
```python
def ILP_throughput(graph_list=None, max_time=1000, collection=None, db="Topology_Data",
                   threads=10, channel_bandwidth=16e9, e=0, k=20, node_file_start=0.01,
                   fibre_num=1, weighted=None,
                   actor=None, capacity_constraint=False,
                   max_solutions=100):
```

### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `graph_list` | list | None | 包含(graph, _id)元组的列表，每个元组包含网络图和唯一标识符 |
| `max_time` | int | 1000 | ILP求解器的最大运行时间（秒） |
| `collection` | str | None | MongoDB集合名称 |
| `db` | str | "Topology_Data" | MongoDB数据库名称 |
| `threads` | int | 10 | Gurobi求解器使用的线程数 |
| `channel_bandwidth` | float | 16e9 | 每个信道的带宽（Hz），默认16 GHz |
| `e` | int | 0 | k最短路径算法的额外路径长度参数 |
| `k` | int | 20 | k最短路径算法中考虑的路径数量 |
| `node_file_start` | float | 0.01 | Gurobi节点文件开始阈值 |
| `fibre_num` | int | 1 | 光纤数量（支持多光纤网络） |
| `weighted` | str/None | None | 路径权重类型（如'weight'） |
| `actor` | object | None | Ray actor对象，用于进度更新 |
| `capacity_constraint` | bool | False | 是否启用容量约束 |
| `max_solutions` | int | 100 | 最大解的数量 |

---

## 函数执行流程

### 1. 环境初始化阶段

```python
# 根据主机名设置Gurobi许可证文件路径
if socket.gethostname() == "MacBook-Pro":
    os.environ['GRB_LICENSE_FILE'] = "/Users/robin/Documents/network-code/gurobi-licence/gurobi.lic"
    node_file_dir = "/Users/robin/Documents/network-code/nodefiles"
else:
    os.environ['GRB_LICENSE_FILE'] = "/home/uceeatz/gurobi.lic"
    node_file_dir = "/rdata/ong/robin/nodefiles/"
```

**功能**：根据运行环境（笔记本电脑或服务器）设置Gurobi求解器的许可证文件和节点文件目录。

### 2. 主循环处理每个网络图

```python
for graph, _id in graph_list:
    # 处理每个网络图
```

对输入列表中的每个网络图执行以下步骤：

#### 2.1 图预处理
```python
graph = nx.relabel.convert_node_labels_to_integers(graph, first_label=1)
assert type(graph) == nx.classes.graph.Graph
```

- 将节点标签重新编号为从1开始的整数
- 验证图类型为NetworkX Graph对象

#### 2.2 创建光网络对象
```python
network = Network.OpticalNetwork(graph, channel_bandwidth=channel_bandwidth, fibre_num=1)
network.routing_channels = fibre_num * network.channels
```

**调用的类**：`Network.OpticalNetwork`

**功能**：
- 创建光网络对象，包含：
  - 图结构（`self.graph`）
  - 信道数量（`self.channels`）
  - 信道带宽（`self.channel_bandwidth`）
  - RWA对象（`self.rwa`）
  - 物理层对象（`self.physical_layer`）
- 设置路由信道数量 = 光纤数 × 每光纤信道数

#### 2.3 调用ILP求解函数
```python
data = network.rwa.maximise_uniform_bandwidth_demand(
    max_time=max_time,
    e=e,
    k=k, 
    _id=_id,
    threads=threads,
    node_file_dir=node_file_dir,
    node_file_start=node_file_start,
    c_type="I",
    solution_file_dir=None,
    capacity_constraint=capacity_constraint,
    verbose=0, 
    weighted=weighted,
    emphasis=2, 
    max_solutions=max_solutions, 
    channels=network.channels, 
    fibre_num=fibre_num
)
```

**核心函数**：`maximise_uniform_bandwidth_demand`（详见下文分析）

**返回值**：包含以下字段的字典
- `rwa`: 路由和波长分配结果
- `objective`: ILP目标函数值（最大吞吐量）
- `status`: 求解器状态
- `gap`: 最优性间隙

#### 2.4 处理求解结果

##### 情况1：目标值为0（无可行解）
```python
if data["objective"] == 0:
    max_capacity = [0]
    node_pair_capacities = 0
    rwa_assignment = None
```

##### 情况2：多光纤网络（fibre_num > 1）
```python
if fibre_num > 1:
    # 转换RWA分配为多光纤格式
    rwa_assignment = Tools.convert_rwa_to_multi_fibre(
        data["rwa"], fibre_num, network.routing_channels
    )
    
    max_capacity = 0
    for i in range(fibre_num):
        # 为每个光纤计算容量
        network.physical_layer.add_uniform_launch_power_to_links(int(network.channels))
        network.physical_layer.add_wavelengths_to_links(rwa_assignment[i])
        network.physical_layer.add_non_linear_NSR_to_links(
            channel_bandwidth=channel_bandwidth, 
            channels_full=int(network.channels)
        )
        max_capacity += network.physical_layer.get_lightpath_capacities_PLI(rwa_assignment[i])[0]
```

**关键步骤**：
1. **转换RWA格式**：将单光纤RWA分配转换为多光纤格式
2. **添加物理层属性**：
   - 统一发射功率
   - 波长分配
   - 非线性噪声（NSR）
3. **计算容量**：对每个光纤计算光路容量并累加

##### 情况3：单光纤网络（fibre_num == 1）
```python
else:
    network.physical_layer.add_uniform_launch_power_to_links(network.channels)
    network.physical_layer.add_wavelengths_to_links(data["rwa"])
    network.physical_layer.add_non_linear_NSR_to_links(
        channels_full=network.channels,
        channel_bandwidth=network.channel_bandwidth
    )
    max_capacity = network.physical_layer.get_lightpath_capacities_PLI(data["rwa"])[0]
    node_pair_capacities = {str(key): value for key, value in max_capacity[2].items()}
    rwa_assignment = Tools.write_database_dict(data["rwa"])
```

#### 2.5 更新数据库
```python
Database.update_data_with_id(db, collection, _id, newvals={
    "$set": {
        "ILP-throughput RWA": rwa_assignment,
        "ILP-throughput": float(data["objective"]),
        "ILP-throughput Capacity": max_capacity,
        "ILP-throughput status": int(data["status"].value),
        "ILP-throughput gap": float(data["gap"]),
        "ILP-throughput e": e,
        "ILP-throughput weighted": weighted,
        "ILP-throughput k": int(k),
        "ILP-throughput channel number": int(network.channels/fibre_num),
        "ILP-throughput channel bandwidth": float(network.channel_bandwidth*fibre_num),
        "ILP-throughput fibre number": int(fibre_num),
        "ILP-throughput timestamp Capacity": datetime.datetime.utcnow(),
        "ILP-throughput capacity constraint": False,
        "ILP-throughput emphasis": 2
    }
})
```

---

## 核心函数分析

### 1. `maximise_uniform_bandwidth_demand`

**位置**：`thesis/ptd/networktoolbox/NetworkToolkit/Routing/ILP.py` (行2152-2317)

**功能**：使用ILP方法最大化统一带宽需求，这是整个吞吐量计算的核心优化函数。

#### 1.1 函数签名
```python
def maximise_uniform_bandwidth_demand(self, e=1, k=5,
                                      solver_name="GRB", max_time=1000, T=0,
                                      shortest_paths_only=False, capacity_constraint=False,
                                      threads=4, node_file_start=0.1, collection="ilp-test", db="Topology_Data",
                                      _id=0, emphasis=0, node_file_dir="/scratch/datasets/gurobi/nodefiles",
                                      solution_file_dir="/scratch/datasets/gurobi/solutions/{}",
                                      c_type="C",
                                      verbose=0, weighted=None,
                                      mip_gap=1e-4, max_solutions=100, 
                                      channels=100, fibre_num=1):
```

#### 1.2 执行步骤

##### 步骤1：计算k最短路径
```python
k_SP = Tools.get_k_shortest_paths_MNH(self.graph, e=e, k=k, weighted=weighted)
```

**调用的函数**：`Tools.get_k_shortest_paths_MNH`

**功能**：
- 为所有节点对计算k条最短路径
- 考虑参数`e`：允许路径长度比最短路径长`e`跳
- 返回格式：`[((s,d), [path1, path2, ...]), ...]`

**算法**：使用Yen's算法或类似方法计算k最短路径

##### 步骤2：初始化物理层并计算SNR
```python
pl = PhysicalLayer(self.graph, channels, self.channel_bandwidth)
pl.add_wavelengths_full_occupation(channels_full=channels)
pl.add_uniform_launch_power_to_links(channels)
pl.add_non_linear_NSR_to_links(
    channels_full=channels,
    channel_bandwidth=self.channel_bandwidth
)
```

**功能**：
- 创建物理层对象
- 假设所有信道都被占用（最坏情况）
- 添加统一的发射功率
- 计算非线性噪声（NSR）

**SNR计算**：
```python
if shortest_paths_only:
    SNR_list = pl.get_SNR_shortest_path_node_pair(channels, unique_paths)
else:
    SNR_list = pl.get_SNR_k_SP(channels, k_SP)
```

##### 步骤3：计算路径容量
```python
C = [[float(self.channel_bandwidth*np.log2(1+SNR_list[i][1][k])) 
      for k in range(len(SNR_list[i][1]))] 
     for i in range(len(SNR_list))]
```

**公式**：使用香农容量公式
```
C = B × log₂(1 + SNR)
```
其中：
- `B` = 信道带宽
- `SNR` = 信噪比

##### 步骤4：构建ILP模型

**变量定义**：
```python
# 二进制决策变量：delta[i][k][w] = 1 表示节点对i使用路径k和波长w
delta = [[[ILP.add_var(var_type="B", name='rwa-w-{}-k-{}-z-{}'.format(w,p,k)) 
           for w in range(W)] 
          for p in range(len(k_SP[k][1]))] 
         for k in range(K)]

# 连续变量：c 表示统一带宽需求（目标函数）
c = ILP.add_var(var_type=c_type)
```

**参数**：
- `W` = 波长数量 = `channels * fibre_num`
- `K` = 节点对数量
- `E` = 边数量

**路径-边关联矩阵**：
```python
delta_i = [[[1 if (edges[_e] in Tools.nodes_to_edges(k_SP[i][1][k]) 
                   or (edges[_e][1], edges[_e][0]) in Tools.nodes_to_edges(k_SP[i][1][k])) 
             else 0 
             for _e in range(E)] 
            for k in range(len(k_SP[i][1]))] 
           for i in range(len(k_SP))]
```

**功能**：`delta_i[i][k][e] = 1` 表示节点对i的路径k经过边e

##### 步骤5：添加约束

**约束1：波长冲突约束**
```python
for _e in range(E):
    for w in range(W):
        ILP.add_constr(
            mip.xsum(delta[i][k][w]*delta_i[i][k][_e] 
                     for i in range(len(k_SP)) 
                     for k in range(len(k_SP[i][1]))) <= 1
        )
```

**含义**：每条边上的每个波长最多只能被一条光路使用（避免波长冲突）

**约束2：带宽需求约束**
```python
for i in range(len(k_SP)):
    ILP.add_constr(
        mip.xsum(delta[i][k][w] * C[i][k] 
                 for k in range(len(k_SP[i][1])) 
                 for w in range(W)) >= c*T[i]
    )
```

**含义**：每个节点对i的带宽需求必须至少为`c × T[i]`，其中：
- `c` = 统一带宽需求（优化变量）
- `T[i]` = 节点对i的需求权重（通常为1/N，N为节点对数）

**约束3：容量约束（可选）**
```python
if capacity_constraint:
    for i in range(len(k_SP)):
        ILP.add_constr(
            mip.xsum(delta[i][k][w] * C[i][k] - c*T[i] 
                     for k in range(len(k_SP[i][1])) 
                     for w in range(W)) <= max(C[i])
        )
```

##### 步骤6：设置目标函数
```python
ILP.objective = c
ILP.sense = "MAX"
```

**目标**：最大化统一带宽需求`c`

##### 步骤7：求解ILP
```python
ILP.emphasis = emphasis
status = ILP.optimize(max_seconds=max_time, max_solutions=max_solutions)
```

**求解器**：Gurobi (GRB)

**参数**：
- `emphasis=2`：强调寻找可行解
- `max_time`：最大求解时间
- `max_solutions`：最大解数量

##### 步骤8：提取结果
```python
rwa_assignment = self.convert_delta_to_rwa_assignment(delta, k_SP, W)
data = {
    "rwa": rwa_assignment,
    "objective": ILP.objective_values[0],
    "status": status,
    "gap": ILP.gap
}
```

**调用的函数**：`convert_delta_to_rwa_assignment`（详见下文）

---

### 2. `convert_delta_to_rwa_assignment`

**位置**：`thesis/ptd/networktoolbox/NetworkToolkit/Routing/ILP.py` (行2907-2934)

**功能**：将ILP求解器返回的二进制决策变量`delta`转换为路由和波长分配（RWA）字典格式。

#### 函数实现
```python
def convert_delta_to_rwa_assignment(self, delta, k_SP, W, numeric_delta=False, dict_data=False):
    if not dict_data:
        if not numeric_delta:
            # 从ILP变量中提取值
            delta_int = [
                [[delta[i][k][w].x for w in range(W)] 
                 for k in range(len(k_SP[i][1]))]
                for i in range(len(k_SP))
            ]
        else:
            delta_int = delta
        
        # 构建RWA字典：{wavelength: [path1, path2, ...]}
        rwa_assignment = {w: [] for w in range(W)}
        for i in range(len(k_SP)):
            for k in range(len(k_SP[i][1])):
                for w in range(W):
                    if delta_int[i][k][w] == 1:
                        rwa_assignment[w].append(k_SP[i][1][k])
    
    return rwa_assignment
```

**输入**：
- `delta`: ILP二进制变量矩阵
- `k_SP`: k最短路径列表
- `W`: 波长数量

**输出**：
- `rwa_assignment`: 字典，格式为 `{wavelength: [path1, path2, ...]}`

**示例**：
```python
rwa_assignment = {
    0: [[1, 2, 3], [4, 5, 6]],  # 波长0上的两条光路
    1: [[1, 3, 5]],              # 波长1上的一条光路
    ...
}
```

---

### 3. `get_k_shortest_paths_MNH`

**位置**：`thesis/ptd/networktoolbox/NetworkToolkit/Routing/Tools.py` (行480-532)

**功能**：为图中所有节点对计算k条最短路径（Minimum Number of Hops, MNH）。

#### 函数签名
```python
def get_k_shortest_paths_MNH(graph, e=None, k=1, weighted=None, data_dict=False, ...):
```

#### 实现逻辑
```python
if not data_dict:
    k_sp = []
    # 遍历所有节点对
    for i in range(1, len(graph)+1):
        for j in range(1, len(graph)+1):
            if j < i:  # 跳过已处理的节点对
                pass
            elif j == i:  # 跳过自环
                pass
            else:
                # 计算节点i到j的k条最短路径
                k_sp_paths = k_shortest_paths(graph, i, j, k, weight=weighted)
                k_sp.append(((i,j), k_sp_paths))
    
    # 如果指定了e，过滤路径长度
    if e is not None:
        k_sp = list(map(
            lambda k_sp: ((k_sp[0][0], k_sp[0][1]), 
                         list(filter(
                             lambda x: True if len(x) <= len(min(k_sp[1], key=lambda x: len(x)))+e 
                                      else False, 
                             k_sp[1]
                         ))), 
            k_sp
        ))
```

**算法**：使用Yen's k最短路径算法

**返回格式**：
```python
[((1,2), [[1,2], [1,3,2]]),  # 节点对(1,2)的两条路径
 ((1,3), [[1,3], [1,2,3]]),  # 节点对(1,3)的两条路径
 ...]
```

---

### 4. PhysicalLayer 相关函数

#### 4.1 `add_uniform_launch_power_to_links`

**位置**：`thesis/ptd/networktoolbox/NetworkToolkit/PhysicalLayer.py` (行635-649)

**功能**：为所有链路添加统一的发射功率。

```python
def add_uniform_launch_power_to_links(self, channels):
    graph = self.graph
    launch_powers = {}
    graph_edges = graph.edges()
    for edge in graph_edges:
        # 为每个信道设置发射功率为1mW (10^-3 W)
        launch_powers[edge] = {"launch_powers": [(10 ** (-3))] * channels}
        launch_powers[(edge[1], edge[0])] = {"launch_powers": [(10 ** (-3))] * channels}
    nx.set_edge_attributes(graph, launch_powers)
    return graph
```

**参数**：
- `channels`: 信道数量

**功率值**：默认1 mW (10⁻³ W)

#### 4.2 `add_wavelengths_to_links`

**位置**：`thesis/ptd/networktoolbox/NetworkToolkit/PhysicalLayer.py` (行600-622)

**功能**：将RWA分配结果添加到图的边属性中。

```python
def add_wavelengths_to_links(self, wavelengths_dict):
    graph = self.graph
    wavelengths = {}
    graph_edges = graph.edges()
    
    # 初始化所有边的波长列表
    for edge in graph_edges:
        wavelengths[edge] = {"wavelengths": []}
        wavelengths[(edge[1], edge[0])] = {"wavelengths": []}
    
    # 根据RWA分配添加波长
    for key in wavelengths_dict:  # key = 波长编号
        for path in wavelengths_dict[key]:  # path = 节点路径
            edges = self.nodes_to_edges(path)  # 将路径转换为边列表
            for edge in edges:
                wavelengths[edge]["wavelengths"].append(key)
                wavelengths[(edge[1], edge[0])]["wavelengths"].append(key)
    
    nx.set_edge_attributes(graph, wavelengths)
    return graph
```

**输入格式**：`{wavelength: [path1, path2, ...]}`

**功能**：在图的每条边上记录使用该边的所有波长编号

#### 4.3 `add_non_linear_NSR_to_links`

**位置**：`thesis/ptd/networktoolbox/NetworkToolkit/PhysicalLayer.py` (行685-717)

**功能**：计算并添加非线性噪声（NSR - Noise-to-Signal Ratio）到链路。

```python
def add_non_linear_NSR_to_links(self, Plot=False, channels_full=156, channel_bandwidth=float(16e9)):
    graph = self.graph
    NSR = {}
    graph_edges = graph.edges()
    
    for edge in graph_edges:
        try:
            NSR[edge] = {
                "NSR": self.get_SNR_non_linear(
                    edge, graph, 
                    channels_full=channels_full,
                    channel_bandwidth=self.channel_bandwidth,
                    name="SNR_{}_{}".format(edge[0], edge[1])
                )
            }
            NSR[(edge[1], edge[0])] = {
                "NSR": self.get_SNR_non_linear(
                    edge, graph, 
                    channels_full=channels_full,
                    channel_bandwidth=self.channel_bandwidth
                )
            }
        except Exception as err:
            traceback.print_exc()
        nx.set_edge_attributes(graph, NSR)
    
    return graph
```

**调用的函数**：`get_SNR_non_linear`

**功能**：
- 计算每个信道的非线性噪声
- 考虑交叉相位调制（XPM）和四波混频（FWM）等非线性效应
- 返回每个信道的NSR值列表

**NSR计算**：
```
NSR = (P_ASE + P_NLI) / P_signal
```
其中：
- `P_ASE` = 放大自发辐射噪声
- `P_NLI` = 非线性干扰功率
- `P_signal` = 信号功率

#### 4.4 `get_lightpath_capacities_PLI`

**位置**：`thesis/ptd/networktoolbox/NetworkToolkit/PhysicalLayer.py` (行768-815)

**功能**：根据RWA分配和物理层参数计算每条光路的容量。

```python
def get_lightpath_capacities_PLI(self, wavelengths_dict, optim_snr=False):
    node_pair_capacities = {}
    graph = self.graph
    channel_bandwidth = self.channel_bandwidth
    capacity_total = 0
    n = 0
    capacity_matrix = [[0 for j in graph.nodes()] for i in graph.nodes()]
    
    for key in wavelengths_dict:  # key = 波长编号
        for path in wavelengths_dict[key]:  # path = 节点路径
            n += 1
            edges = self.nodes_to_edges(path)  # 转换为边列表
            
            # 计算路径的总NSR
            NSR = 0
            for edge in edges:
                index = graph[edge[0]][edge[1]]["wavelengths"].index(key)
                if optim_snr:
                    NSR += graph[edge[0]][edge[1]]["NSR_opt"][index]
                else:
                    NSR += graph[edge[0]][edge[1]]["NSR"][index]
            
            # 计算SNR和容量
            SNR = 1 / NSR
            capacity = 2 * channel_bandwidth * np.log2(1 + SNR)
            
            # 累加节点对容量
            if (path[0], path[-1]) in node_pair_capacities.keys():
                node_pair_capacities[(path[0], path[-1])] += capacity
            else:
                node_pair_capacities[(path[0], path[-1])] = capacity
            
            capacity_total += capacity
            capacity_matrix[path[0] - 1][path[-1] - 1] += capacity
    
    capacity_average = capacity_total / n
    return capacity_total, capacity_average, node_pair_capacities
```

**容量计算公式**：
```
SNR = 1 / NSR_total
C = 2 × B × log₂(1 + SNR)
```

其中：
- `NSR_total` = 路径上所有边的NSR之和
- `B` = 信道带宽
- 系数2表示双向传输

**返回值**：
1. `capacity_total`: 总容量（所有光路容量之和）
2. `capacity_average`: 平均容量（总容量/光路数）
3. `node_pair_capacities`: 每个节点对的容量字典

---

### 5. Tools 辅助函数

#### 5.1 `convert_rwa_to_multi_fibre`

**位置**：`thesis/ptd/networktoolbox/NetworkToolkit/Tools.py` (行76-81)

**功能**：将单光纤RWA分配转换为多光纤格式。

```python
def convert_rwa_to_multi_fibre(rwa, num_fibres, num_channels):
    rwa_multi_fibre = [{} for _ in range(num_fibres)]
    for wavelength, paths in rwa.items():
        fibre_index = wavelength // num_channels
        new_wavelength = wavelength % num_channels
        rwa_multi_fibre[fibre_index][new_wavelength] = paths
    return rwa_multi_fibre
```

**转换逻辑**：
- 将波长编号按光纤分组
- 每个光纤内的波长重新编号为0到num_channels-1

**示例**：
```python
# 输入：单光纤，4个波长
rwa = {0: [path1], 1: [path2], 2: [path3], 3: [path4]}

# 输出：2个光纤，每个2个波长
rwa_multi_fibre = [
    {0: [path1], 1: [path2]},  # 光纤0
    {0: [path3], 1: [path4]}   # 光纤1
]
```

#### 5.2 `write_database_dict`

**功能**：将RWA字典转换为可存储到数据库的格式（通常将键转换为字符串）。

**用途**：MongoDB等数据库要求字典键为字符串类型

---

## 数据流分析

### 输入数据流

```
graph_list (包含多个网络图)
    ↓
[对每个图]
    ↓
graph (NetworkX Graph对象)
    ↓
Network.OpticalNetwork(graph, ...)
    ↓
network对象 (包含graph, rwa, physical_layer等)
```

### ILP求解数据流

```
network.graph
    ↓
Tools.get_k_shortest_paths_MNH()
    ↓
k_SP (k最短路径列表)
    ↓
PhysicalLayer计算SNR
    ↓
计算路径容量C
    ↓
构建ILP模型 (变量delta, 约束, 目标函数c)
    ↓
Gurobi求解器
    ↓
delta变量值
    ↓
convert_delta_to_rwa_assignment()
    ↓
rwa_assignment (RWA字典)
```

### 容量计算数据流

```
rwa_assignment
    ↓
[单光纤或多光纤处理]
    ↓
PhysicalLayer.add_wavelengths_to_links()
    ↓
PhysicalLayer.add_uniform_launch_power_to_links()
    ↓
PhysicalLayer.add_non_linear_NSR_to_links()
    ↓
PhysicalLayer.get_lightpath_capacities_PLI()
    ↓
max_capacity (总容量)
node_pair_capacities (节点对容量)
```

### 输出数据流

```
计算结果
    ↓
Tools.write_database_dict()
    ↓
数据库格式的RWA
    ↓
Database.update_data_with_id()
    ↓
MongoDB数据库
```

---

## 关键算法与约束

### ILP优化模型

#### 决策变量
- `delta[i][k][w] ∈ {0,1}`: 二进制变量，表示节点对i是否使用路径k和波长w
- `c ≥ 0`: 连续变量，表示统一带宽需求

#### 目标函数
```
maximize c
```

#### 约束条件

**约束1：波长冲突约束**
```
∑(i,k) delta[i][k][w] × delta_i[i][k][e] ≤ 1,  ∀e, w
```
确保每条边上的每个波长最多被一条光路使用。

**约束2：带宽需求约束**
```
∑(k,w) delta[i][k][w] × C[i][k] ≥ c × T[i],  ∀i
```
确保每个节点对的带宽需求至少为`c × T[i]`。

**约束3：容量约束（可选）**
```
∑(k,w) delta[i][k][w] × C[i][k] - c × T[i] ≤ max(C[i]),  ∀i
```

### 物理层模型

#### SNR计算
```
SNR = P_signal / (P_ASE + P_NLI)
```

#### 容量计算
```
C = 2 × B × log₂(1 + SNR)
```

#### 非线性噪声
- **交叉相位调制（XPM）**：相邻信道间的干扰
- **四波混频（FWM）**：多个波长相互作用产生新频率
- **自相位调制（SPM）**：信号自身的非线性效应

---

## 依赖关系图

```
ILP_throughput
│
├── Network.OpticalNetwork (初始化)
│   ├── Router.RWA
│   ├── PhysicalLayer.PhysicalLayer
│   └── Demand.Demand
│
├── network.rwa.maximise_uniform_bandwidth_demand
│   ├── Tools.get_k_shortest_paths_MNH
│   │   └── k_shortest_paths (Yen's算法)
│   │
│   ├── PhysicalLayer (SNR计算)
│   │   ├── add_wavelengths_full_occupation
│   │   ├── add_uniform_launch_power_to_links
│   │   ├── add_non_linear_NSR_to_links
│   │   │   └── get_SNR_non_linear
│   │   └── get_SNR_k_SP
│   │
│   ├── ILP模型构建
│   │   ├── Tools.nodes_to_edges (路径转边)
│   │   └── mip.Model (Gurobi求解器)
│   │
│   └── convert_delta_to_rwa_assignment
│
├── Tools.convert_rwa_to_multi_fibre (多光纤转换)
│
├── PhysicalLayer容量计算
│   ├── add_uniform_launch_power_to_links
│   ├── add_wavelengths_to_links
│   ├── add_non_linear_NSR_to_links
│   └── get_lightpath_capacities_PLI
│
├── Tools.write_database_dict
│
└── Database.update_data_with_id
```

---

## 性能考虑

### 计算复杂度

1. **k最短路径计算**：O(K × N² × (E + N log N))
   - K = 节点对数
   - N = 节点数
   - E = 边数

2. **ILP模型规模**：
   - 变量数：O(K × k × W)
   - 约束数：O(E × W + K)

3. **求解时间**：取决于问题规模和Gurobi求解器性能

### 优化建议

1. **减少k值**：减少考虑的路径数量
2. **设置时间限制**：使用`max_time`参数
3. **使用启发式**：对于大规模问题，考虑使用启发式算法
4. **并行处理**：使用Ray进行多图并行处理

---

## 总结

`ILP_throughput`函数是一个复杂的光网络吞吐量优化系统，它：

1. **使用ILP方法**：通过整数线性规划找到最优的RWA分配
2. **考虑物理层约束**：包括非线性噪声、信号功率等
3. **支持多光纤网络**：可以处理多光纤配置
4. **计算实际容量**：基于物理层模型计算真实的光路容量
5. **结果持久化**：将结果存储到数据库

该函数是评估光网络性能上限的重要工具，为网络规划和优化提供理论参考。

