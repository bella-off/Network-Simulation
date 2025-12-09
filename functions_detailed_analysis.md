# 函数详细分析文档

本文档详细分析了光学网络仿真中使用的关键函数，包括源代码和功能说明。

---

## 目录

1. [get_k_shortest_paths_MNH](#1-get_k_shortest_paths_mnh)
2. [FF_return](#2-ff_return)
3. [network.route](#3-networkroute)
4. [time.perf_counter](#4-timeperf_counter)
5. [add_uniform_launch_power_to_links](#5-add_uniform_launch_power_to_links)
6. [add_wavelengths_to_links](#6-add_wavelengths_to_links)
7. [add_non_linear_NSR_to_links](#7-add_non_linear_nsr_to_links)
8. [get_lightpath_capacities_PLI](#8-get_lightpath_capacities_pli)
9. [write_database_dict](#9-write_database_dict)
10. [update_data_with_id](#10-update_data_with_id)

---

## 1. get_k_shortest_paths_MNH

### 源代码

**位置**: `networktoolbox/NetworkToolkit/Routing/Tools.py`

```python
def get_k_shortest_paths_MNH(graph, e=None, k=1, weighted=None, data_dict=False, multigraph=None, fibres=None, ccc=False, InS=False,
                            parralel=False):
    """
    Updated method to calculate the k-shortest paths for all node-pairs in a graph.
    :param graph:       nx.graph for which to calculate them
    :param e:           amount of paths to return
    :param weighted:    whether to weight the graphs
    :return:            return [((s,d),[[path],...,[path]])]
    """
    if not data_dict:
        k_sp = []
        # Iterate through node pairs (+1 indexed nodes)
        for i in range(1,len(graph)+1):
            for j in range(1, len(graph)+1):
                if j < i:
                    pass
                elif j == i:
                    pass
                else:
                    k_sp_paths = k_shortest_paths(graph, i, j, k, weight=weighted)
                    # k_sp_paths = list(filter(lambda path: True if len(path)==e else False))
                    k_sp.append(((i,j), k_sp_paths))

        # min(k_sp,key=lambda x: min(x[1])
        # min_path_len = len(min(min(k_sp, key=lambda x: len(min(x[1], key=lambda y: len(y))))[1], key=lambda z: len(z)))
        if e is not None:
            k_sp = list(map(lambda k_sp: ((k_sp[0][0], k_sp[0][1]), list(
                filter(lambda x: True if len(x) <= len(min(k_sp[1], key=lambda x: len(x)))+e else False, k_sp[1]))), k_sp))
    else:
        k_sp = {}
        tasks = []
        for i in range(1,len(graph)+1):
            for j in range(1, len(graph)+1):
                if j < i:
                    pass
                elif j == i:
                    pass
                else:
                    if multigraph is not None and not parralel:
                        k_sp_paths_i = k_shortest_paths_multi(graph, multigraph, i, j, k, fibres,  ccc=ccc, InS=InS)
                        k_sp_paths_j = k_shortest_paths_multi(graph, multigraph, j, i, k, fibres,  ccc=ccc, InS=InS)

                        k_sp[(i,j)] = k_sp_paths_i
                        k_sp[(j,i)] = k_sp_paths_j
                    elif multigraph is not None and parralel:
                        tasks.append(k_shortest_paths_multi_parralel.remote(graph, multigraph, i, j, k, fibres,  ccc=ccc, InS=InS))
                    else:
                        k_sp_paths_i = k_shortest_paths(graph, i, j, k, weight=weighted)
                        k_sp_paths_j = k_shortest_paths(graph, j, i, k, weight=weighted)
                        k_sp[(i,j)] = k_sp_paths_i
                        k_sp[(j, i)] = k_sp_paths_j
        if parralel:
            # print(tasks)
            results = ray.get(tasks)
            for ksp, i, j in results:
                k_sp[(i,j)] = ksp

    return k_sp
```

### 功能分析

**功能**: 计算图中所有节点对的 k 最短路径（k-shortest paths）。

**参数说明**:
- `graph`: NetworkX 图对象，表示网络拓扑
- `e`: 可选参数，允许返回的路径长度比最短路径长 e 跳
- `k`: 每个节点对要计算的路径数量（默认 1）
- `weighted`: 是否使用边的权重（布尔值）
- `data_dict`: 是否返回字典格式（默认 False，返回列表）
- `multigraph`: 多图对象（如果使用多图）
- `fibres`: 光纤数量
- `ccc`: 是否使用 CCC（Cross-Connect Capability）
- `InS`: 是否使用 InS（In-Service）
- `parralel`: 是否使用并行计算（Ray）

**返回值**:
- 如果 `data_dict=False`: 返回列表 `[((s,d), [[path1], [path2], ...]), ...]`
- 如果 `data_dict=True`: 返回字典 `{(s,d): [[path1], [path2], ...], ...}`

**算法流程**:
1. 遍历所有节点对 `(i, j)`，其中 `i < j`（避免重复）
2. 对每个节点对调用 `k_shortest_paths()` 计算 k 条最短路径
3. 如果指定了 `e`，过滤路径：只保留长度 ≤ 最短路径长度 + e 的路径
4. 如果使用多图或并行计算，采用相应的处理方式

**使用场景**:
- 在 RWA（路由和波长分配）算法中，需要为每个源-目的节点对找到多条候选路径
- 用于负载均衡和路径多样性

**示例**:
```python
# 计算所有节点对的 3 条最短路径，允许路径长度比最短路径长 2 跳
k_sp = get_k_shortest_paths_MNH(graph, e=2, k=3)
# 结果: [((1,2), [[1,3,2], [1,4,2], [1,5,3,2]]), ((1,3), [...]), ...]
```

---

## 2. FF_return

### 源代码

**位置**: `networktoolbox/NetworkToolkit/Routing/Heuristics.py`

```python
def FF_return(self, graph, channels, W, path, return_all=False):
    """
    Method to return the first available wavelength for a path W: ExChannes matrix and a path.
    :param graph:       Graph to use - nx.Graph()
    :param channels:    Amount of Channels to use - int
    :param W:           Exchange matrix - ndarray
    :param path:        Path to find wavelength for - list
    :return:            Wavelength for path - int
    """
    P = np.zeros((len(list(graph.edges)), 1))  # path vector, Ex1, 1 if path includes edge, 0 if not
    graph_edges = list(graph.edges())
    path_edges = nodes_to_edges(path)
    for index, (s,d) in enumerate(graph_edges):  # creating path vector
        if (s,d) in path_edges or (d,s) in path_edges:
            P[index] = 1

    a = np.einsum('ij,ij->j', W, P)  # finding vector multiplication and then column sum
    #print(a)
    indeces = np.where(a == 0)  # only when the column sum is 0 means it's a valid wavelength
    # print("graph edges: {}".format(graph_edges))
    # print("P: {}".format(P))
    # print("W: {}".format(W))
    # print("path: {}".format(path))
    # print("a: {}".format(a))
    # print("indeces: {}".format(indeces))
    # print(np.multiply(W, P))
    # print(np.sum(np.multiply(W, P), axis=1))
    try:
        if not return_all:
            min_wave = np.min(indeces)  # if no available wavelength it crashes
        else:
            min_wave = indeces
    except Exception as err:
        # print("ERROR: {}".format(err))
        return channels + 1  # return the +1 argument and handle externally
    return min_wave  # otherwise return the wavelength
```

### 功能分析

**功能**: First Fit（首次适配）波长分配算法，为给定路径找到第一个可用的波长。

**参数说明**:
- `graph`: NetworkX 图对象
- `channels`: 可用波长数量（整数）
- `W`: 交换矩阵（Exchange Matrix），形状为 `(E, channels)`，其中 E 是边数
  - `W[i, j] = 1` 表示边 i 上波长 j 已被占用
  - `W[i, j] = 0` 表示边 i 上波长 j 可用
- `path`: 节点路径列表，例如 `[1, 3, 5, 7]`
- `return_all`: 是否返回所有可用波长（默认 False，只返回第一个）

**返回值**:
- 成功: 返回可用波长的索引（整数，从 0 开始）
- 失败: 返回 `channels + 1`（表示没有可用波长，需要外部处理）
- 如果 `return_all=True`: 返回所有可用波长的索引数组

**算法流程**:
1. **创建路径向量 P**: 
   - 大小为 `(E, 1)`，其中 E 是图的边数
   - `P[i] = 1` 表示路径经过边 i，否则为 0
2. **计算波长占用情况**:
   - 使用 `np.einsum('ij,ij->j', W, P)` 计算每个波长在路径上的占用情况
   - 结果 `a[j]` 表示波长 j 在路径上占用的边数
   - `a[j] = 0` 表示波长 j 在路径的所有边上都可用
3. **找到第一个可用波长**:
   - 使用 `np.where(a == 0)` 找到所有可用波长
   - 返回最小索引（First Fit 策略）

**数学原理**:
- 路径向量 P 与交换矩阵 W 的逐元素乘积，然后按列求和
- 如果某列（波长）的和为 0，说明该波长在路径的所有边上都未被占用

**使用场景**:
- 在 RWA 算法中，为每条路径分配波长
- First Fit 策略简单高效，适合启发式算法

**示例**:
```python
# 假设图有 5 条边，3 个波长
W = np.array([[1, 0, 0],  # 边 0: 波长 0 被占用
              [0, 0, 0],  # 边 1: 所有波长可用
              [0, 1, 0],  # 边 2: 波长 1 被占用
              [0, 0, 0],
              [0, 0, 1]])
path = [1, 2, 3]  # 经过边 1 和边 2
wavelength = FF_return(graph, channels=3, W=W, path=path)
# 返回: 0（波长 0 在边 1 和边 2 上都可用）
```

---

## 3. network.route

### 源代码

**位置**: `networktoolbox/NetworkToolkit/Network.py`

```python
# 在 OpticalNetwork.__init__ 中动态分配
if routing_func == "FF-kSP":
    self.route = self.rwa.FF_kSP
elif routing_func == "kSP-FF":
    self.route = self.rwa.k_SP_FF_revised
elif routing_func == "kSP-FF-multicore":
    self.route = self.rwa.k_sp_ff_multicore_revised
elif routing_func == "FF-kSP-multicore":
    self.route = self.rwa.ff_k_sp_multicore_revised
elif routing_func == "kSP-CA-FF":
    self.route = self.rwa.k_SP_CA_FF
elif routing_func == "kSP-baroni-FF":
    self.route = self.rwa.k_SP_baroni_FF
elif routing_func == "ILP-min-wave":
    self.route = self.rwa.minimise_wavelengths_used
elif routing_func == "ILP-max-throughput":
    self.route = self.rwa.maximise_throughput
elif routing_func == "ILP-min-congestion":
    self.route = self.rwa.minimise_congestion
```

### 功能分析

**功能**: `network.route` 是一个动态分配的路由函数，根据 `routing_func` 参数选择不同的 RWA 算法。

**参数说明**:
- `routing_func`: 路由函数名称字符串，决定使用哪种算法

**可用的路由函数**:
1. **"FF-kSP"**: First Fit k-Shortest Paths
   - 先计算 k 条最短路径，然后使用 First Fit 分配波长
2. **"kSP-FF"**: k-Shortest Paths First Fit（修订版）
   - 类似 FF-kSP，但实现略有不同
3. **"kSP-FF-multicore"**: 多核版本的 kSP-FF
4. **"FF-kSP-multicore"**: 多核版本的 FF-kSP
5. **"kSP-CA-FF"**: k-Shortest Paths with Congestion Awareness First Fit
6. **"kSP-baroni-FF"**: Baroni 算法的 kSP-FF 变体
7. **"ILP-min-wave"**: ILP 优化，最小化使用的波长数
8. **"ILP-max-throughput"**: ILP 优化，最大化吞吐量
9. **"ILP-min-congestion"**: ILP 优化，最小化拥塞

**使用方式**:
```python
# 创建网络对象
network = OpticalNetwork(graph, routing_func="FF-kSP")

# 调用路由函数
rwa_assignment = network.route(demand_matrix, e=10, k=5)
```

**返回值**:
- 成功: 返回 RWA 分配字典 `{wavelength: [path1, path2, ...], ...}`
- 失败: 返回 `True`（表示路由失败）

**设计模式**:
- 使用策略模式（Strategy Pattern），在初始化时根据参数选择算法
- 这样可以在运行时灵活切换不同的路由算法

---

## 4. time.perf_counter

### 源代码

**位置**: Python 标准库 `time` 模块

```python
# Python 标准库实现（C 语言）
import time

time_start = time.perf_counter()
# ... 执行代码 ...
time_taken = time.perf_counter() - time_start
```

### 功能分析

**功能**: 返回一个高精度的时间计数器，用于测量代码执行时间。

**特点**:
- **高精度**: 使用系统最高可用精度的时钟（通常是纳秒级）
- **单调递增**: 不受系统时钟调整影响（不会因为系统时间被修改而倒退）
- **适合性能测量**: 专门设计用于性能分析

**返回值**:
- 浮点数，表示从某个固定时间点开始的秒数（具体起点未定义，只用于计算差值）

**使用场景**:
- 测量函数执行时间
- 性能分析和优化
- 算法复杂度验证

**示例**:
```python
import time

# 开始计时
start = time.perf_counter()

# 执行一些操作
result = some_function()

# 结束计时
elapsed = time.perf_counter() - start
print(f"执行时间: {elapsed:.4f} 秒")
```

**与其他时间函数的区别**:
- `time.time()`: 返回系统时间（可能受时钟调整影响）
- `time.perf_counter()`: 返回性能计数器（单调递增，适合测量）
- `time.process_time()`: 返回进程 CPU 时间（不包括睡眠时间）

**在代码中的使用**:
```python
# NetworkSimulator.py 中的使用
time_start = time.perf_counter()
# ... 执行 RWA 算法 ...
time_taken = time.perf_counter() - time_start
print("Time taken to route: {}".format(time_taken))
```

---

## 5. add_uniform_launch_power_to_links

### 源代码

**位置**: `networktoolbox/NetworkToolkit/PhysicalLayer.py`

```python
def add_uniform_launch_power_to_links(self, channels):
    """

    :param channels:
    :return:
    """
    graph = self.graph
    launch_powers = {}
    graph_edges = graph.edges()
    for edge in graph_edges:
        launch_powers[edge] = {"launch_powers": [(10 ** (-3))] * channels}
        launch_powers[(edge[1], edge[0])] = {"launch_powers": [(10 ** (-3))] * channels}
    logging.debug(launch_powers)
    nx.set_edge_attributes(graph, launch_powers)
    return graph
```

### 功能分析

**功能**: 为图中的所有边添加统一的发射功率（launch power）属性。

**参数说明**:
- `channels`: 波长数量（整数）

**算法流程**:
1. 遍历图的所有边
2. 为每条边（双向）设置 `launch_powers` 属性
3. 每个波长使用相同的发射功率：`10^(-3)` 瓦特（1 毫瓦，约 -30 dBm）

**物理意义**:
- **发射功率**: 光信号在光纤中传输时的初始功率
- **统一功率**: 所有波长使用相同的发射功率（简化模型）
- **单位**: 瓦特（W），`10^(-3)` = 0.001 W = 1 mW

**返回值**:
- 返回更新后的图对象（NetworkX Graph）

**使用场景**:
- 在计算物理层性能（如 SNR、容量）之前，需要设置发射功率
- 作为物理层建模的第一步

**示例**:
```python
# 为 100 个波长设置统一的发射功率
network.physical_layer.add_uniform_launch_power_to_links(channels=100)

# 访问边的发射功率
edge = (1, 2)
launch_power = graph[1][2]["launch_powers"][0]  # 0.001 W
```

**注意**:
- 代码中 `[(10 ** (-3))] * channels` 创建了一个包含相同值的列表
- 实际上每个波长都有相同的发射功率值

---

## 6. add_wavelengths_to_links

### 源代码

**位置**: `networktoolbox/NetworkToolkit/PhysicalLayer.py`

```python
def add_wavelengths_to_links(self, wavelengths_dict):
    """

    :return:
    """
    graph = self.graph
    wavelengths = {}
    graph_edges = graph.edges()
    logging.debug(graph_edges)
    logging.debug(wavelengths_dict)
    for edge in graph_edges:
        wavelengths[edge] = {"wavelengths": []}
        wavelengths[(edge[1], edge[0])] = {"wavelengths": []}

    for key in wavelengths_dict:
        for path in wavelengths_dict[key]:
            edges = self.nodes_to_edges(path)
            for edge in edges:
                wavelengths[edge]["wavelengths"].append(key)
                wavelengths[(edge[1], edge[0])]["wavelengths"].append(key)
    nx.set_edge_attributes(graph, wavelengths)
    # logging.info(self.RWA_graph[1][8]["wavelengths"])
    return graph
```

### 功能分析

**功能**: 将 RWA 分配结果（波长-路径映射）添加到图的边属性中。

**参数说明**:
- `wavelengths_dict`: RWA 分配字典，格式为 `{wavelength: [path1, path2, ...], ...}`
  - `wavelength`: 波长索引（整数）
  - `path`: 节点路径列表，例如 `[1, 3, 5, 7]`

**算法流程**:
1. **初始化**: 为所有边创建空的 `wavelengths` 列表
2. **遍历 RWA 分配**: 对于每个波长和其对应的路径列表
3. **路径转边**: 使用 `nodes_to_edges()` 将节点路径转换为边序列
4. **更新边属性**: 将波长索引添加到路径上所有边的 `wavelengths` 列表中
5. **双向处理**: 同时更新 `(u, v)` 和 `(v, u)` 两个方向的边

**数据结构**:
```python
# 输入
wavelengths_dict = {
    0: [[1, 2, 3], [4, 5, 6]],  # 波长 0 用于路径 [1,2,3] 和 [4,5,6]
    1: [[1, 4, 3]]              # 波长 1 用于路径 [1,4,3]
}

# 输出（图边属性）
graph[1][2]["wavelengths"] = [0]  # 边 (1,2) 使用波长 0
graph[2][3]["wavelengths"] = [0]  # 边 (2,3) 使用波长 0
graph[1][4]["wavelengths"] = [1]  # 边 (1,4) 使用波长 1
```

**返回值**:
- 返回更新后的图对象

**使用场景**:
- 在 RWA 分配完成后，将分配结果存储到图中
- 后续用于计算物理层性能（如 SNR、非线性噪声）

**示例**:
```python
# RWA 分配结果
rwa_assignment = {
    0: [[1, 2, 3], [4, 5]],
    1: [[1, 4, 3]]
}

# 添加到图中
network.physical_layer.add_wavelengths_to_links(rwa_assignment)

# 查询边上的波长
print(graph[1][2]["wavelengths"])  # [0]
print(graph[1][4]["wavelengths"])  # [1]
```

---

## 7. add_non_linear_NSR_to_links

### 源代码

**位置**: `networktoolbox/NetworkToolkit/PhysicalLayer.py`

```python
def add_non_linear_NSR_to_links(self, Plot=False, channels_full=156, channel_bandwidth=float(16e9)):
    """

    :param Plot:
    :return:
    """
    graph = self.graph
    
    
    NSR = {}

    graph_edges = graph.edges()
    for edge in graph_edges:
        NSR[edge] = {
            "NSR": self.get_SNR_non_linear(edge, graph, channels_full=self.channels,
                                           channel_bandwidth=self.channel_bandwidth,
                                           name="SNR_{}_{}".format(edge[0], edge[1]))}
        NSR[(edge[1], edge[0])] = {"NSR": self.get_SNR_non_linear(edge, graph, channels_full=channels_full,
                                                                  channel_bandwidth=channel_bandwidth)}
        nx.set_edge_attributes(graph, NSR)

        if Plot:
            import NetworkToolkit.Plotting as Plotting
            # print(self.RWA_graph[edge[0]][edge[1]]["NSR"])
            SNR = list(map(lambda x: 1 / x, graph[edge[0]][edge[1]]["NSR"]))
            Plotting.plot_bar_channels(Plotting.concatenate_empty_spectrum(SNR, edge, graph),
                                       file_name="Figures/ACMN1_1/SNR_{}_{}".format(edge[0], edge[1]),
                                       variable_x="f_i", title="SNR Spectrum", x_axes_unit="Frequency Channel",
                                       y_axes_unit="dB", x_axes="Channel", y_axes="SNR")
    return graph
```

### 功能分析

**功能**: 计算并添加非线性噪声比（Non-linear Noise-to-Signal Ratio, NSR）到图的边属性中。

**参数说明**:
- `Plot`: 是否绘制 SNR 频谱图（默认 False）
- `channels_full`: 完整信道数量（默认 156）
- `channel_bandwidth`: 信道带宽（默认 16 GHz）

**算法流程**:
1. **遍历所有边**: 对每条边计算 NSR
2. **调用计算函数**: 使用 `get_SNR_non_linear()` 计算非线性噪声
3. **存储结果**: 将 NSR 值存储到边的 `NSR` 属性中
4. **双向处理**: 同时更新两个方向的边
5. **可选绘图**: 如果 `Plot=True`，绘制 SNR 频谱图

**物理意义**:
- **NSR (Noise-to-Signal Ratio)**: 噪声与信号功率的比值
- **非线性噪声**: 由于光纤的非线性效应（如自相位调制、交叉相位调制、四波混频）产生的噪声
- **SNR (Signal-to-Noise Ratio)**: 信噪比 = 1 / NSR

**返回值**:
- 返回更新后的图对象

**使用场景**:
- 在计算光路容量之前，需要知道每条边上的 NSR
- NSR 用于后续的容量计算（Shannon 容量公式）

**示例**:
```python
# 计算非线性 NSR
network.physical_layer.add_non_linear_NSR_to_links(
    Plot=False, 
    channels_full=100, 
    channel_bandwidth=50e9
)

# 访问边的 NSR
edge = (1, 2)
nsr_list = graph[1][2]["NSR"]  # NSR 值列表（每个波长一个值）
snr_list = [1/nsr for nsr in nsr_list]  # 转换为 SNR
```

**注意**:
- NSR 值越小越好（噪声越小）
- SNR 值越大越好（信噪比越高）
- 非线性噪声与发射功率、光纤长度、波长占用情况相关

---

## 8. get_lightpath_capacities_PLI

### 源代码

**位置**: `networktoolbox/NetworkToolkit/PhysicalLayer.py`

```python
def get_lightpath_capacities_PLI(self, wavelengths_dict, optim_snr=False):
    """

    :return:
    """
    node_pair_capacities = {}
    graph = self.graph
    channel_bandwidth =self.channel_bandwidth
    capacity_total = 0
    n = 0
    capacity_matrix = [[0 for j in graph.nodes()] for i in graph.nodes()]
    for key in wavelengths_dict:
        logging.debug(key)
        for path in wavelengths_dict[key]:

            n += 1
            edges = self.nodes_to_edges(path)
            logging.debug("edges: {}".format(edges))
            # logging.info("key: {} {}".format(key, len(self.RWA_graph[4][7]["NSR"])))
            NSR = 0
            for edge in edges:
                index = graph[edge[0]][edge[1]]["wavelengths"].index(key)
                if optim_snr:
                    NSR += graph[edge[0]][edge[1]]["NSR_opt"][index]
                else:
                    NSR += graph[edge[0]][edge[1]]["NSR"][index]
            SNR = 1 / NSR
            logging.debug("SNR:{}".format(10 * np.log10(SNR)))

            capacity = 2 * channel_bandwidth * np.log2(1 + SNR)
            # print(SNR, channel_bandwidth, capacity)
            # adding capacities to the node-pair dict
            if (path[0], path[-1]) in node_pair_capacities.keys():
                node_pair_capacities[(path[0], path[-1])] += capacity
            else:
                node_pair_capacities[(path[0], path[-1])] = capacity
            # print(np.log2(1 + SNR))
            capacity_total += capacity
            capacity_matrix[path[0] - 1][path[-1] - 1] += capacity
            logging.debug("Capacity: {}".format(capacity / 1e9))
            logging.debug("Capacity Total: {}".format(capacity_total / 1e12))
    #  print(capacity_total / 1e12)
    # print(n)
    #  print(self.N_lambda)
    # print(len(self.lightpath_routes_consecutive_single))

    capacity_average = capacity_total / n
    return capacity_total, capacity_average, node_pair_capacities
```

### 功能分析

**功能**: 计算考虑物理层损伤（Physical Layer Impairments, PLI）的光路容量。

**参数说明**:
- `wavelengths_dict`: RWA 分配字典，格式为 `{wavelength: [path1, path2, ...], ...}`
- `optim_snr`: 是否使用优化的 SNR（默认 False，使用普通 NSR）

**算法流程**:
1. **遍历所有光路**: 对每个波长和其对应的路径
2. **计算路径 NSR**: 
   - 将路径上所有边的 NSR 累加
   - `NSR_path = sum(NSR_edge for edge in path)`
3. **计算 SNR**: `SNR = 1 / NSR_path`
4. **计算容量**: 使用 Shannon 容量公式
   - `capacity = 2 * channel_bandwidth * log2(1 + SNR)`
   - 因子 2 表示双向传输
5. **累加统计**:
   - 总容量: 所有光路容量之和
   - 节点对容量: 按源-目的节点对分组
   - 容量矩阵: 节点对容量矩阵
6. **计算平均值**: `capacity_average = capacity_total / n`

**数学公式**:
- **Shannon 容量公式**: `C = B * log2(1 + SNR)`
  - `C`: 容量（bps）
  - `B`: 带宽（Hz）
  - `SNR`: 信噪比
- **双向传输**: 乘以 2（上行和下行）

**返回值**:
- `capacity_total`: 总容量（浮点数，单位：bps）
- `capacity_average`: 平均每条光路的容量（浮点数，单位：bps）
- `node_pair_capacities`: 节点对容量字典 `{(s,d): capacity, ...}`

**使用场景**:
- 在 RWA 分配完成后，计算网络的吞吐量
- 评估网络性能
- 用于吞吐量优化

**示例**:
```python
# 计算光路容量
capacity_total, capacity_avg, node_pair_caps = network.physical_layer.get_lightpath_capacities_PLI(
    rwa_assignment, 
    optim_snr=False
)

print(f"总容量: {capacity_total/1e12:.4f} Tbps")
print(f"平均容量: {capacity_avg/1e9:.2f} Gbps")
print(f"节点对 (1,3) 的容量: {node_pair_caps[(1,3)]/1e9:.2f} Gbps")
```

**注意**:
- NSR 是累加的（路径越长，噪声越大）
- SNR 是 NSR 的倒数
- 容量受物理层损伤限制，不是理论最大值

---

## 9. write_database_dict

### 源代码

**位置**: `networktoolbox/NetworkToolkit/Tools.py`

```python
def write_database_dict(dict):
    new_dict = {str(key): value for key, value in dict.items()}
    return new_dict
```

### 功能分析

**功能**: 将字典的键转换为字符串，以便存储到 MongoDB 数据库。

**参数说明**:
- `dict`: 输入字典，键可能是整数或其他类型

**返回值**:
- 新字典，所有键都转换为字符串

**使用场景**:
- MongoDB 要求字典的键必须是字符串
- RWA 分配字典的键通常是整数（波长索引），需要转换为字符串

**示例**:
```python
# 输入
rwa_assignment = {
    0: [[1, 2, 3], [4, 5]],
    1: [[1, 4, 3]]
}

# 转换
rwa_write = write_database_dict(rwa_assignment)
# 输出: {"0": [[1, 2, 3], [4, 5]], "1": [[1, 4, 3]]}
```

**注意**:
- 这是一个简单的数据转换函数
- 确保 MongoDB 可以正确存储数据

---

## 10. update_data_with_id

### 源代码

**位置**: `networktoolbox/NetworkToolkit/Database.py`

```python
def update_data_with_id(db_name, collection_name, _id, newvals):
    """
    Method to update data given it's unique id (normally used with graph lists whih
    reutrns a graph and id which can be used to update values.
    :param db_name:         Database where to update data
    :param collection_name: Collection where to update data
    :param _id:             _id to use for the data.
    :param newvals:         newvals to which to assign data e.g. {"$set": {"S": S}}
    :return: None
    """
    # Client to connect to database, change port number to whatever your tunnel is using
    client = pymongo.MongoClient('mongodb://localhost:{}'.format(port), username=user, password=pwd)
    db = client[db_name]
    data = db[collection_name]
    # Tools.assert_python_types(newvals)
    data.update_one({"_id": _id}, newvals)
    client.close()
    return None
```

### 功能分析

**功能**: 根据文档的 `_id` 更新 MongoDB 数据库中的文档。

**参数说明**:
- `db_name`: 数据库名称（字符串）
- `collection_name`: 集合名称（字符串）
- `_id`: 文档的唯一标识符（通常是 ObjectId）
- `newvals`: MongoDB 更新操作字典，例如 `{"$set": {"field": value}}`

**MongoDB 更新操作**:
- `{"$set": {...}}`: 设置字段值
- `{"$inc": {...}}`: 增加字段值
- `{"$push": {...}}`: 向数组添加元素
- 等等

**算法流程**:
1. **连接数据库**: 使用 PyMongo 连接到 MongoDB
2. **选择集合**: 获取指定的数据库和集合
3. **执行更新**: 使用 `update_one()` 更新匹配 `_id` 的文档
4. **关闭连接**: 释放数据库连接

**返回值**:
- `None`

**使用场景**:
- 在仿真完成后，将结果保存到数据库
- 更新拓扑的 RWA 分配、吞吐量等结果

**示例**:
```python
# 更新数据库中的结果
Database.update_data_with_id(
    db="Topology_Data",
    collection="topology-paper",
    _id=graph_id,
    newvals={
        "$set": {
            "FF-kSP RWA": rwa_write,
            "FF-kSP Capacity": throughput,
            "FF-kSP-connections": M,
            "FF-kSP time": time_taken
        }
    }
)
```

**注意**:
- 使用 `update_one()` 只更新第一个匹配的文档
- `_id` 是唯一标识符，通常只会匹配一个文档
- 需要确保数据库连接配置正确（端口、用户名、密码）

---

## 总结

本文档详细分析了光学网络仿真中的 10 个关键函数：

1. **路径计算**: `get_k_shortest_paths_MNH` - 计算 k 最短路径
2. **波长分配**: `FF_return` - First Fit 波长分配
3. **路由函数**: `network.route` - 动态路由算法选择
4. **性能测量**: `time.perf_counter` - 高精度时间测量
5. **物理层建模**:
   - `add_uniform_launch_power_to_links` - 设置发射功率
   - `add_wavelengths_to_links` - 添加波长分配
   - `add_non_linear_NSR_to_links` - 计算非线性噪声
   - `get_lightpath_capacities_PLI` - 计算光路容量
6. **数据存储**:
   - `write_database_dict` - 数据格式转换
   - `update_data_with_id` - 数据库更新

这些函数共同构成了完整的光学网络仿真流程：从路径计算、RWA 分配、物理层建模到结果存储。

