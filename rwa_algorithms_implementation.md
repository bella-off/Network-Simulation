# RWA 算法实现详细分析

本文档详细分析了 `Network.py` 中动态分配的路由和波长分配（RWA）算法的实现原理和源代码。

**位置**: `networktoolbox/NetworkToolkit/Network.py`

---

## 目录

1. [动态路由函数分配机制](#1-动态路由函数分配机制)
2. [启发式算法（Heuristics）](#2-启发式算法heuristics)
   - [2.1 FF-kSP](#21-ff-ksp)
   - [2.2 kSP-FF (Revised)](#22-ksp-ff-revised)
   - [2.3 kSP-CA-FF](#23-ksp-ca-ff)
   - [2.4 kSP-baroni-FF](#24-ksp-baroni-ff)
   - [2.5 FF-kSP-multicore](#25-ff-ksp-multicore)
   - [2.6 kSP-FF-multicore](#26-ksp-ff-multicore)
3. [整数线性规划算法（ILP）](#3-整数线性规划算法ilp)
   - [3.1 ILP-min-wave](#31-ilp-min-wave)
   - [3.2 ILP-max-throughput](#32-ilp-max-throughput)
   - [3.3 ILP-min-congestion](#33-ilp-min-congestion)
4. [算法对比总结](#4-算法对比总结)

---

## 1. 动态路由函数分配机制

### 源代码位置

**文件**: `networktoolbox/NetworkToolkit/Network.py`

```python
class OpticalNetwork:
    def __init__(self, graph, B_o=5e12, channel_bandwidth=32e9,
                 mimic_topology="nsf", routing_func="FF-kSP", fibre_num=1,
                 multicore=False, pre_calculate_ksp=False, e=None, k=None,
                 channels=None):
        # ... 初始化代码 ...
        
        # 动态分配路由函数
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

### 设计模式

使用了**策略模式（Strategy Pattern）**：
- 在初始化时根据 `routing_func` 参数选择算法
- `self.route` 直接绑定到相应的 RWA 方法
- 运行时可以灵活调用，无需条件判断

### 使用方式

```python
# 创建网络对象，指定路由算法
network = OpticalNetwork(graph, routing_func="FF-kSP")

# 调用路由函数（实际调用 self.rwa.FF_kSP）
rwa_assignment = network.route(demand_matrix, e=10, k=5)
```

---

## 2. 启发式算法（Heuristics）

### 2.1 FF-kSP

**完整名称**: First Fit k-Shortest Paths

**位置**: `networktoolbox/NetworkToolkit/Routing/Heuristics.py`

#### 源代码（核心部分）

```python
def FF_kSP(self, traffic_matrix_connection_requests, e=0, k=1, 
           order_aware=False, sort_length=False, rwa_assignment_previous=None, 
           return_blocked=False, random_sorting=False, connection_pairs=None,
           _ids=None, save_ids=False):
    """
    This method implements a first fit k shortest paths algorithm 
    for an optical transport network.
    """
    # 1. 计算 k 最短路径
    if connection_pairs is None:
        k_SP = get_k_shortest_paths_MNH(self.graph, e=e, k=k)
        # 转换为 (s, d, paths) 格式
        s_d_pairs = list(map(lambda x: (x[0][0], x[0][1], x[1]), k_SP))
        s_d_pairs = np.asarray(s_d_pairs)
        
        # 2. 根据流量矩阵扩展连接对
        s_d_pairs_with_traffic = np.zeros((3,))
        for item in s_d_pairs:
            # 根据需求矩阵重复连接对
            count = int(traffic_matrix_connection_requests[item[0]-1, item[1]-1])
            s_d_pairs_item = np.tile(item, (count, 1))
            s_d_pairs_with_traffic = np.vstack((s_d_pairs_with_traffic, s_d_pairs_item))
        s_d_pairs_with_traffic = np.delete(s_d_pairs_with_traffic, 0, 0)
    else:
        # 使用提供的连接对列表
        s_d_pairs_with_traffic = []
        for s, d in connection_pairs:
            k_sp = k_shortest_paths(self.graph, s, d, k, weight="weight")
            s_d_pairs_with_traffic.append((s, d, k_sp))
    
    # 3. 初始化 RWA 分配和波长占用矩阵 W
    if rwa_assignment_previous is not None:
        rwa_assignment = rwa_assignment_previous
        W = np.zeros((len(list(self.graph.edges)), self.channels))
        for key in rwa_assignment:
            for path in rwa_assignment[key]:
                W = self.add_wavelength_path_to_W(self.graph, W, path, key)
    else:
        rwa_assignment = {i: [] for i in range(self.channels)}
        W = np.zeros((len(list(self.graph.edges)), self.channels))
    
    # 4. 可选：随机排序连接对
    if random_sorting:
        random.shuffle(s_d_pairs_with_traffic)
    
    # 5. 为每个连接请求分配路径和波长
    blocked_connections = 0
    for idx, item in enumerate(s_d_pairs_with_traffic):
        # item = (source, destination, [path1, path2, ..., pathk])
        path_wavelength = np.zeros((len(item[2]),), dtype=int)
        paths_wavelengths = []
        
        # 5.1 对每条候选路径，使用 First Fit 找到可用波长
        for index, path in enumerate(item[2]):
            FF = self.FF_return(self.graph, self.channels, W, path)
            path_wavelength[index] = FF  # 波长的索引
            paths_wavelengths.append((path, FF))
        
        # 5.2 选择最小波长索引的路径（First Fit 策略）
        index_array = np.argsort(path_wavelength)
        index_min = index_array[0]
        
        # 5.3 检查是否阻塞
        if path_wavelength[index_min] == self.channels + 1:
            # 没有可用波长，连接被阻塞
            if return_blocked:
                blocked_connections += 1
                continue
            else:
                return True  # 返回 True 表示失败
        
        # 5.4 分配路径和波长
        selected_path = item[2][index_min]
        selected_wavelength = path_wavelength[index_min]
        rwa_assignment[selected_wavelength].append(selected_path)
        
        # 5.5 更新波长占用矩阵 W
        W = self.add_wavelength_path_to_W(self.graph, W, selected_path, 
                                          selected_wavelength)
    
    return rwa_assignment
```

#### 算法原理

**FF-kSP 算法流程**:

1. **路径计算阶段**:
   - 为每个源-目的节点对计算 k 条最短路径
   - 根据流量矩阵扩展连接请求数量

2. **波长分配阶段**（First Fit）:
   - 对每个连接请求的 k 条候选路径
   - 使用 First Fit 为每条路径找到第一个可用波长
   - **选择策略**: 选择波长索引最小的路径-波长组合

3. **更新阶段**:
   - 将选中的路径添加到对应波长的 RWA 字典中
   - 更新波长占用矩阵 W

**特点**:
- ✅ **简单高效**: First Fit 策略计算快速
- ✅ **顺序处理**: 按顺序处理连接请求
- ❌ **局部最优**: 不考虑全局优化

**时间复杂度**: O(N² × k × W × E)，其中 N 是节点数，k 是路径数，W 是波长数，E 是边数

---

### 2.2 kSP-FF (Revised)

**完整名称**: k-Shortest Paths First Fit (Revised Version)

**位置**: `networktoolbox/NetworkToolkit/Routing/Heuristics.py`

#### 源代码（核心部分）

```python
def k_SP_FF_revised(self, traffic_matrix_connection_requests=None, e=0, k=1,
                    order_aware=False, sort_length=False, rwa_assignment_previous=None,
                    return_blocked=False, random_sorting=False, connection_pairs=None,
                    _ids=None, save_ids=False):
    """
    Revised version of kSP-FF algorithm.
    """
    # 1. 计算 k 最短路径
    if traffic_matrix_connection_requests is not None:
        if self.ksp is None:
            k_SP = get_k_shortest_paths_MNH(self.graph, e=e, k=k)
        else:
            k_SP = self.ksp
        # 使用辅助函数转换为连接对
        s_d_pairs_with_traffic = self.k_sp_to_s_d_pairs_with_traffic(
            k_SP, traffic_matrix_connection_requests)
    elif connection_pairs is not None:
        s_d_pairs_with_traffic = []
        for s, d in connection_pairs:
            k_sp = k_shortest_paths(self.graph, s, d, k, weight="weight")
            s_d_pairs_with_traffic.append((s, d, k_sp))
    
    # 2. 初始化
    current_path = np.zeros((len(s_d_pairs_with_traffic), 3), object)
    if rwa_assignment_previous is not None:
        rwa_assignment = rwa_assignment_previous
        W = np.zeros((len(list(self.graph.edges)), self.channels))
        for key in rwa_assignment:
            for path in rwa_assignment[key]:
                W = self.add_wavelength_path_to_W(self.graph, W, path, key)
    else:
        rwa_assignment = {i: [] for i in range(self.channels)}
        W = np.zeros((len(list(self.graph.edges)), self.channels))
    
    if random_sorting:
        random.shuffle(s_d_pairs_with_traffic)
    
    # 3. 路径和波长分配（关键差异在这里）
    blocked_connections = 0
    for idx, item in enumerate(s_d_pairs_with_traffic):
        path_wavelength = np.zeros((len(item[2]),), dtype=int)
        path_len = np.zeros((len(item[2]),), dtype=int)
        paths_wavelengths = []
        
        # 3.1 为每条路径找到可用波长
        for index, path in enumerate(item[2]):
            FF = self.FF_return(self.graph, self.channels, W, path)
            path_wavelength[index] = FF
            paths_wavelengths.append((path, FF))
            
            # 记录路径长度（仅当有可用波长时）
            if FF <= self.channels:
                path_len[index] = len(path)
        
        # 3.2 过滤：只考虑有可用波长的路径
        indices = np.where(path_len != 0)[0]
        paths = [item[2][i] for i in indices]
        paths_wavelengths = [paths_wavelengths[i] for i in indices]
        path_len = path_len[indices]
        
        # 3.3 如果没有可用路径，阻塞
        if len(path_len) == 0:
            if return_blocked:
                blocked_connections += 1
                continue
            else:
                return True
        
        # 3.4 选择最短路径（按路径长度排序）
        index_array = np.argsort(path_len)  # 按路径长度排序
        index_min = index_array[0]
        
        # 3.5 分配
        selected_path = paths_wavelengths[index_min][0]
        selected_wavelength = paths_wavelengths[index_min][1]
        rwa_assignment[selected_wavelength].append(selected_path)
        
        # 3.6 更新矩阵
        W = self.add_wavelength_path_to_W(self.graph, W, selected_path, 
                                          selected_wavelength)
    
    return rwa_assignment
```

#### 算法原理

**kSP-FF (Revised) 与 FF-kSP 的关键差异**:

| 特性 | FF-kSP | kSP-FF (Revised) |
|------|--------|------------------|
| **路径选择策略** | 选择波长索引最小的路径 | 选择**路径长度最短**的路径 |
| **过滤机制** | 考虑所有路径 | 先过滤掉无可用波长的路径 |
| **优化目标** | 最小化波长索引 | 最小化路径长度（跳数） |

**算法流程**:

1. 计算 k 条最短路径
2. 为每条路径找到可用波长（First Fit）
3. **过滤**: 只保留有可用波长的路径
4. **选择**: 在可用路径中，选择**路径长度最短**的
5. 分配路径和波长

**特点**:
- ✅ **路径优化**: 优先选择最短路径，减少延迟
- ✅ **更稳健**: 先过滤不可用路径，避免阻塞
- ✅ **适合实时**: 路径长度短，传输延迟小

---

### 2.3 kSP-CA-FF

**完整名称**: k-Shortest Paths Congestion-Aware First Fit

**位置**: `networktoolbox/NetworkToolkit/Routing/Heuristics.py`

#### 源代码

```python
def k_SP_CA_FF(self, traffic_matrix_connection_requests, e=0, k=1):
    """
    Method to do the lightpath assignment with k-shortest-paths 
    congestion-aware and the wavelength assignment with first fit.
    """
    # 1. 计算 k 最短路径
    k_SP = get_k_shortest_paths_MNH(self.graph, e=e, k=k)
    
    # 2. 拥塞感知的路径分配（LA: Lightpath Assignment）
    LA = self.kSP_CA_LA(self.graph, k_SP, traffic_matrix_connection_requests)
    
    # 3. First Fit 波长分配（WA: Wavelength Assignment）
    rwa_assignment = self.FF_WA(self.graph, LA, self.channels, 
                                self.channel_bandwidth)
    return rwa_assignment
```

#### 算法原理

**两阶段算法**:

1. **阶段 1: 拥塞感知路径分配（kSP-CA-LA）**
   - 考虑链路拥塞情况
   - 优先选择负载较低的路径
   - 避免热点链路

2. **阶段 2: First Fit 波长分配（FF-WA）**
   - 在已选路径上使用 First Fit 分配波长

**拥塞感知机制**:
- 计算每条链路的当前负载（已有光路数）
- 在选择路径时，考虑路径上所有链路的负载总和
- 优先选择负载较低的路径

**特点**:
- ✅ **负载均衡**: 避免某些链路过载
- ✅ **网络效率**: 提高资源利用率
- ✅ **适合高负载**: 在网络负载较高时表现更好

---

### 2.4 kSP-baroni-FF

**完整名称**: k-Shortest Paths Baroni-style First Fit

**位置**: `networktoolbox/NetworkToolkit/Routing/Heuristics.py`

#### 源代码

```python
def k_SP_baroni_FF(self, traffic_matrix_connection_requests, e=0, k=1,
                    rwa_assignment_previous=None):
    """
    Method to do the lightpath assignment with k-SP baroni style 
    and the wavelength with first fit.
    """
    # 1. 计算 k 最短路径
    k_SP = get_k_shortest_paths_MNH(self.graph, e=e, k=k)
    
    # 2. Baroni 风格的路径分配
    LA = self.static_optimised_baroni_MNH_LA(self.graph, 
                                             traffic_matrix_connection_requests, 
                                             k_SP, e=e)
    
    # 3. First Fit 波长分配
    rwa_assignment = self.FF_WA(self.graph, LA, self.channels, 
                                self.channel_bandwidth, 
                                rwa_assignment_previous=rwa_assignment_previous)
    return rwa_assignment
```

#### 算法原理

**Baroni 算法**:
- 基于 Baroni 等人提出的优化路径分配方法
- 使用静态优化的路径选择策略
- 考虑最小跳数（MNH: Minimum Number of Hops）

**特点**:
- ✅ **理论优化**: 基于优化理论
- ✅ **最小跳数**: 优先选择跳数最少的路径
- ✅ **适合静态**: 适合静态或准静态流量模式

---

### 2.5 FF-kSP-multicore

**完整名称**: First Fit k-Shortest Paths (Multicore Version)

**位置**: `networktoolbox/NetworkToolkit/Routing/Heuristics.py`

#### 源代码（核心部分）

```python
def ff_k_sp_multicore_revised(self, traffic_matrix_connection_requests=None, 
                               e=0, k=1, order_aware=False, sort_length=False,
                               rwa_assignment_previous=None, return_blocked=False,
                               random_sorting=False, connection_pairs=None,
                               _ids=None, save_ids=False,
                               cross_talk_matrix=None, cross_talk_threshold=None,
                               cores=1, return_blocked_xt=False, xt_dict=None,
                               symetric=True, reroute=False, lightpath_dict=None):
    """
    Multicore version of FF-kSP algorithm.
    Handles multiple cores in each fiber.
    """
    # 1. 处理多核光纤
    if not symetric:
        graph_copy = nx.to_directed(self.graph)
    else:
        graph_copy = self.graph
    
    # 2. 获取源-目的对和流量
    s_d_pairs_with_traffic = self.get_s_d_with_traffic(
        graph_copy, traffic_matrix_connection_requests, connection_pairs, 
        k=k, e=e, symetric=symetric, weight='weight')
    
    # 3. 多核初始化
    rwa_assignment, W_multicore = self.rwa_init_multicore(
        graph_copy, rwa_assignment_previous, cores=cores, symetric=symetric)
    edge_assignment = self.edge_assignment_multicore_init(
        graph_copy, rwa_assignment, cores=cores, symetric=symetric)
    
    # 4. 路径和波长分配（考虑多个核心）
    blocked_connections = 0
    blocked_conn_xt = 0  # 串扰阻塞
    
    for s_d_pair in s_d_pairs_with_traffic:
        # 计算 k 最短路径
        paths = k_shortest_paths(graph_copy, s, d, k, weight='weight')
        
        # 为每条路径选择核心和波长
        for path in paths:
            # 选择核心（考虑串扰）
            chosen_cores = self.choose_core(graph_copy, s, d, chosen_cores,
                                           xt_matrix, occupied_cores, cores=cores)
            
            # First Fit 波长分配
            wavelength = self.FF_return_multicore(graph_copy, channels, 
                                                  W_multicore, path, core)
            
            # 检查串扰约束
            if cross_talk_threshold is not None:
                if self.check_crosstalk(path, core, xt_matrix) > cross_talk_threshold:
                    blocked_conn_xt += 1
                    continue
            
            # 分配
            rwa_assignment[core][wavelength].append(path)
            # 更新多核占用矩阵
            W_multicore = self.update_W_multicore(W_multicore, path, 
                                                  wavelength, core)
    
    return rwa_assignment
```

#### 算法原理

**多核光纤扩展**:

1. **核心选择**:
   - 每条光纤包含多个核心（cores）
   - 需要考虑核心之间的串扰（crosstalk）
   - 选择串扰最小的核心

2. **波长分配**:
   - 在选定的核心上使用 First Fit 分配波长
   - 每个核心独立维护波长占用矩阵

3. **串扰约束**:
   - 检查路径是否满足串扰阈值
   - 超过阈值则阻塞连接

**特点**:
- ✅ **多核支持**: 利用多核光纤增加容量
- ✅ **串扰感知**: 考虑核心间干扰
- ✅ **容量提升**: 多核可显著增加网络容量

---

### 2.6 kSP-FF-multicore

**完整名称**: k-Shortest Paths First Fit (Multicore Version)

**位置**: `networktoolbox/NetworkToolkit/Routing/Heuristics.py`

#### 算法原理

类似 `kSP-FF-revised`，但在多核光纤上实现：
- 先选择最短路径
- 然后选择核心和波长
- 考虑串扰约束

**与 FF-kSP-multicore 的区别**:
- `FF-kSP-multicore`: 优先选择波长索引小的路径
- `kSP-FF-multicore`: 优先选择路径长度短的路径

---

## 3. 整数线性规划算法（ILP）

### 3.1 ILP-min-wave

**完整名称**: ILP Minimize Wavelengths Used

**位置**: `networktoolbox/NetworkToolkit/Routing/ILP.py`

#### 源代码（核心部分）

```python
def minimise_wavelengths_used(self, T, e=0, k=1, solver_name="GRB", max_time=1000):
    """
    Method to route a connection request matrix via ILP 
    with minimum amount of wavelengths used.
    """
    import mip.model as mip
    from mip import Model, MAXIMIZE, CBC, INTEGER, OptimizationStatus
    
    # 1. 计算 k 最短路径
    k_SP = Tools.get_k_shortest_paths_MNH(self.graph, e=e, k=k)
    W = self.channels  # 波长数
    K = len(k_SP)     # 节点对数
    E = len(list(self.graph.edges()))  # 边数
    edges = list(self.graph.edges)
    
    # 2. 创建 ILP 模型
    ILP = mip.Model('ILP', solver_name=solver_name)
    ILP.verbose = 0
    
    # 3. 定义变量
    # delta[i][k][w]: 二进制变量，表示节点对 i 的第 k 条路径是否使用波长 w
    delta = [[[ILP.add_var(var_type="B") for w in range(W)] 
              for p in range(len(k_SP[k][1]))] 
             for k in range(K)]
    
    # u_w[w]: 二进制变量，表示波长 w 是否被使用
    u_w = [ILP.add_var(var_type="B") for w in range(W)]
    
    # delta_i[i][k][e]: 指示矩阵，路径 k 是否经过边 e
    delta_i = [[[1 if (edges[_e] in Tools.nodes_to_edges(k_SP[i][1][k]) or 
                      (edges[_e][1], edges[_e][0]) in Tools.nodes_to_edges(k_SP[i][1][k])) 
                else 0 for _e in range(E)] 
                for k in range(len(k_SP[i][1]))] 
               for i in range(len(k_SP))]
    
    # 4. 目标函数：最小化使用的波长数
    ILP.objective = mip.minimize(mip.xsum(u_w[w] for w in range(W)))
    
    # 5. 约束条件
    # 约束 1: 每条边的每个波长最多只能被一条路径使用（波长连续性约束）
    for _e in range(E):
        for w in range(W):
            ILP.add_constr(
                mip.xsum(delta[i][k][w] * delta_i[i][k][_e] 
                        for i in range(len(k_SP)) 
                        for k in range(len(k_SP[i][1]))) <= 1)
    
    # 约束 2: 如果路径使用波长 w，则 u_w[w] = 1
    for z in range(K):
        for k in range(0, len(k_SP[z][1])):
            for w in range(W):
                ILP.add_constr(u_w[w] >= delta[z][k][w])
                ILP.add_constr(u_w[w] <= 1)
    
    # 约束 3: 满足所有连接需求
    T_c = [T[z[0][0]-1, z[0][1]-1] for z in k_SP]
    for z in range(K):
        ILP.add_constr(
            mip.xsum(delta[z][k][w] 
                    for k in range(len(k_SP[z][1])) 
                    for w in range(W)) == T_c[z])
    
    # 6. 求解
    ILP.emphasis = 2
    ILP.optimize(max_seconds=max_time)
    
    # 7. 转换结果
    rwa_assignment = self.convert_delta_to_rwa_assignment(delta, k_SP, W)
    
    if len(ILP.objective_values) < 1:
        return True  # 无解
    
    return rwa_assignment
```

#### 数学公式

**决策变量**:
- `δ[i][k][w] ∈ {0, 1}`: 节点对 i 的第 k 条路径是否使用波长 w
- `u[w] ∈ {0, 1}`: 波长 w 是否被使用

**目标函数**:
```
minimize: Σ(u[w])  for w = 0 to W-1
```

**约束条件**:

1. **波长连续性约束**（每条边每个波长最多一条路径）:
```
Σ(δ[i][k][w] × δ_i[i][k][e]) ≤ 1  for all e, w
```

2. **波长使用指示**:
```
u[w] ≥ δ[i][k][w]  for all i, k, w
```

3. **需求满足约束**:
```
Σ(δ[i][k][w]) = T_c[i]  for all i (节点对)
```

**特点**:
- ✅ **全局最优**: 找到使用波长数最少的解
- ❌ **计算复杂**: 时间复杂度高，适合小规模网络
- ❌ **求解时间**: 可能需要较长时间

---

### 3.2 ILP-max-throughput

**完整名称**: ILP Maximize Throughput

**位置**: `networktoolbox/NetworkToolkit/Routing/ILP.py`

#### 源代码（核心部分）

```python
def maximise_throughput(self, T, e=0, k=1, solver_name="GRB", max_time=1000):
    """
    Method to maximize network throughput using ILP.
    """
    import mip.model as mip
    
    # 1. 计算 k 最短路径（使用权重）
    k_SP = Tools.get_k_shortest_paths_MNH(self.graph, e=e, k=k, weighted='weight')
    W = self.channels
    K = len(k_SP)
    E = len(list(self.graph.edges()))
    edges = list(self.graph.edges)
    
    # 2. 计算每条路径的容量（基于 SNR）
    SNR_list = self.SNR_list  # 预先计算的 SNR 列表
    C = [[float(self.channel_bandwidth * np.log2(1 + SNR_list[i][1][k])) 
          for k in range(len(SNR_list[i][1]))] 
         for i in range(len(SNR_list))]
    
    # 3. 创建 ILP 模型
    ILP = mip.Model('ILP', solver_name=solver_name)
    ILP.verbose = 0
    
    # 4. 定义变量
    delta_i = [[[1 if (edges[_e] in Tools.nodes_to_edges(k_SP[i][1][k]) or 
                      (edges[_e][1], edges[_e][0]) in Tools.nodes_to_edges(k_SP[i][1][k])) 
                else 0 for _e in range(E)] 
                for k in range(len(k_SP[i][1]))] 
               for i in range(len(k_SP))]
    
    delta = [[[ILP.add_var(var_type="B") for w in range(W)] 
              for p in range(len(k_SP[k][1]))] 
             for k in range(K)]
    
    # 5. 目标函数：最大化总容量
    ILP.objective = mip.maximize(
        mip.xsum(C[i][k] * delta[i][k][w] 
                for i in range(K) 
                for k in range(len(k_SP[i][1])) 
                for w in range(W)))
    
    # 6. 约束条件
    # 约束 1: 波长连续性
    for _e in range(E):
        for w in range(W):
            ILP.add_constr(
                mip.xsum(delta[i][k][w] * delta_i[i][k][_e] 
                        for i in range(len(k_SP)) 
                        for k in range(len(k_SP[i][1]))) <= 1)
    
    # 约束 2: 满足需求
    T_c = [T[z[0][0]-1, z[0][1]-1] for z in k_SP]
    for z in range(K):
        ILP.add_constr(
            mip.xsum(delta[z][k][w] 
                    for k in range(len(k_SP[z][1])) 
                    for w in range(W)) == T_c[z])
    
    # 7. 求解
    ILP.emphasis = 2
    ILP.optimize(max_seconds=max_time)
    
    rwa_assignment = self.convert_delta_to_rwa_assignment(delta, k_SP, W)
    
    if len(ILP.objective_values) < 1:
        return True
    
    return rwa_assignment
```

#### 数学公式

**目标函数**:
```
maximize: Σ(C[i][k] × δ[i][k][w])  for all i, k, w
```

其中 `C[i][k]` 是路径容量:
```
C[i][k] = B × log₂(1 + SNR[i][k])
```

**约束条件**:
- 波长连续性约束（同 ILP-min-wave）
- 需求满足约束（同 ILP-min-wave）

**特点**:
- ✅ **最大化吞吐量**: 考虑物理层损伤，最大化网络容量
- ✅ **SNR 感知**: 使用真实的 SNR 值计算容量
- ❌ **计算复杂**: 需要预先计算 SNR，求解时间长

---

### 3.3 ILP-min-congestion

**完整名称**: ILP Minimize Congestion

**位置**: `networktoolbox/NetworkToolkit/Routing/ILP.py`

#### 源代码（核心部分）

```python
def minimise_congestion(self, T, e=0, k=1, solver_name="GRB", max_time=1000):
    """
    Method to minimize network congestion using ILP.
    """
    import mip.model as mip
    
    # 1. 计算 k 最短路径
    k_SP = Tools.get_k_shortest_paths_MNH(self.graph, e=e, k=k)
    W = self.channels
    K = len(k_SP)
    E = len(list(self.graph.edges()))
    edges = list(self.graph.edges)
    
    # 2. 创建 ILP 模型
    ILP = mip.Model('ILP', solver_name=solver_name)
    ILP.verbose = 0
    
    # 3. 定义变量
    delta_i = [[[1 if (edges[_e] in Tools.nodes_to_edges(k_SP[i][1][k]) or 
                      (edges[_e][1], edges[_e][0]) in Tools.nodes_to_edges(k_SP[i][1][k])) 
                else 0 for _e in range(E)] 
                for k in range(len(k_SP[i][1]))] 
               for i in range(len(k_SP))]
    
    delta = [[[ILP.add_var(var_type="B") for w in range(W)] 
              for p in range(len(k_SP[k][1]))] 
             for k in range(K)]
    
    # ε[e]: 整数变量，表示边 e 上的拥塞（光路数）
    epsilon = [ILP.add_var(var_type="I") for e in range(E)]
    
    # 4. 目标函数：最小化总拥塞
    ILP.objective = mip.minimize(mip.xsum(epsilon[e] for e in range(E)))
    
    # 5. 约束条件
    # 约束 1: 波长连续性
    for _e in range(E):
        for w in range(W):
            ILP.add_constr(
                mip.xsum(delta[i][k][w] * delta_i[i][k][_e] 
                        for i in range(len(k_SP)) 
                        for k in range(len(k_SP[i][1]))) <= 1)
    
    # 约束 2: 满足需求
    T_c = [T[z[0][0]-1, z[0][1]-1] for z in k_SP]
    for z in range(K):
        ILP.add_constr(
            mip.xsum(delta[z][k][w] 
                    for k in range(len(k_SP[z][1])) 
                    for w in range(W)) == T_c[z])
    
    # 约束 3: 定义拥塞（边上的光路总数）
    for e in range(E):
        ILP.add_constr(
            epsilon[e] >= mip.xsum(delta[z][k][w] * delta_i[z][k][e] 
                                   for z in range(K) 
                                   for k in range(len(k_SP[z][1])) 
                                   for w in range(W)))
    
    # 6. 求解
    ILP.emphasis = 2
    ILP.optimize(max_seconds=max_time)
    
    rwa_assignment = self.convert_delta_to_rwa_assignment(delta, k_SP, W)
    
    if len(ILP.objective_values) < 1:
        return True
    
    return rwa_assignment
```

#### 数学公式

**决策变量**:
- `δ[i][k][w] ∈ {0, 1}`: 路径选择
- `ε[e] ∈ ℤ⁺`: 边 e 上的拥塞（光路数）

**目标函数**:
```
minimize: Σ(ε[e])  for all e
```

**约束条件**:

1. 波长连续性约束
2. 需求满足约束
3. **拥塞定义**:
```
ε[e] ≥ Σ(δ[i][k][w] × δ_i[i][k][e])  for all e
```

**特点**:
- ✅ **负载均衡**: 最小化网络拥塞，平衡链路负载
- ✅ **网络健康**: 避免某些链路过载
- ❌ **计算复杂**: ILP 求解时间长

---

## 4. 算法对比总结

### 算法分类

| 类别 | 算法 | 优化目标 | 时间复杂度 | 适用场景 |
|------|------|----------|------------|----------|
| **启发式** | FF-kSP | 快速分配 | O(N²kWE) | 实时路由，大规模网络 |
| | kSP-FF-revised | 最短路径 | O(N²kWE) | 低延迟应用 |
| | kSP-CA-FF | 负载均衡 | O(N²kWE + C) | 高负载网络 |
| | kSP-baroni-FF | 理论优化 | O(N²kWE) | 静态流量 |
| | FF-kSP-multicore | 多核容量 | O(N²kWE×C) | 多核光纤网络 |
| | kSP-FF-multicore | 多核+短路径 | O(N²kWE×C) | 多核+低延迟 |
| **ILP** | ILP-min-wave | 最小波长数 | 指数时间 | 小规模，最优解 |
| | ILP-max-throughput | 最大吞吐量 | 指数时间 | 小规模，容量优化 |
| | ILP-min-congestion | 最小拥塞 | 指数时间 | 小规模，负载均衡 |

### 选择建议

1. **大规模网络，实时路由**: 使用 `FF-kSP` 或 `kSP-FF-revised`
2. **需要负载均衡**: 使用 `kSP-CA-FF`
3. **多核光纤网络**: 使用 `FF-kSP-multicore` 或 `kSP-FF-multicore`
4. **小规模网络，需要最优解**: 使用 ILP 算法
5. **最大化吞吐量**: 使用 `ILP-max-throughput`
6. **最小化拥塞**: 使用 `ILP-min-congestion` 或 `kSP-CA-FF`

---

## 总结

本文档详细分析了 9 种 RWA 算法的实现原理：

- **启发式算法**: 快速、简单，适合大规模网络
- **ILP 算法**: 全局最优，但计算复杂度高

每种算法都有其适用场景，选择时应根据：
- 网络规模
- 实时性要求
- 优化目标（延迟、吞吐量、负载均衡）
- 可用计算资源

