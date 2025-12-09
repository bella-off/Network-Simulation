# FF-kSP 和 heuristic_throughput 详细分析文档

本文档详细分析了 `thesis/throughput_upperbound` 文件夹中的核心功能：
1. **FF-kSP 算法实现** (`FF_kSP` 函数)
2. **k-SP-FF 算法实现** (`k_SP_FF_revised` 函数)
3. **heuristic_throughput 函数实现**

---

## 目录

1. [FF-kSP 算法详细分析](#1-ff-ksp-算法详细分析)
2. [k-SP-FF 算法详细分析](#2-k-sp-ff-算法详细分析)
3. [FF-kSP 与 k-SP-FF 算法对比](#3-ff-ksp-与-k-sp-ff-算法对比)
4. [heuristic_throughput 函数详细分析](#4-heuristic_throughput-函数详细分析)
5. [ff_k_sp.py 脚本分析](#5-ff_k_sppy-脚本分析)
6. [算法流程总结](#6-算法流程总结)
7. [关键数据结构](#7-关键数据结构)

---

## 1. FF-kSP 算法详细分析

### 1.1 算法概述

**FF-kSP (First Fit k-Shortest Paths)** 是一种启发式 RWA（路由和波长分配）算法，结合了：
- **k-Shortest Paths (kSP)**: 为每个节点对计算 k 条最短路径
- **First Fit (FF)**: 为每条路径分配第一个可用的波长

### 1.2 函数签名

**位置**: `networktoolbox/NetworkToolkit/Routing/Heuristics.py`

```python
def FF_kSP(self, traffic_matrix_connection_requests, e=0, k=1, 
           order_aware=False, sort_length=False,
           rwa_assignment_previous=None, return_blocked=False, 
           random_sorting=False, connection_pairs=None,
           _ids=None, save_ids=False):
```

### 1.3 参数说明

| 参数 | 类型 | 说明 |
|------|------|------|
| `traffic_matrix_connection_requests` | numpy.ndarray | N×N 流量矩阵，`T_c[i,j]` 表示节点 i 到节点 j 的连接请求数 |
| `e` | int | 路径长度参数，允许路径长度 ≤ 最短路径长度 + e |
| `k` | int | k 最短路径数量 |
| `order_aware` | bool | 是否考虑连接顺序 |
| `sort_length` | bool | 是否按路径长度排序 |
| `rwa_assignment_previous` | dict | 之前的 RWA 分配结果（用于增量分配） |
| `return_blocked` | bool | 是否返回阻塞的连接数 |
| `random_sorting` | bool | 是否随机排序连接请求 |
| `connection_pairs` | list | 连接对列表（如果提供，替代流量矩阵） |
| `_ids` | list | 连接请求的 ID 列表 |
| `save_ids` | bool | 是否保存连接 ID 信息 |

### 1.4 算法实现详解

#### 步骤 1: 计算 k 最短路径

```python
if connection_pairs is None:
    k_SP = get_k_shortest_paths_MNH(self.graph, e=e, k=k)
```

**功能**: 为所有节点对计算 k 条最短路径

**返回格式**: 
```python
k_SP = [
    ((源节点, 目的节点), [路径1, 路径2, ..., 路径k]),
    ...
]
```

**示例**:
```python
k_SP = [
    ((1, 2), [[1, 3, 2], [1, 4, 2], [1, 5, 3, 2]]),  # 节点对 (1,2) 的 3 条路径
    ((1, 3), [[1, 3], [1, 2, 3], [1, 4, 3]]),         # 节点对 (1,3) 的 3 条路径
    ...
]
```

#### 步骤 2: 转换数据结构

```python
s_d_pairs = list(map(lambda x: (x[0][0], x[0][1], x[1]), k_SP))
s_d_pairs = np.asarray(s_d_pairs)
```

**功能**: 将路径数据转换为更易处理的格式

**转换前**:
```python
((1, 2), [[1, 3, 2], [1, 4, 2]])
```

**转换后**:
```python
(1, 2, [[1, 3, 2], [1, 4, 2]])
```

#### 步骤 3: 根据流量矩阵扩展连接请求

```python
s_d_pairs_with_traffic = np.zeros((3,))
for item in s_d_pairs:
    s_d_pairs_item = np.tile(item, (int(traffic_matrix_connection_requests[item[0] - 1, item[1] - 1]), 1))
    s_d_pairs_with_traffic = np.vstack((s_d_pairs_with_traffic, s_d_pairs_item))
s_d_pairs_with_traffic = np.delete(s_d_pairs_with_traffic, 0, 0)
```

**功能**: 根据流量矩阵中的需求数量，为每个节点对创建多个连接请求

**示例**:
- 如果 `T_c[0, 1] = 3`（节点 1 到节点 2 需要 3 个连接）
- 则 `(1, 2, paths)` 会被复制 3 次

**结果格式**:
```python
[
    (1, 2, [[1, 3, 2], [1, 4, 2]]),  # 连接请求 1
    (1, 2, [[1, 3, 2], [1, 4, 2]]),  # 连接请求 2
    (1, 2, [[1, 3, 2], [1, 4, 2]]),  # 连接请求 3
    (1, 3, [[1, 3], [1, 2, 3]]),     # 连接请求 4
    ...
]
```

#### 步骤 4: 初始化 RWA 分配和波长占用矩阵

```python
if rwa_assignment_previous is not None:
    rwa_assignment = rwa_assignment_previous
    W = np.zeros((len(list(self.graph.edges)), self.channels))
    for key in rwa_assignment:
        for path in rwa_assignment[key]:
            W = self.add_wavelength_path_to_W(self.graph, W, path, key)
else:
    rwa_assignment = {i: [] for i in range(self.channels)}
    W = np.zeros((len(list(self.graph.edges)), self.channels))
```

**功能**: 
- `rwa_assignment`: RWA 分配字典，格式为 `{波长索引: [路径列表], ...}`
- `W`: 波长占用矩阵，形状为 `(边数, 波长数)`
  - `W[i, j] = 1` 表示边 i 上波长 j 已被占用
  - `W[i, j] = 0` 表示边 i 上波长 j 可用

#### 步骤 5: 遍历连接请求并分配波长

```python
for idx, item in enumerate(s_d_pairs_with_traffic):
    path_wavelength = np.zeros((len(item[2]),), dtype=np.int)
    paths_wavelengths = []
    
    # 为每条候选路径找到可用波长
    for index, path in enumerate(item[2]):
        FF = self.FF_return(self.graph, self.channels, W, path)
        path_wavelength[index] = FF
        paths_wavelengths.append((path, FF))
    
    # 选择波长索引最小的路径（First Fit 策略）
    index_array = np.argsort(path_wavelength)
    index_min = index_array[0]
    
    # 检查是否阻塞
    if path_wavelength[index_min] == self.channels + 1:
        if return_blocked:
            blocked_connections += 1
            continue
        else:
            return True  # 路由失败
    
    # 分配路径和波长
    rwa_assignment[path_wavelength[index_min]].append(item[2][index_min])
    W = self.add_wavelength_path_to_W(self.graph, W, item[2][index_min], 
                                      path_wavelength[index_min])
```

**关键逻辑**:

1. **为每条路径找波长**: 对每个节点对的 k 条路径，使用 `FF_return` 找到第一个可用波长
2. **选择最佳路径**: 选择波长索引最小的路径（First Fit 策略）
3. **检查阻塞**: 如果所有路径都没有可用波长（返回 `channels + 1`），则阻塞
4. **更新分配**: 将路径添加到对应波长的列表中
5. **更新占用矩阵**: 更新 `W` 矩阵，标记路径上所有边和波长为已占用

### 1.5 FF_return 函数详解

**位置**: `networktoolbox/NetworkToolkit/Routing/Heuristics.py`

```python
def FF_return(self, graph, channels, W, path, return_all=False):
    """
    First Fit 波长分配算法
    :param graph: NetworkX 图对象
    :param channels: 可用波长数量
    :param W: 交换矩阵 (E, channels)，W[i,j]=1 表示边 i 上波长 j 被占用
    :param path: 节点路径列表，例如 [1, 3, 5, 7]
    :param return_all: 是否返回所有可用波长
    :return: 可用波长的索引，如果没有则返回 channels + 1
    """
    # 创建路径向量 P
    P = np.zeros((len(list(graph.edges)), 1))
    graph_edges = list(graph.edges())
    path_edges = nodes_to_edges(path)
    
    for index, (s,d) in enumerate(graph_edges):
        if (s,d) in path_edges or (d,s) in path_edges:
            P[index] = 1
    
    # 计算每个波长在路径上的占用情况
    a = np.einsum('ij,ij->j', W, P)
    
    # 找到所有可用波长（占用为 0）
    indeces = np.where(a == 0)
    
    try:
        if not return_all:
            min_wave = np.min(indeces)  # 返回第一个可用波长
        else:
            min_wave = indeces
    except Exception as err:
        return channels + 1  # 没有可用波长
    
    return min_wave
```

**算法原理**:
- 路径向量 `P`: 标识路径经过哪些边
- 矩阵运算: `a = W · P`，计算每个波长在路径上的占用情况
- First Fit: 返回第一个可用波长（索引最小）

### 1.6 返回值

```python
if return_blocked and not save_ids:
    return rwa_assignment, blocked_connections
elif return_blocked and save_ids:
    return rwa_assignment, blocked_connections, additional_id_info
else:
    return rwa_assignment
```

**返回格式**:
- `rwa_assignment`: `{波长索引: [路径列表], ...}`
- `blocked_connections`: 阻塞的连接数（如果 `return_blocked=True`）
- `additional_id_info`: 连接 ID 信息（如果 `save_ids=True`）

---

## 2. k-SP-FF 算法详细分析

### 2.1 算法概述

**k-SP-FF (k-Shortest Paths First Fit)** 是另一种启发式 RWA 算法，与 FF-kSP 的主要区别在于**路径选择策略**：
- **FF-kSP**: 优先选择波长索引最小的路径（First Fit 策略）
- **k-SP-FF**: 优先选择路径长度最短的路径（在可用波长中）

### 2.2 函数签名

**位置**: `networktoolbox/NetworkToolkit/Routing/Heuristics.py`

```python
def k_SP_FF_revised(self, traffic_matrix_connection_requests=None, e=0, k=1, 
                    order_aware=False, sort_length=False,
                    rwa_assignment_previous=None, return_blocked=False, 
                    random_sorting=False, connection_pairs=None,
                    _ids=None, save_ids=False):
```

### 2.3 参数说明

参数与 `FF_kSP` 基本相同，详见 [FF-kSP 参数说明](#13-参数说明)。

### 2.4 算法实现详解

#### 步骤 1: 计算 k 最短路径

```python
if traffic_matrix_connection_requests is not None:
    if self.ksp is None:
        k_SP = get_k_shortest_paths_MNH(self.graph, e=e, k=k)
    else:
        k_SP = self.ksp
    s_d_pairs_with_traffic = self.k_sp_to_s_d_pairs_with_traffic(k_SP, traffic_matrix_connection_requests)
```

**关键差异**: 
- 使用 `k_sp_to_s_d_pairs_with_traffic` 函数处理数据，而不是手动转换
- 支持使用预计算的 k 最短路径（`self.ksp`）

#### 步骤 2: k_sp_to_s_d_pairs_with_traffic 函数

```python
def k_sp_to_s_d_pairs_with_traffic(self, ksp, traffic_matrix_connections_requests):
    ksp = list(map(lambda x: (x[0][0], x[0][1], x[1]), ksp))
    s_d_pairs_with_traffic = []
    for item in ksp:
        s_d_pairs_with_traffic += [[tuple_item for tuple_item in item]] * int(
            traffic_matrix_connections_requests[item[0] - 1, item[1] - 1])
    return s_d_pairs_with_traffic
```

**功能**: 
- 将 k 最短路径数据转换为 `(源节点, 目的节点, 路径列表)` 格式
- 根据流量矩阵的需求数量，为每个节点对创建多个连接请求

**返回格式**:
```python
[
    [1, 2, [[1, 3, 2], [1, 4, 2]]],  # 连接请求 1
    [1, 2, [[1, 3, 2], [1, 4, 2]]],  # 连接请求 2（如果 T_c[0,1]=2）
    [1, 3, [[1, 3], [1, 2, 3]]],     # 连接请求 3
    ...
]
```

#### 步骤 3: 初始化 RWA 分配和波长占用矩阵

与 `FF_kSP` 相同，初始化 `rwa_assignment` 和 `W` 矩阵。

#### 步骤 4: 遍历连接请求并分配波长（关键差异）

```python
for idx, item in enumerate(s_d_pairs_with_traffic):
    path_wavelength = np.zeros((len(item[2]),), dtype=np.int)
    path_len = np.zeros((len(item[2]),), dtype=np.int)
    paths_wavelengths = []
    
    # 为每条候选路径找可用波长
    for index, path in enumerate(item[2]):
        FF = self.FF_return(self.graph, self.channels, W, path)
        path_wavelength[index] = FF
        paths_wavelengths.append((path, FF))
        
        # 关键：记录路径长度（仅当有可用波长时）
        if FF > self.channels:
            pass  # 没有可用波长
        elif FF <= self.channels:
            path_len[index] = len(path)  # 记录路径长度
    
    # 关键差异：过滤掉没有可用波长的路径
    path_wavelength = path_wavelength[np.where(path_len != 0)]
    paths = np.asarray(item[2])[np.where(path_len != 0)]
    paths_wavelengths = np.asarray(paths_wavelengths)[np.where(path_len != 0)]
    path_len = path_len[np.where(path_len != 0)]
    
    # 检查是否所有路径都被阻塞
    if len(path_len) == 0:
        if return_blocked:
            blocked_connections += 1
            continue
        else:
            return True  # 路由失败
    
    # 关键差异：按路径长度排序，选择最短路径
    index_array = np.argsort(path_len)  # 按路径长度排序
    index_min = index_array[0]  # 选择最短路径
    
    # 分配路径和波长
    rwa_assignment[paths_wavelengths[index_min][1]].append(
        paths_wavelengths[index_min][0])
    W = self.add_wavelength_path_to_W(self.graph, W, 
                                      paths_wavelengths[index_min][0],
                                      paths_wavelengths[index_min][1])
```

**关键差异点**:

1. **路径长度记录**: 
   - `k_SP_FF_revised` 会记录每条有可用波长的路径的长度
   - 只考虑有可用波长的路径

2. **路径过滤**:
   - 过滤掉所有没有可用波长的路径
   - 只从有可用波长的路径中选择

3. **选择策略**:
   - **FF-kSP**: `np.argsort(path_wavelength)` - 按波长索引排序
   - **k-SP-FF**: `np.argsort(path_len)` - 按路径长度排序

### 2.5 算法流程图

```
输入: 流量矩阵 T_c, 图 G, 参数 k, e
  ↓
1. 计算所有节点对的 k 最短路径
  ↓
2. 使用 k_sp_to_s_d_pairs_with_traffic 处理数据
  ↓
3. 初始化 RWA 分配和波长占用矩阵 W
  ↓
4. 遍历每个连接请求:
   ├─ 为每条候选路径找可用波长 (FF_return)
   ├─ 记录有可用波长的路径长度
   ├─ 过滤掉没有可用波长的路径
   ├─ 按路径长度排序
   ├─ 选择路径长度最短的路径
   ├─ 检查是否阻塞
   ├─ 分配路径和波长
   └─ 更新波长占用矩阵 W
  ↓
输出: RWA 分配字典 {波长: [路径列表]}
```

### 2.6 返回值

与 `FF_kSP` 相同，返回 RWA 分配字典。

---

## 3. FF-kSP 与 k-SP-FF 算法对比

### 3.1 核心区别总结

| 维度 | FF-kSP | k-SP-FF |
|------|--------|---------|
| **算法名称** | First Fit k-Shortest Paths | k-Shortest Paths First Fit |
| **路径选择策略** | 优先选择波长索引最小的路径 | 优先选择路径长度最短的路径（在可用波长中） |
| **排序依据** | `np.argsort(path_wavelength)` | `np.argsort(path_len)` |
| **路径过滤** | 不过滤，所有路径都考虑 | 过滤掉没有可用波长的路径 |
| **数据预处理** | 手动转换数据结构 | 使用 `k_sp_to_s_d_pairs_with_traffic` 函数 |
| **路径长度考虑** | 不考虑路径长度 | 优先考虑路径长度 |

### 3.2 详细对比分析

#### 3.2.1 路径选择策略差异

**FF-kSP 策略**:
```python
# 为所有路径找波长（包括没有可用波长的）
for index, path in enumerate(item[2]):
    FF = self.FF_return(self.graph, self.channels, W, path)
    path_wavelength[index] = FF  # 可能是 channels + 1（阻塞）

# 直接按波长索引排序
index_array = np.argsort(path_wavelength)
index_min = index_array[0]  # 选择波长索引最小的路径
```

**k-SP-FF 策略**:
```python
# 为所有路径找波长，并记录路径长度
for index, path in enumerate(item[2]):
    FF = self.FF_return(self.graph, self.channels, W, path)
    path_wavelength[index] = FF
    if FF <= self.channels:
        path_len[index] = len(path)  # 只记录有可用波长的路径长度

# 过滤掉没有可用波长的路径
path_wavelength = path_wavelength[np.where(path_len != 0)]
paths = np.asarray(item[2])[np.where(path_len != 0)]
path_len = path_len[np.where(path_len != 0)]

# 按路径长度排序
index_array = np.argsort(path_len)
index_min = index_array[0]  # 选择路径长度最短的路径
```

#### 3.2.2 选择示例

假设节点对 (1, 5) 有 3 条路径：

| 路径 | 路径长度 | 可用波长 | FF-kSP 选择 | k-SP-FF 选择 |
|------|---------|---------|------------|-------------|
| [1, 2, 3, 5] | 3 | 波长 5 | ✓ (波长最小) | ✓ (路径最短) |
| [1, 4, 5] | 2 | 波长 8 | ✗ | ✓ (路径最短) |
| [1, 6, 7, 8, 5] | 4 | 无可用 | ✗ | ✗ (被过滤) |

**FF-kSP 选择**: 路径 [1, 2, 3, 5]，因为波长索引 5 最小

**k-SP-FF 选择**: 路径 [1, 4, 5]，因为路径长度 2 最短（在可用波长中）

#### 3.2.3 数据预处理差异

**FF-kSP**:
```python
s_d_pairs = list(map(lambda x: (x[0][0], x[0][1], x[1]), k_SP))
s_d_pairs = np.asarray(s_d_pairs)
s_d_pairs_with_traffic = np.zeros((3,))
for item in s_d_pairs:
    s_d_pairs_item = np.tile(item, (int(traffic_matrix_connection_requests[...]), 1))
    s_d_pairs_with_traffic = np.vstack((s_d_pairs_with_traffic, s_d_pairs_item))
s_d_pairs_with_traffic = np.delete(s_d_pairs_with_traffic, 0, 0)
```

**k-SP-FF**:
```python
s_d_pairs_with_traffic = self.k_sp_to_s_d_pairs_with_traffic(k_SP, traffic_matrix_connection_requests)
```

**差异**: k-SP-FF 使用封装好的函数，代码更简洁。

### 3.3 性能特点对比

#### 3.3.1 FF-kSP 特点

**优点**:
1. **波长利用率**: 优先使用低索引波长，可能提高波长利用率
2. **实现简单**: 逻辑直接，易于理解
3. **计算效率**: 不需要过滤路径，计算更快

**缺点**:
1. **路径长度**: 可能选择较长的路径，增加网络资源消耗
2. **阻塞率**: 在高负载情况下可能阻塞较多连接

#### 3.3.2 k-SP-FF 特点

**优点**:
1. **路径优化**: 优先选择最短路径，减少网络资源消耗
2. **阻塞处理**: 提前过滤无可用波长的路径，避免无效尝试
3. **代码质量**: 使用封装函数，代码更清晰

**缺点**:
1. **波长利用率**: 可能使用较高索引的波长
2. **计算开销**: 需要过滤路径，略微增加计算量

### 3.4 适用场景

#### FF-kSP 适用于:
- 波长资源充足的情况
- 需要最大化波长利用率
- 对路径长度不敏感的场景

#### k-SP-FF 适用于:
- 需要优化路径长度
- 网络资源有限的情况
- 需要减少网络延迟的场景

### 3.5 算法复杂度对比

| 操作 | FF-kSP | k-SP-FF |
|------|--------|---------|
| **路径查找** | O(k × E × W) | O(k × E × W) |
| **路径过滤** | O(1) | O(k) |
| **排序操作** | O(k log k) | O(k log k) |
| **总体复杂度** | O(k × E × W + k log k) | O(k × E × W + k log k) |

**结论**: 两种算法的复杂度基本相同，k-SP-FF 增加了路径过滤步骤，但影响很小。

### 3.6 实际性能对比

在实际网络仿真中，两种算法的性能差异取决于：

1. **网络拓扑**: 
   - 密集网络：k-SP-FF 可能表现更好（路径长度差异大）
   - 稀疏网络：FF-kSP 可能表现更好（路径长度差异小）

2. **流量负载**:
   - 低负载：两种算法性能相近
   - 高负载：k-SP-FF 可能阻塞更少（提前过滤）

3. **波长数量**:
   - 波长充足：FF-kSP 可能更优（波长利用率高）
   - 波长稀缺：k-SP-FF 可能更优（路径优化）

---

## 4. heuristic_throughput 函数详细分析

### 2.1 函数概述

**位置**: `networktoolbox/NetworkToolkit/NetworkSimulator.py`

**功能**: 使用启发式算法（如 FF-kSP）计算网络的最大吞吐量，通过逐步增加流量需求来找到网络容量上限。

### 2.2 函数签名

```python
@conditional_decorator(ray.remote(num_cpus=1, memory=3000 * 1024 * 1024), (not windows))
def heuristic_throughput(graph_list=None, collection=None, db="Topology_Data", 
                         e=10, k=10, route_function="FF-kSP",
                         m_step=100, max_count=10, channel_bandwidth=16e9, 
                         m_start=0, fibre_num=1, pb_actor=None):
```

### 2.3 参数说明

| 参数 | 类型 | 说明 |
|------|------|------|
| `graph_list` | list | 图列表，每个元素为 `(graph, _id, T_c)` |
| `collection` | str | MongoDB 集合名称 |
| `db` | str | MongoDB 数据库名称 |
| `e` | int | 路径长度参数 |
| `k` | int | k 最短路径数量 |
| `route_function` | str | 路由函数名称（如 "FF-kSP"） |
| `m_step` | int | 流量增长步长 |
| `max_count` | int | 最大迭代次数 |
| `channel_bandwidth` | float | 信道带宽（Hz） |
| `m_start` | int | 起始流量倍数 |
| `fibre_num` | int | 光纤数量 |
| `pb_actor` | Ray Actor | 进度条 Actor（用于显示进度） |

### 2.4 算法实现详解

#### 步骤 1: 初始化和网络创建

```python
for graph, _id, T_c in graph_list:
    graph = nx.relabel.convert_node_labels_to_integers(graph, first_label=1)
    assert type(graph) == nx.classes.graph.Graph
    assert nx.is_connected(graph) is True
    network = Network.OpticalNetwork(graph, channel_bandwidth=channel_bandwidth, 
                                     routing_func=route_function, fibre_num=fibre_num)
```

**功能**:
- 重新标记节点编号（从 1 开始）
- 验证图类型和连通性
- 创建光学网络对象，指定路由函数

#### 步骤 2: 初始化变量

```python
rwa_assignment = False
time_start = time.perf_counter()
M = m_start
demand_matrix_old = np.zeros((len(graph), len(graph)))
rwa_active = None
success = False
```

**变量说明**:
- `M`: 流量倍数，用于计算实际流量需求 `M × T_c`
- `demand_matrix_old`: 上一次成功分配的流量矩阵
- `rwa_active`: 当前活跃的 RWA 分配
- `success`: 是否成功分配过

#### 步骤 3: 二分搜索最大流量（外层循环）

```python
for i in range(max_count):
    while rwa_assignment != True:
        M += m_step
        demand_matrix_new = np.ceil(np.array(T_c) * M)
        # ... 路由分配 ...
```

**策略**: 使用二分搜索找到最大可路由的流量

**流程**:
1. 逐步增加流量倍数 `M`
2. 计算新的流量需求矩阵
3. 尝试路由分配
4. 如果成功，继续增加；如果失败，回退并减小步长

#### 步骤 4: 增量路由分配（FF-kSP 特有）

```python
if route_function == "FF-kSP" or "kSP-FF":
    if not success:
        # 第一次分配：分配全部需求
        rwa_assignment = network.route(demand_matrix_new - demand_matrix_old, e=e, k=k)
    elif (demand_matrix_new - demand_matrix_old).sum() > 0:
        # 增量分配：只分配新增的需求
        rwa_assignment = network.route(demand_matrix_new - demand_matrix_old, e=e, k=k,
                                       rwa_assignment_previous=rwa_assignment)
    
    if rwa_assignment != True:
        rwa_active = rwa_assignment
        demand_matrix_old = demand_matrix_new
        success = True
```

**关键特性**:
- **增量分配**: 只路由新增的流量需求，保留之前的分配
- **效率优化**: 避免重复计算已分配的连接

**示例**:
- 第一次: `M=100`, 分配 100 个连接 → 成功
- 第二次: `M=200`, 只分配新增的 100 个连接 → 成功
- 第三次: `M=300`, 只分配新增的 100 个连接 → 失败

#### 步骤 5: 二分搜索细化

```python
if int(M) > 1:
    M -= m_step  # 回退到上次成功的值

demand_matrix = np.ceil(np.array(T_c) * M)
if route_function == "FF-kSP" or "kSP-FF":
    rwa_assignment = rwa_active  # 使用上次成功的分配
else:
    rwa_assignment = network.route(demand_matrix_new, e=e, k=k)

m_step /= 2  # 减小步长
m_step = np.ceil(m_step)
```

**策略**: 
- 当路由失败时，回退到上次成功的流量值
- 减小步长，进行更精细的搜索
- 重复直到找到最大可路由流量

#### 步骤 6: 计算吞吐量

```python
network.physical_layer.add_uniform_launch_power_to_links(network.channels)

if fibre_num > 1:
    # 多光纤情况
    rwa_assignment = Tools.single_to_multi_fibre_rwa(rwa_assignment, 
                                                      network.routing_channels,
                                                      network.channels)
    throughput = 0
    for i in range(fibre_num):
        network.physical_layer.add_wavelengths_to_links(rwa_assignment[i])
        network.physical_layer.add_non_linear_NSR_to_links()
        throughput += network.physical_layer.get_lightpath_capacities_PLI(rwa_assignment[i])[0]
else:
    # 单光纤情况
    network.physical_layer.add_wavelengths_to_links(rwa_assignment)
    network.physical_layer.add_non_linear_NSR_to_links()
    throughput = network.physical_layer.get_lightpath_capacities_PLI(rwa_assignment)[0]
```

**流程**:
1. **设置发射功率**: 为所有边设置统一的发射功率
2. **添加波长分配**: 将 RWA 结果添加到图的边属性中
3. **计算非线性噪声**: 计算每条边上的 NSR（噪声信号比）
4. **计算容量**: 使用 Shannon 公式计算每条光路的容量
5. **累加吞吐量**: 如果是多光纤，累加所有光纤的吞吐量

#### 步骤 7: 保存结果到数据库

```python
Database.update_data_with_id(db, collection, _id,
                            newvals={"$set": {
                                "{} RWA".format(route_function): rwa_write,
                                "{}-connections".format(route_function): M,
                                "{} Capacity".format(route_function): throughput,
                                "{} time".format(route_function): time_taken,
                                "{} e".format(route_function): e,
                                "{} k".format(route_function): k,
                                # ... 更多参数 ...
                            }})
```

**保存的数据**:
- RWA 分配结果
- 最大连接数 `M`
- 总吞吐量
- 计算时间
- 算法参数

### 2.5 算法流程图

```
开始
  ↓
初始化网络和变量
  ↓
for i in range(max_count):
  ↓
  while rwa_assignment != True:
    ↓
    M += m_step
    demand_matrix = T_c × M
    ↓
    尝试路由分配
    ↓
    成功? ──No──→ 继续增加 M
    Yes
    ↓
    保存分配结果
    ↓
  M -= m_step  (回退)
  m_step /= 2  (减小步长)
  ↓
计算物理层吞吐量
  ↓
保存结果到数据库
  ↓
结束
```

---

## 5. ff_k_sp.py 脚本分析

### 3.1 脚本概述

**位置**: `thesis/throughput_upperbound/simulations/ff_k_sp.py`

**功能**: 使用 FF-kSP 算法计算网络吞吐量的主脚本

### 3.2 主要代码分析

#### 3.2.1 参数解析

```python
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-mc", type=int, default=1, help="maximum number of graphs to read")
    return parser.parse_args()
```

#### 3.2.2 数据读取

```python
graph_list = nt.Database.read_topology_dataset_list(db, collection, 
                                                    find_dic={"name": "NSFNET"},
                                                    node_data=True)
```

**功能**: 从 MongoDB 读取 NSFNET 拓扑图

#### 3.2.3 流量矩阵生成

```python
matrix_one = np.ones((len(graph_list[0][0].nodes), len(graph_list[0][0].nodes)))
np.fill_diagonal(matrix_one, 0)
T_c = (matrix_one / (len(graph_list[0][0].nodes) * (len(graph_list[0][0].nodes) - 1))).tolist()
```

**功能**: 生成均匀流量矩阵
- 所有节点对之间的流量需求相等
- 每个节点对的流量 = `1 / (n × (n-1))`
- 总流量需求 = 1（归一化）

#### 3.2.4 并行计算调用

```python
result = nt.NetworkSimulator.parralel_heuristic_throughput(
    graph_list, 
    collection=collection, 
    db=db,
    workers=len(graph_list),
    route_function="FF-kSP",
    e=100, 
    k=5, 
    m_step=200, 
    channel_bandwidth=50e9,
    max_count=10,
    m_start=0,
    port=port,
    hostname=hostname, 
    fibre_num=1
)
```

**参数说明**:
- `e=100`: 允许路径长度比最短路径长 100 跳（非常宽松）
- `k=5`: 每个节点对计算 5 条最短路径
- `m_step=200`: 流量增长步长为 200
- `channel_bandwidth=50e9`: 信道带宽 50 GHz

#### 3.2.5 结果读取和显示

```python
results = list(nt.Database.read_data(db, collection, 
                                     find_dic={"_id": _id},
                                     max_count=1))
if len(results) > 0:
    result_data = results[0]
    throughput = result_data["FF-kSP Capacity"]
    connections = result_data["FF-kSP-connections"]
    time_taken = result_data["FF-kSP time"]
    print(f"Throughput: {throughput:.2e} bps ({throughput/1e12:.4f} Tbps)")
    print(f"Max Connections M: {connections}")
    print(f"Computation Time: {time_taken:.2f} 秒")
```

### 3.3 parralel_heuristic_throughput 函数

**位置**: `networktoolbox/NetworkToolkit/NetworkSimulator.py`

```python
def parralel_heuristic_throughput(graph_list, collection=None, db="Topology_Data", 
                                  workers=50, route_function="FF-kSP",
                                  e=10, k=10, m_step=100,
                                  channel_bandwidth=16e9, max_count=10, m_start=0, 
                                  port=6379, hostname="128.40.43.93", fibre_num=1):
    ray.shutdown()
    if hostname is not None:
        ray.init(address='{}:{}'.format(hostname, port), 
                 _redis_password='5241590000000000',
                 ignore_reinit_error=True)
    else:
        ray.init()
    
    indeces = Tools.create_start_stop_list(len(graph_list), workers)
    pb = Tools.ProgressBar(workers)
    actor = pb.actor
    
    results = [heuristic_throughput.remote(db=db,
                                          collection=collection,
                                          graph_list=graph_list[indeces[ind]:indeces[ind + 1]],
                                          route_function=route_function, e=e, k=k, m_step=m_step,
                                          channel_bandwidth=channel_bandwidth,
                                          max_count=max_count, m_start=m_start, 
                                          fibre_num=fibre_num, pb_actor=actor)
              for ind in range(workers)]
    pb.print_until_done()
    results = ray.get(results)
```

**功能**: 
- 初始化 Ray 集群（本地或远程）
- 将图列表分配给多个 worker
- 并行执行 `heuristic_throughput`
- 显示进度条
- 收集所有结果

---

## 6. 算法流程总结

### 4.1 FF-kSP 算法完整流程

```
输入: 流量矩阵 T_c, 图 G, 参数 k, e
  ↓
1. 计算所有节点对的 k 最短路径
  ↓
2. 根据流量矩阵扩展连接请求
  ↓
3. 初始化 RWA 分配和波长占用矩阵 W
  ↓
4. 遍历每个连接请求:
   ├─ 为每条候选路径找可用波长 (FF_return)
   ├─ 选择波长索引最小的路径
   ├─ 检查是否阻塞
   ├─ 分配路径和波长
   └─ 更新波长占用矩阵 W
  ↓
输出: RWA 分配字典 {波长: [路径列表]}
```

### 4.2 heuristic_throughput 完整流程

```
输入: 图列表, 流量矩阵 T_c, 参数
  ↓
1. 初始化网络和变量
  ↓
2. 二分搜索最大流量:
   ├─ 增加流量倍数 M
   ├─ 计算需求矩阵 = T_c × M
   ├─ 调用路由算法 (FF-kSP)
   ├─ 成功? → 继续增加
   └─ 失败? → 回退并减小步长
  ↓
3. 计算物理层吞吐量:
   ├─ 设置发射功率
   ├─ 添加波长分配
   ├─ 计算非线性噪声 (NSR)
   └─ 计算容量 (Shannon 公式)
  ↓
4. 保存结果到数据库
  ↓
输出: 最大连接数 M, 总吞吐量
```

---

## 7. 关键数据结构

### 5.1 RWA 分配字典

```python
rwa_assignment = {
    0: [[1, 2, 3], [4, 5, 6]],      # 波长 0 用于两条路径
    1: [[1, 4, 3]],                  # 波长 1 用于一条路径
    2: [[2, 5, 7, 9], [3, 6, 8]],   # 波长 2 用于两条路径
}
```

### 5.2 波长占用矩阵 W

```python
W = np.zeros((边数, 波长数))
# W[i, j] = 1 表示边 i 上波长 j 被占用
# W[i, j] = 0 表示边 i 上波长 j 可用
```

### 5.3 流量矩阵 T_c

```python
T_c = np.array([
    [0, 0.1, 0.2, ...],  # 节点 1 到其他节点的流量需求
    [0.1, 0, 0.15, ...], # 节点 2 到其他节点的流量需求
    ...
])
```

### 5.4 连接请求列表

```python
s_d_pairs_with_traffic = [
    (源节点, 目的节点, [路径1, 路径2, ..., 路径k]),
    (源节点, 目的节点, [路径1, 路径2, ..., 路径k]),
    ...
]
```

---

## 8. 算法特点分析

### 8.1 FF-kSP 算法特点

**优点**:
1. **简单高效**: 实现简单，计算速度快
2. **路径多样性**: 使用 k 条路径，提供路径选择灵活性
3. **增量分配**: 支持增量路由，可以基于之前的分配继续分配
4. **First Fit 策略**: 简单直观，易于实现
5. **波长优化**: 优先使用低索引波长，可能提高波长利用率

**缺点**:
1. **非最优**: 是启发式算法，不保证最优解
2. **路径长度**: 可能选择较长的路径，增加网络资源消耗
3. **阻塞率**: 在高负载情况下可能阻塞较多连接

### 8.2 k-SP-FF 算法特点

**优点**:
1. **路径优化**: 优先选择最短路径，减少网络资源消耗
2. **阻塞处理**: 提前过滤无可用波长的路径，避免无效尝试
3. **代码质量**: 使用封装函数，代码更清晰
4. **增量分配**: 支持增量路由，可以基于之前的分配继续分配
5. **路径多样性**: 使用 k 条路径，提供路径选择灵活性

**缺点**:
1. **非最优**: 是启发式算法，不保证最优解
2. **波长利用率**: 可能使用较高索引的波长
3. **计算开销**: 需要过滤路径，略微增加计算量

### 8.3 heuristic_throughput 函数特点

**优点**:
1. **自动搜索**: 自动找到最大可路由流量
2. **二分搜索**: 使用二分搜索策略，效率较高
3. **增量路由**: FF-kSP 和 k-SP-FF 都支持增量路由，提高效率
4. **物理层建模**: 考虑物理层损伤，计算真实吞吐量
5. **算法选择**: 支持通过 `route_function` 参数选择不同算法

**缺点**:
1. **计算时间长**: 需要多次迭代和路由计算
2. **参数敏感**: 步长 `m_step` 的选择影响搜索精度和速度
3. **内存占用**: 需要存储多个 RWA 分配结果

---

## 9. 性能优化建议

### 9.1 FF-kSP 优化

1. **并行计算 k 最短路径**: 使用 Ray 并行计算
2. **缓存路径**: 如果多次调用，可以缓存 k 最短路径
3. **优化数据结构**: 使用更高效的数据结构存储路径

### 9.2 k-SP-FF 优化

1. **并行计算 k 最短路径**: 使用 Ray 并行计算
2. **缓存路径**: 如果多次调用，可以缓存 k 最短路径
3. **优化过滤操作**: 使用向量化操作提高路径过滤效率
4. **预计算路径长度**: 可以预先计算路径长度，避免重复计算

### 9.3 heuristic_throughput 优化

1. **自适应步长**: 根据网络规模动态调整 `m_step`
2. **提前终止**: 如果连续多次失败，可以提前终止
3. **并行处理**: 使用 Ray 并行处理多个图
4. **算法选择**: 根据网络特性选择合适的算法（FF-kSP 或 k-SP-FF）

---

## 10. 使用示例

### 8.1 基本使用

```python
import NetworkToolkit as nt
import numpy as np

# 读取图
graph_list = nt.Database.read_topology_dataset_list("Topology_Data", "topology-paper",
                                                    find_dic={"name": "NSFNET"},
                                                    node_data=True)

# 生成流量矩阵
matrix_one = np.ones((len(graph_list[0][0].nodes), len(graph_list[0][0].nodes)))
np.fill_diagonal(matrix_one, 0)
T_c = (matrix_one / (len(graph_list[0][0].nodes) * (len(graph_list[0][0].nodes) - 1))).tolist()
graph_list = [(graph, _id, T_c) for graph, _id in graph_list]

# 运行计算
nt.NetworkSimulator.parralel_heuristic_throughput(
    graph_list,
    collection="topology-paper",
    db="Topology_Data",
    route_function="FF-kSP",
    e=100,
    k=5,
    m_step=200,
    channel_bandwidth=50e9,
    max_count=10,
    m_start=0,
    fibre_num=1
)
```

### 8.2 参数调优

- **e 参数**: 较大的 e 值允许更长的路径，但可能降低性能
- **k 参数**: 较大的 k 值提供更多路径选择，但计算时间增加
- **m_step 参数**: 较小的步长提高精度，但增加迭代次数

---

## 11. 总结

本文档详细分析了 FF-kSP、k-SP-FF 算法和 `heuristic_throughput` 函数的实现：

1. **FF-kSP 算法**: 
   - 结合 k 最短路径和 First Fit 波长分配
   - 优先选择波长索引最小的路径
   - 支持增量路由分配
   - 实现简单，计算高效

2. **k-SP-FF 算法**:
   - 结合 k 最短路径和 First Fit 波长分配
   - 优先选择路径长度最短的路径（在可用波长中）
   - 过滤无可用波长的路径
   - 代码更清晰，路径更优化

3. **算法对比**:
   - **选择策略**: FF-kSP 按波长索引，k-SP-FF 按路径长度
   - **性能**: 取决于网络拓扑和负载情况
   - **适用场景**: FF-kSP 适合波长优化，k-SP-FF 适合路径优化

4. **heuristic_throughput 函数**:
   - 使用二分搜索找到最大可路由流量
   - 考虑物理层损伤计算真实吞吐量
   - 支持并行计算和进度显示
   - 支持两种算法（通过 `route_function` 参数选择）

5. **应用场景**:
   - 网络容量评估
   - 拓扑设计优化
   - 算法性能比较
   - RWA 算法研究

---

**文档生成时间**: 2024年
**最后更新**: 2024年

