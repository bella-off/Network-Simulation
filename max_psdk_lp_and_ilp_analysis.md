# max_psdk_lp 和 max_psdk_ilp 详细分析文档

本文档详细分析了 `thesis/throughput_upperbound/simulations` 文件夹中的两个脚本：`max_psdk_lp.py` 和 `max_psdk_ilp.py`，它们用于计算网络吞吐量的上界。

---

## 目录

1. [概述](#1-概述)
2. [max_psdk_lp.py 详细分析](#2-max_psdk_lppy-详细分析)
3. [max_psdk_ilp.py 详细分析](#3-max_psdk_ilppy-详细分析)
4. [核心函数：maximise_throughput_p](#4-核心函数maximise_throughput_p)
5. [核心函数：throughput_upper_bound_bottle_neck](#5-核心函数throughput_upper_bound_bottle_neck)
6. [算法对比](#6-算法对比)
7. [数学原理](#7-数学原理)
8. [使用场景](#8-使用场景)

---

## 1. 概述

### 1.1 功能说明

**`max_psdk_lp.py`** 和 **`max_psdk_ilp.py`** 都用于计算光学网络吞吐量的上界，但使用不同的方法：

- **`max_psdk_lp.py`**: 使用**线性规划（LP）**方法，通过优化路径概率分布 `p_sdk` 来最大化吞吐量上界
- **`max_psdk_ilp.py`**: 从已有的**整数线性规划（ILP）RWA 结果**中计算吞吐量上界

### 1.2 关键概念

- **`p_sdk`**: 路径概率分布，表示节点对 `(s, d)` 使用第 `k` 条路径的概率
- **`theta_star`**: 吞吐量上界（最大可路由的流量倍数）
- **`I_nsdk`**: 节点路径指示器，表示节点 `n` 是否在节点对 `(s, d)` 的第 `k` 条路径上
- **`I_esdk`**: 边路径指示器，表示边 `(e1, e2)` 是否在节点对 `(s, d)` 的第 `k` 条路径上

---

## 2. max_psdk_lp.py 详细分析

### 2.1 文件位置

**位置**: `thesis/throughput_upperbound/simulations/max_psdk_lp.py`

### 2.2 源代码分析

```python
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

    # 从 pickle 文件读取数据
    graph_list = utils.read_data("/rdata/ong/robin/thesis/throughput_bound/LP-{}.pkl".format(coll))
    
    # 调用并行计算函数
    nt.NetworkSimulator.parralel_max_psdk_edge_bound(
        graph_list, 
        db=db, 
        collection=coll, 
        k=5,
        e=None, 
        threads=1, 
        workers=len(graph_list), 
        local=True, 
        bandwidth=50e9, 
        channels=100, 
        hostname=hostname, 
        port=port
    )
```

### 2.3 功能详解

#### 2.3.1 数据读取

- **数据源**: 从 pickle 文件读取预处理的数据
- **数据格式**: `graph_list = [(graph, _id, T_c), ...]`
  - `graph`: NetworkX 图对象
  - `_id`: MongoDB 文档 ID
  - `T_c`: 归一化流量矩阵

#### 2.3.2 核心函数调用

调用 `parralel_max_psdk_edge_bound` 函数，该函数：

1. **初始化 Ray 集群**: 用于并行计算
2. **分配任务**: 将图列表分配给多个 worker
3. **并行执行**: 每个 worker 调用 `calculate_max_psdk_edge_bound`
4. **计算吞吐量上界**: 使用 LP 方法，基于**边约束**（`edge_bound=True`）

#### 2.3.3 关键参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `k` | 5 | k 最短路径数量 |
| `e` | None | 路径长度参数（无限制） |
| `threads` | 1 | Gurobi 求解器线程数 |
| `workers` | len(graph_list) | 并行 worker 数量 |
| `local` | True | 使用本地 Ray 集群 |
| `bandwidth` | 50e9 | 信道带宽（50 GHz） |
| `channels` | 100 | 波长数量 |

### 2.4 计算流程

```
1. 读取 graph_list（包含 graph, _id, T_c）
   ↓
2. 初始化 OpticalNetwork 对象
   ↓
3. 调用 maximise_throughput_p(T_c*100000, edge_bound=True)
   ↓
4. LP 优化：
   - 变量: p_sdk (连续变量), theta_star (连续变量)
   - 约束: 路径概率约束、节点资源约束、边资源约束
   - 目标: 最小化 theta_star（等价于最大化吞吐量）
   ↓
5. 返回结果:
   - p_sdk: 最优路径概率分布
   - theta_star: 吞吐量上界
   ↓
6. 保存到数据库:
   - "max_e p_sdk": 路径概率分布
   - "theta_star_e": 基于边的吞吐量上界
   - "max_e p_sdk LP assigned": 可分配的连接数
```

### 2.5 数据库输出字段

结果保存到 MongoDB，包含以下字段：

- `max_e p_sdk`: 最优路径概率分布
- `max_e p_sdk LP gap`: LP 求解的最优性间隙
- `max_e p_sdk LP written`: 写入标志（1）
- `max_e p_sdk LP time`: 计算时间
- `max_e p_sdk LP status`: 求解状态
- `max_e p_sdk LP timestamp`: 时间戳
- `max_e p_sdk LP e`: 路径长度参数
- `max_e p_sdk LP k`: k 值
- `max_e p_sdk LP assigned`: 可分配的连接数
- `theta_star_e`: 基于边的吞吐量上界
- `max_e p_sdk T_c`: 流量矩阵

---

## 3. max_psdk_ilp.py 详细分析

### 3.1 文件位置

**位置**: `thesis/throughput_upperbound/simulations/max_psdk_ilp.py`

### 3.2 源代码分析

```python
import sys
import NetworkToolkit as nt
import numpy as np
import ray
import networkx as nx
import utils
import traceback
from datetime import datetime

# 辅助函数定义...

if __name__ == "__main__":
    db = "DRLRoutingDB"
    coll = "ilp_rwa_test"
    
    ray.shutdown()
    # 从数据库读取数据（包含 RWA 结果）
    graph_list_lp = nt.Database.read_topology_dataset_list(
        db, coll, 
        "T_c", "ILP-connections RWA", "ILP-connections M",
        find_dic={"ILP-connections RWA":{"$exists":True}}, 
        max_count=100, 
        parralel=False
    )
    
    # 保存到 pickle 文件
    utils.dump_data(graph_list_lp, "/rdata/ong/robin/thesis/throughput_bound/{}.pkl".format(coll))
    
    # 计算吞吐量上界（基于边约束和节点约束）
    throughput_upper_bound_bottle_neck_parralel(
        graph_list_lp, 
        lambda_n=100, 
        k=5, 
        db=db, 
        collection=coll, 
        min_edge_bound=True   # 基于边约束
    )
    throughput_upper_bound_bottle_neck_parralel(
        graph_list_lp, 
        lambda_n=100, 
        k=5, 
        db=db, 
        collection=coll, 
        min_edge_bound=False  # 基于节点约束
    )
```

### 3.3 功能详解

#### 3.3.1 数据读取

从数据库读取包含 ILP RWA 结果的数据：

- **数据格式**: `graph_list = [(graph, _id, T_c, rwa, M), ...]`
  - `graph`: NetworkX 图对象
  - `_id`: MongoDB 文档 ID
  - `T_c`: 归一化流量矩阵
  - `rwa`: ILP 路由和波长分配结果（字典：`{波长: [路径列表]}`）
  - `M`: ILP 连接倍数

#### 3.3.2 核心函数

调用 `throughput_upper_bound_bottle_neck_parralel` 函数，该函数：

1. **初始化 Ray 集群**: 用于并行计算
2. **分配任务**: 将图列表分配给多个 worker
3. **并行执行**: 每个 worker 调用 `throughput_upper_bound_bottle_neck`
4. **计算吞吐量上界**: 从 RWA 结果中提取路径概率分布，然后计算上界

#### 3.3.3 关键参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `lambda_n` | 100 | 节点波长数量（每个节点的度数 × 波长数） |
| `k` | 5 | k 最短路径数量 |
| `min_edge_bound` | True/False | 是否使用边约束（True）或节点约束（False） |

### 3.4 计算流程

```
1. 读取 graph_list（包含 graph, _id, T_c, rwa, M）
   ↓
2. 从 RWA 结果计算路径概率分布：
   rwa_to_path_prob_dists(graph, rwa, k=k)
   ↓
3. 计算节点路径指示器：
   I_nsdk = node_in_path_indicator(graph, k=k)
   ↓
4. 计算路径概率分布矩阵：
   p_sdk = path_prob_dist_specific(graph, path_probs, k=k)
   ↓
5. 计算流量：
   - transit_traffic(n): 节点 n 的转接流量
   - nodal_traffic(n): 节点 n 的本地流量
   - edge_traffic(e1, e2): 边 (e1, e2) 的流量
   ↓
6. 计算吞吐量上界：
   - 节点约束: theta_star = min((degree[n] * lambda_n) / (transit + nodal))
   - 边约束: theta_star = min(lambda_n / edge_traffic)
   ↓
7. 保存到数据库:
   - "theta_star_ilp": 基于节点的吞吐量上界
   - "theta_star_min_ilp": 基于边的吞吐量上界
```

### 3.5 辅助函数

#### 3.5.1 `node_in_path_indicator`

```python
def node_in_path_indicator(graph, k=20):
    """计算节点路径指示器 I_nsdk[n, s, d, k]"""
    k_sp = nt.Routing.Tools.get_k_shortest_paths_MNH(graph, k=k, e=1000, data_dict=True)
    I_nsdk = np.zeros((len(graph), len(graph), len(graph), k))
    for n in range(len(graph)):
        node = n+1
        for s_d_ind, (s,d) in enumerate(k_sp):
            for ind_p, path in enumerate(k_sp[(s,d)]):
                if node in path:
                    I_nsdk[n, s-1, d-1, ind_p] = 1
    return I_nsdk
```

**功能**: 创建指示器矩阵，标记节点 `n` 是否在节点对 `(s, d)` 的第 `k` 条路径上。

#### 3.5.2 `rwa_to_path_prob_dists`

```python
def rwa_to_path_prob_dists(graph, rwa, k=20, e=1000):
    """从 RWA 结果计算路径概率分布"""
    k_sp = nt.Routing.Tools.get_k_shortest_paths_MNH(graph, k=k, e=e, data_dict=True)
    path_probs = {(s,d):[0 for i in range(k)] for s,d in k_sp}
    
    # 统计每个路径的使用次数
    for wave in rwa:
        for path in rwa[wave]:
            s, d = path[0], path[-1]
            path_ind = k_sp[(s,d)].index(path)
            path_probs[(s,d)][path_ind] += 1
            path_probs[(d,s)][path_ind] += 1  # 对称
    
    # 归一化
    for s,d in path_probs:
        normaliser = sum(path_probs[(s,d)])
        if normaliser == 0:
            continue
        for ind, item in enumerate(path_probs[(s,d)]):
            path_probs[(s,d)][ind] /= normaliser
            path_probs[(d,s)][ind] /= normaliser
    
    return path_probs
```

**功能**: 从 RWA 分配结果中提取路径使用频率，并归一化为概率分布。

#### 3.5.3 `edge_in_path_indicator`

```python
def edge_in_path_indicator(graph, k=20):
    """计算边路径指示器 I_esdk[e1, e2, s, d, k]"""
    k_sp = nt.Routing.Tools.get_k_shortest_paths_MNH(graph, k=k, e=1000, data_dict=True)
    I_esdk = np.zeros((len(graph), len(graph), len(graph), len(graph), k))
    for e1, e2 in graph.edges:
        for s_d_ind, (s,d) in enumerate(k_sp):
            for ind_p, path in enumerate(k_sp[(s,d)]):
                path_edges = nt.Tools.nodes_to_edges(path)
                if (e1, e2) in path_edges or (e2, e1) in path_edges:
                    I_esdk[e1-1, e2-1, s-1, d-1, ind_p] = 1
                    I_esdk[e2-1, e1-1, d-1, s-1, ind_p] = 1
    return I_esdk
```

**功能**: 创建指示器矩阵，标记边 `(e1, e2)` 是否在节点对 `(s, d)` 的第 `k` 条路径上。

### 3.6 数据库输出字段

结果保存到 MongoDB，包含以下字段：

**基于节点约束**:
- `theta_star_ilp`: 基于节点的吞吐量上界
- `theta_star_ilp LP assigned`: 可分配的连接数
- `ILP-connections LP assigned`: ILP 连接数

**基于边约束**:
- `theta_star_min_ilp`: 基于边的吞吐量上界
- `theta_star_min_ilp LP assigned`: 可分配的连接数

---

## 4. 核心函数：maximise_throughput_p

### 4.1 函数位置

**位置**: `networktoolbox/NetworkToolkit/Routing/ILP.py`

### 4.2 函数签名

```python
def maximise_throughput_p(self, T_c=None, e=1, k=5,
                          solver_name="GRB", max_time=1000, _id=0, 
                          node_file_start=0.1, threads=10,
                          emphasis=2, max_gap=1e-4, max_solutions=100, 
                          blocking_rate=0,
                          node_file_dir="/scratch/datasets/gurobi/nodefiles", 
                          edge_bound=False):
```

### 4.3 优化模型

#### 4.3.1 决策变量

- **`p_sdk[s][d][k]`**: 连续变量（`var_type="C"`），表示节点对 `(s, d)` 使用第 `k` 条路径的概率
- **`theta_star`**: 连续变量，表示吞吐量上界（最大可路由的流量倍数）

#### 4.3.2 约束条件

**约束 1: 路径概率范围**
```python
0 ≤ p_sdk[s][d][k] ≤ 1  # 对于所有 s, d, k
```

**约束 2: 路径概率归一化**
```python
∑_k p_sdk[s][d][k] = 1  # 对于所有 s, d (s ≠ d)
```

**约束 3: 节点资源约束（节点约束）**
```python
(2 * ∑_{s,d,k} T_c[s,d] * I_nsdk[n,s,d,k] * p_sdk[s,d,k]) / (degree[n] * W) 
+ (∑_d T_c[n,d]) / (degree[n] * W) 
≤ theta_star  # 对于所有节点 n
```

**约束 4: 边资源约束（边约束，当 edge_bound=True）**
```python
∑_{s,d,k} T_c[s,d] * I_esdk[e1,e2,s,d,k] * p_sdk[s,d,k] / W 
≤ theta_star  # 对于所有边 (e1, e2)
```

#### 4.3.3 目标函数

```python
minimize theta_star
```

**说明**: 最小化 `theta_star` 等价于最大化吞吐量，因为 `theta_star` 是吞吐量的上界。

### 4.4 数学原理

#### 4.4.1 节点资源约束

对于节点 `n`，总资源为 `degree[n] * W`（度数 × 波长数）。

**转接流量**（transit traffic）:
```
transit_traffic(n) = 2 * ∑_{s,d,k} T_c[s,d] * I_nsdk[n,s,d,k] * p_sdk[s,d,k]
```
其中因子 2 是因为路径是双向的。

**本地流量**（nodal traffic）:
```
nodal_traffic(n) = ∑_d T_c[n,d]
```

**资源利用率**:
```
utilization(n) = (transit_traffic(n) + nodal_traffic(n)) / (degree[n] * W)
```

**约束**: `utilization(n) ≤ theta_star`，因此：
```
theta_star ≥ (transit_traffic(n) + nodal_traffic(n)) / (degree[n] * W)
```

#### 4.4.2 边资源约束

对于边 `(e1, e2)`，总资源为 `W`（波长数）。

**边流量**:
```
edge_traffic(e1, e2) = ∑_{s,d,k} T_c[s,d] * I_esdk[e1,e2,s,d,k] * p_sdk[s,d,k]
```

**资源利用率**:
```
utilization(e1, e2) = edge_traffic(e1, e2) / W
```

**约束**: `utilization(e1, e2) ≤ theta_star`，因此：
```
theta_star ≥ edge_traffic(e1, e2) / W
```

#### 4.4.3 吞吐量上界

吞吐量上界为所有约束中最严格的一个：

**节点约束**:
```
theta_star_node = min_n (degree[n] * W) / (transit_traffic(n) + nodal_traffic(n))
```

**边约束**:
```
theta_star_edge = min_{e1,e2} W / edge_traffic(e1, e2)
```

**最终上界**:
```
theta_star = min(theta_star_node, theta_star_edge)
```

### 4.5 返回值

```python
{
    "status": status,           # 求解状态
    "gap": ILP.gap,            # 最优性间隙
    "p_sdk": p_sdk,            # 最优路径概率分布
    "theta_star": theta_upper_bound  # 吞吐量上界
}
```

---

## 5. 核心函数：throughput_upper_bound_bottle_neck

### 5.1 函数位置

**位置**: `thesis/throughput_upperbound/simulations/max_psdk_ilp.py`

### 5.2 函数签名

```python
@ray.remote
def throughput_upper_bound_bottle_neck(graph_list, lambda_n=10, k=20, 
                                       db="Topology_Data", collection=None, 
                                       pb_actor=None, min_edge_bound=False):
```

### 5.3 计算流程

#### 5.3.1 从 RWA 提取路径概率

```python
path_probs = rwa_to_path_prob_dists(graph, rwa, k=k)
p_sdk = path_prob_dist_specific(graph, path_probs, k=k)
```

#### 5.3.2 计算流量

**转接流量**:
```python
transit_traffic = lambda n: 2*np.sum([
    sum([sum([
        T_c[s,d] * I_nsdk[n, s, d, k] * p_sdk[s,d,k] 
        for k in range(k) 
        if n != s and n != d and d>s
    ]) for d in range(len(graph))]) 
    for s in range(len(graph))
])
```

**本地流量**:
```python
nodal_traffic = lambda n: np.sum([
    T_c[n, d] for d in range(len(graph)) if d != n
])
```

**边流量**（当 `min_edge_bound=True`）:
```python
edge_traffic = lambda e1, e2: np.sum([
    sum([sum([
        T_c[s,d] * I_esdk[e1, e2, s, d, k] * p_sdk[s,d,k] 
        for k in range(k) if d>s
    ]) for d in range(len(graph))]) 
    for s in range(len(graph))
])
```

#### 5.3.3 计算吞吐量上界

**节点约束**:
```python
theta_upper_bound_node = [
    (graph.degree[node] * lambda_n) / (transit_traffic(node-1) + nodal_traffic(node-1))
    for node in graph.nodes
]
theta_upper_bound = np.min(theta_upper_bound_node) / 2
```

**边约束**（当 `min_edge_bound=True`）:
```python
theta_upper_bound_edge = [
    lambda_n / edge_traffic(e1-1, e2-1) 
    for e1, e2 in graph.edges
]
theta_upper_bound = np.min(theta_upper_bound_edge) / 2
```

**注意**: 除以 2 是因为路径是双向的，需要避免重复计算。

---

## 6. 算法对比

### 6.1 方法对比表

| 维度 | max_psdk_lp.py | max_psdk_ilp.py |
|------|----------------|-----------------|
| **方法** | 线性规划（LP）优化 | 从 ILP RWA 结果计算 |
| **输入** | `(graph, _id, T_c)` | `(graph, _id, T_c, rwa, M)` |
| **优化变量** | `p_sdk`（连续） | 无（从 RWA 提取） |
| **约束类型** | 节点约束 + 边约束 | 节点约束 或 边约束 |
| **计算方式** | Gurobi LP 求解器 | 直接计算 |
| **时间复杂度** | O(多项式) | O(N² × k) |
| **精度** | 理论最优 | 依赖于 ILP RWA 质量 |
| **适用场景** | 理论分析、上界计算 | 实际 RWA 结果评估 |

### 6.2 优缺点对比

#### 6.2.1 max_psdk_lp.py

**优点**:
1. **理论最优**: LP 方法给出理论上的最优路径概率分布
2. **灵活性**: 可以优化路径选择，不受实际 RWA 限制
3. **上界准确**: 提供严格的吞吐量上界

**缺点**:
1. **计算复杂度**: 对于大规模网络，LP 求解可能较慢
2. **实际可行性**: 最优解可能无法通过实际 RWA 实现
3. **内存消耗**: 需要存储大量变量和约束

#### 6.2.2 max_psdk_ilp.py

**优点**:
1. **计算快速**: 直接计算，无需优化求解
2. **实际可行**: 基于实际 RWA 结果，结果更贴近实际
3. **内存效率**: 不需要存储优化模型

**缺点**:
1. **依赖 RWA 质量**: 结果依赖于 ILP RWA 的质量
2. **非最优**: 不是理论最优解
3. **需要预处理**: 需要先有 ILP RWA 结果

### 6.3 使用建议

**使用 `max_psdk_lp.py` 当**:
- 需要理论上的吞吐量上界
- 进行网络容量规划
- 评估网络拓扑的理论性能
- 不需要考虑实际 RWA 约束

**使用 `max_psdk_ilp.py` 当**:
- 已有 ILP RWA 结果
- 需要评估实际 RWA 方案的性能
- 进行快速评估
- 需要与实际实现对比

---

## 7. 数学原理

### 7.1 吞吐量上界公式

#### 7.1.1 节点约束上界

对于节点 `n`，吞吐量上界为：

```
theta_star_n = (degree[n] * W) / (transit_traffic(n) + nodal_traffic(n))
```

其中：
- `degree[n]`: 节点 `n` 的度数
- `W`: 波长数量
- `transit_traffic(n)`: 节点 `n` 的转接流量
- `nodal_traffic(n)`: 节点 `n` 的本地流量

#### 7.1.2 边约束上界

对于边 `(e1, e2)`，吞吐量上界为：

```
theta_star_e = W / edge_traffic(e1, e2)
```

其中：
- `W`: 波长数量
- `edge_traffic(e1, e2)`: 边 `(e1, e2)` 的流量

#### 7.1.3 全局上界

全局吞吐量上界为所有约束的最小值：

```
theta_star = min(min_n theta_star_n, min_e theta_star_e)
```

### 7.2 路径概率分布

路径概率分布 `p_sdk[s][d][k]` 满足：

1. **非负性**: `p_sdk[s][d][k] ≥ 0`
2. **归一化**: `∑_k p_sdk[s][d][k] = 1`（对于所有 `s, d`，`s ≠ d`）
3. **对称性**: `p_sdk[s][d][k] = p_sdk[d][s][k]`（如果路径是对称的）

### 7.3 流量计算

#### 7.3.1 转接流量

节点 `n` 的转接流量为所有经过该节点但不以该节点为源或目的地的流量：

```
transit_traffic(n) = 2 * ∑_{s,d,k} T_c[s,d] * I_nsdk[n,s,d,k] * p_sdk[s,d,k]
```

其中因子 2 是因为路径是双向的。

#### 7.3.2 本地流量

节点 `n` 的本地流量为所有以该节点为源或目的地的流量：

```
nodal_traffic(n) = ∑_d T_c[n,d] + ∑_s T_c[s,n]
```

由于 `T_c` 通常是对称的，可以简化为：

```
nodal_traffic(n) = 2 * ∑_d T_c[n,d]
```

#### 7.3.3 边流量

边 `(e1, e2)` 的流量为所有使用该边的路径的流量：

```
edge_traffic(e1, e2) = ∑_{s,d,k} T_c[s,d] * I_esdk[e1,e2,s,d,k] * p_sdk[s,d,k]
```

---

## 8. 使用场景

### 8.1 max_psdk_lp.py 使用场景

1. **网络容量规划**:
   - 评估网络的理论最大吞吐量
   - 为网络升级提供参考

2. **拓扑设计**:
   - 比较不同拓扑结构的性能
   - 优化网络结构

3. **理论研究**:
   - 分析网络性能的理论上界
   - 验证算法的性能

### 8.2 max_psdk_ilp.py 使用场景

1. **RWA 方案评估**:
   - 评估 ILP RWA 方案的性能
   - 比较不同 RWA 方案的效果

2. **性能分析**:
   - 分析实际网络的吞吐量上界
   - 识别网络瓶颈

3. **优化验证**:
   - 验证 RWA 方案是否接近理论上界
   - 评估优化算法的效果

---

## 9. 总结

### 9.1 核心区别

- **`max_psdk_lp.py`**: 使用 LP 优化方法，计算理论最优的路径概率分布和吞吐量上界
- **`max_psdk_ilp.py`**: 从实际 ILP RWA 结果中提取路径概率分布，计算实际的吞吐量上界

### 9.2 关键指标

- **`theta_star`**: 吞吐量上界，表示最大可路由的流量倍数
- **`p_sdk`**: 路径概率分布，表示路径选择策略
- **`I_nsdk` / `I_esdk`**: 路径指示器，标记节点/边是否在路径上

### 9.3 应用价值

两种方法结合使用，可以：
1. 通过 LP 方法获得理论上界
2. 通过 ILP 方法评估实际性能
3. 比较理论与实际的差距
4. 指导网络优化和设计

---

**文档生成时间**: 2024年
**最后更新**: 2024年

