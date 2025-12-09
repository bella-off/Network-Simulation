# ILP Connections 和 Cycle Graph ILP 分析文档

本文档详细分析了 `thesis/throughput_upperbound/simulations` 文件夹中的两个 ILP（整数线性规划）相关脚本：`ilp_connections.py` 和 `cycle_graph_ilp.py`。

---

## 目录

1. [ilp_connections.py](#1-ilp_connectionspy)
2. [cycle_graph_ilp.py](#2-cycle_graph_ilppy)
3. [核心功能对比](#3-核心功能对比)
4. [技术实现细节](#4-技术实现细节)
5. [使用场景](#5-使用场景)

---

## 1. ilp_connections.py

### 1.1 文件概述

**位置**: `thesis/throughput_upperbound/simulations/ilp_connections.py`

**功能**: 使用整数线性规划（ILP）优化算法，为光学网络拓扑计算最优的路由和波长分配（RWA），以最大化连接需求。

### 1.2 源代码分析

```python
import NetworkToolkit as nt
import numpy as np

if __name__ == "__main__":
    hostname = "128.40.41.48"
    port = 7112
    query = { "nodes" : 14, "ILP Capacity" : { "$exists" : True }, "ILP-connections" : { "$exists" : False }}
    graph_list = nt.Database.read_topology_dataset_list("Topology_Data", "topology-paper", 
                                                    find_dic=query,
                                                    node_data=False, max_count=10000)
    matrix_one = np.ones((len(graph_list[0][0].nodes), len(graph_list[0][0].nodes)))
    np.fill_diagonal(matrix_one, 0)
    T_c = (matrix_one / (len(graph_list[0][0].nodes) * (len(graph_list[0][0].nodes) - 1))).tolist()
    graph_list = [(graph, _id, T_c) for graph,_id in graph_list]

    nt.NetworkSimulator.parralel_ILP_connections(graph_list, db="Topology_Data",collection="topology-paper", max_time=48*3600, workers=len(graph_list),
                                 threads=1, fibre_num=1, hostname=hostname, port=port,
                                 insert=False, bandwidth=16e9, throughput=False, blocking_rate=0,
                                 k=20)
```

### 1.3 功能详解

#### 1.3.1 数据查询

- **数据库**: `Topology_Data`
- **集合**: `topology-paper`
- **查询条件**:
  - `nodes = 14`: 只选择 14 个节点的拓扑图
  - `ILP Capacity` 存在: 确保已经计算过 ILP 容量
  - `ILP-connections` 不存在: 只处理尚未计算 ILP 连接数的图
- **最大数量**: 10000 个图

#### 1.3.2 流量矩阵生成

```python
matrix_one = np.ones((n, n))  # 创建全 1 矩阵
np.fill_diagonal(matrix_one, 0)  # 对角线置 0（节点不与自己通信）
T_c = matrix_one / (n * (n - 1))  # 归一化：每个节点对的流量需求
```

**流量矩阵 `T_c` 的含义**:
- 所有节点对之间的流量需求均匀分布
- 每个节点对的流量需求 = `1 / (n * (n - 1))`
- 总流量需求 = 1（归一化）

#### 1.3.3 ILP 优化参数

- **`max_time`**: 48 小时（172800 秒）- 每个图的最大求解时间
- **`workers`**: 等于图的数量 - 每个图分配一个 worker 并行处理
- **`threads`**: 1 - 每个 ILP 求解器使用的线程数
- **`fibre_num`**: 1 - 单光纤网络
- **`bandwidth`**: 16 GHz - 信道带宽
- **`throughput`**: False - 不计算吞吐量（只计算连接数）
- **`blocking_rate`**: 0 - 阻塞率（允许阻塞的连接比例）
- **`k`**: 20 - k 最短路径数量

### 1.4 核心函数：`parralel_ILP_connections`

该函数使用 Ray 进行分布式并行计算：

1. **初始化 Ray 集群**: 连接到远程 Ray 集群（`hostname:port`）
2. **任务分配**: 将图列表分配给多个 worker
3. **并行执行**: 每个 worker 调用 `ILP_connections` 函数
4. **结果收集**: 收集所有 worker 的结果

### 1.5 ILP 优化目标

`ILP_connections` 函数调用 `maximise_connection_demand`，优化目标为：

**最大化满足的连接需求数量**

- 给定流量矩阵 `T_c`，每个节点对 `(s, d)` 有流量需求
- ILP 求解器找到最优的路由和波长分配，使得满足的连接数最大化
- 考虑约束条件：
  - 路径约束：只能使用 k 最短路径
  - 波长冲突：同一光纤上的同一波长不能同时用于重叠的路径
  - 容量约束：每个波长有固定的带宽容量

### 1.6 输出结果

结果保存到 MongoDB 数据库，包含：
- `ILP-connections`: 满足的连接数（目标函数值）
- `ILP-connections RWA`: 路由和波长分配结果
- `ILP-connections time`: 计算时间
- `ILP-connections status`: 求解状态（最优解、可行解、不可行等）
- `ILP-connections gap`: 最优性间隙（如果未找到最优解）

---

## 2. cycle_graph_ilp.py

### 2.1 文件概述

**位置**: `thesis/throughput_upperbound/simulations/cycle_graph_ilp.py`

**功能**: 专门用于处理循环图（cycle graphs）的 ILP 优化计算。循环图是一种特殊的网络拓扑结构，常用于理论分析和基准测试。

### 2.2 源代码分析

```python
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
```

### 2.3 功能详解

#### 2.3.1 数据查询

- **数据库**: `Topology_Data`
- **集合**: `cycle_graphs` - 专门存储循环图的集合
- **查询条件**:
  - `nodes <= 40`: 只选择节点数不超过 40 的循环图
- **并行读取**: `parralel=False` - 不使用并行读取

#### 2.3.2 循环图（Cycle Graph）特性

循环图是一种特殊的图结构：
- **定义**: 所有节点排列成一个环，每个节点只与两个相邻节点相连
- **边数**: 等于节点数（`E = N`）
- **度数**: 每个节点的度数为 2
- **应用**: 常用于理论分析、算法测试和性能基准

#### 2.3.3 ILP 优化参数

- **`k`**: 5 - k 最短路径数量（比 `ilp_connections.py` 更小）
- **`threads`**: 1 - 单线程求解
- **`throughput`**: False - 不计算吞吐量
- **`workers`**: 等于图的数量 - 并行处理
- **`local`**: True - 使用本地 Ray 集群（而不是远程）
- **`bandwidth`**: 12.5 GHz - 信道带宽（比 `ilp_connections.py` 更小）

### 2.4 与 `ilp_connections.py` 的区别

| 特性 | ilp_connections.py | cycle_graph_ilp.py |
|------|-------------------|-------------------|
| **目标拓扑** | 通用拓扑（14 节点） | 循环图（≤40 节点） |
| **流量矩阵** | 均匀分布（全连接） | 从数据库读取 `T_c` |
| **k 值** | 20 | 5 |
| **带宽** | 16 GHz | 12.5 GHz |
| **最大时间** | 48 小时 | 默认（1000 秒） |
| **Ray 模式** | 远程集群 | 本地集群 |
| **用途** | 大规模拓扑优化 | 循环图基准测试 |

### 2.5 注释掉的代码

文件中有一段被注释掉的代码，显示了另一种调用方式：

```python
# nt.NetworkSimulator.maximum_throughput_routing(collection="cycle_graph",
#                     hostname=hostname, port=port, desc="topology upgrade", fibres=1, bandwidth=50e9, 
#                     throughput=False, max_time=3600*12, ILP_multi_fibre=False, k=5, insert=False, local=True,
#                     graph_list=graph_list, ILP_threads=5, InS=True)
```

这段代码使用了 `maximum_throughput_routing` 函数，可能是用于最大化吞吐量的优化（而不是最大化连接数）。

---

## 3. 核心功能对比

### 3.1 共同点

1. **都使用 ILP 优化**: 两个脚本都使用整数线性规划求解 RWA 问题
2. **都使用并行计算**: 都通过 Ray 进行分布式并行处理
3. **都保存到数据库**: 结果都保存到 MongoDB 数据库
4. **都不计算吞吐量**: 都设置 `throughput=False`，只优化连接数

### 3.2 主要差异

| 维度 | ilp_connections.py | cycle_graph_ilp.py |
|------|-------------------|-------------------|
| **应用场景** | 通用网络拓扑优化 | 循环图理论分析 |
| **拓扑规模** | 固定 14 节点 | 可变（≤40 节点） |
| **流量模式** | 均匀全连接 | 从数据库读取 |
| **优化参数** | 更宽松（k=20, 48小时） | 更严格（k=5, 默认时间） |
| **计算资源** | 远程集群 | 本地集群 |

---

## 4. 技术实现细节

### 4.1 ILP 求解器

两个脚本都使用 **Gurobi** 作为 ILP 求解器：

- **许可证**: 从环境变量 `GRB_LICENSE_FILE` 读取
- **线程数**: 可配置（两个脚本都使用 1 线程）
- **最大时间**: 可配置（`max_time` 参数）
- **节点文件**: 用于存储求解过程中的节点信息（防止内存溢出）

### 4.2 并行计算架构

使用 **Ray** 分布式计算框架：

```python
# 初始化 Ray 集群
ray.init(address='{}:{}'.format(hostname, port), 
         _redis_password='5241590000000000', 
         dashboard_port=8265)

# 创建任务列表
tasks = [ILP_connections.remote(...) for ind in range(workers)]

# 并行执行并收集结果
results = ray.get(tasks)
```

**优势**:
- 分布式计算：可以在多台机器上并行处理
- 容错性：单个任务失败不影响其他任务
- 可扩展性：可以动态添加 worker

### 4.3 数据库操作

使用 **MongoDB** 存储结果：

- **读取**: `read_topology_dataset_list()` - 批量读取拓扑图
- **更新**: `update_data_with_id()` - 更新现有文档
- **插入**: `insert_graph()` - 插入新文档（如果 `insert=True`）

### 4.4 核心优化函数

#### 4.4.1 `maximise_connection_demand`

**目标函数**: 最大化满足的连接需求数量

**决策变量**:
- `delta[sd][k][w]`: 二进制变量，表示节点对 `sd` 是否使用路径 `k` 和波长 `w`

**约束条件**:
1. **路径约束**: 每个连接只能使用一条路径和一个波长
2. **波长冲突**: 重叠路径不能使用同一波长
3. **需求约束**: 满足的连接数不能超过需求

#### 4.4.2 `maximise_connection_demand_multi_fibre`

多光纤版本的优化函数，额外考虑：
- **光纤选择**: 每条路径可以选择不同的光纤
- **光纤容量**: 每条光纤有独立的波长集合
- **交叉连接能力（CCC）**: 是否允许在节点处进行光纤间转换

---

## 5. 使用场景

### 5.1 ilp_connections.py 的使用场景

1. **大规模拓扑优化**:
   - 处理大量网络拓扑（最多 10000 个）
   - 计算每个拓扑的最优连接数
   - 用于性能评估和基准测试

2. **研究应用**:
   - 评估不同拓扑结构的性能
   - 分析网络容量和连接数的关系
   - 为论文提供实验数据

3. **生产环境**:
   - 为实际网络设计提供参考
   - 评估网络升级方案
   - 优化网络资源配置

### 5.2 cycle_graph_ilp.py 的使用场景

1. **理论分析**:
   - 循环图是理论研究的经典拓扑
   - 用于验证算法正确性
   - 分析算法复杂度

2. **基准测试**:
   - 作为算法性能的基准
   - 比较不同算法的效果
   - 评估优化算法的改进

3. **教学演示**:
   - 用于教学和演示
   - 展示 ILP 优化过程
   - 解释 RWA 问题的求解方法

---

## 6. 代码执行流程

### 6.1 ilp_connections.py 执行流程

```
1. 连接 MongoDB 数据库
   ↓
2. 查询符合条件的拓扑图（14 节点，已有 ILP Capacity，无 ILP-connections）
   ↓
3. 为每个图生成均匀流量矩阵 T_c
   ↓
4. 初始化 Ray 集群（远程）
   ↓
5. 将图列表分配给 workers
   ↓
6. 并行执行 ILP_connections（每个图一个 worker）
   ↓
7. 每个 worker 执行：
   - 创建 OpticalNetwork 对象
   - 调用 maximise_connection_demand
   - 使用 Gurobi 求解 ILP
   - 保存结果到数据库
   ↓
8. 收集所有结果
   ↓
9. 完成
```

### 6.2 cycle_graph_ilp.py 执行流程

```
1. 连接 MongoDB 数据库
   ↓
2. 查询循环图（节点数 ≤ 40）
   ↓
3. 从数据库读取每个图的流量矩阵 T_c
   ↓
4. 初始化 Ray 集群（本地）
   ↓
5. 将图列表分配给 workers
   ↓
6. 并行执行 ILP_connections（每个图一个 worker）
   ↓
7. 每个 worker 执行：
   - 创建 OpticalNetwork 对象
   - 调用 maximise_connection_demand
   - 使用 Gurobi 求解 ILP（k=5）
   - 保存结果到数据库
   ↓
8. 收集所有结果
   ↓
9. 完成
```

---

## 7. 参数说明

### 7.1 关键参数

| 参数 | 说明 | ilp_connections.py | cycle_graph_ilp.py |
|------|------|-------------------|-------------------|
| `max_time` | 最大求解时间（秒） | 172800 (48小时) | 1000 (默认) |
| `k` | k 最短路径数量 | 20 | 5 |
| `bandwidth` | 信道带宽（Hz） | 16e9 | 12.5e9 |
| `threads` | 求解器线程数 | 1 | 1 |
| `fibre_num` | 光纤数量 | 1 | 1 |
| `throughput` | 是否计算吞吐量 | False | False |
| `blocking_rate` | 阻塞率 | 0 | 0 |
| `workers` | 并行 worker 数量 | len(graph_list) | len(graph_list) |
| `local` | 是否使用本地 Ray | False | True |

### 7.2 数据库字段

**ilp_connections.py 输出字段**:
- `ILP-connections`: 满足的连接数
- `ILP-connections RWA`: 路由和波长分配
- `ILP-connections time`: 计算时间
- `ILP-connections status`: 求解状态
- `ILP-connections gap`: 最优性间隙
- `ILP-connections e`: 路径长度参数
- `ILP-connections k`: k 值
- `ILP-connections timestamp`: 时间戳

**cycle_graph_ilp.py 输出字段**:
- 与 `ilp_connections.py` 相同（使用相同的函数）

---

## 8. 注意事项

### 8.1 性能考虑

1. **计算时间**: ILP 求解是 NP-hard 问题，对于大规模网络可能需要很长时间
2. **内存使用**: 大规模网络的 ILP 模型可能占用大量内存
3. **并行效率**: 需要确保 Ray 集群有足够的资源

### 8.2 数据一致性

1. **查询条件**: 确保查询条件正确，避免重复计算
2. **结果验证**: 检查求解状态，确保得到有效解
3. **数据库连接**: 确保数据库连接稳定

### 8.3 错误处理

1. **求解失败**: 如果 ILP 求解失败，检查状态字段
2. **超时处理**: 如果超过 `max_time`，求解器会返回当前最佳解
3. **网络错误**: Ray 集群连接失败时需要重试

---

## 9. 总结

`ilp_connections.py` 和 `cycle_graph_ilp.py` 是两个用于光学网络 RWA 优化的脚本，它们都使用整数线性规划来最大化满足的连接需求。主要区别在于：

- **ilp_connections.py**: 面向通用拓扑的大规模优化，使用更宽松的参数和远程计算集群
- **cycle_graph_ilp.py**: 面向循环图的专门优化，使用更严格的参数和本地计算集群

两个脚本都通过 Ray 实现并行计算，大大提高了处理效率，适用于大规模网络拓扑的优化计算。

---

## 10. 相关文件

- `NetworkToolkit/NetworkSimulator.py`: 包含 `parralel_ILP_connections` 和 `ILP_connections` 函数
- `NetworkToolkit/Routing/ILP.py`: 包含 `maximise_connection_demand` 等核心优化函数
- `NetworkToolkit/Database.py`: 包含数据库操作函数

---

**文档生成时间**: 2024年
**最后更新**: 2024年

