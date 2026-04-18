# `cfm_jax.py` 计算逻辑梳理

这个文档对应文件 [cfm_jax.py](/Volumes/uceebd1/pycharm_projects/network_simulations/thesis/throughput_upperbound/simulations/cfm_jax.py)，目的是把它的整体思路、调用链、核心公式、数据流和性能特点理清楚。

它不是一个"完全重新发明"的模型，而是把原来的 [compute_throughput_cfm.py](/Volumes/uceebd1/pycharm_projects/network_simulations/thesis/throughput_upperbound/simulations/compute_throughput_cfm.py) 做了一次"纯数组化"的重构：  
原版热路径里会频繁访问 `setup` 对象的懒属性，JAX 很难把这部分安全地编译进计算图；`cfm_jax.py` 的做法是先把热路径需要的量一次性 materialize 成数组，再把单链路 NSR 的主要运算写成更 JAX-friendly 的 kernel。

## 1. 这个脚本在干什么

整体流程可以概括成 5 步：

1. 从 MongoDB 读 topology 和 band-aware RWA。
2. 根据 band selection 构造多波段信道网格。
3. 把 RWA 转成"每条链路 x 每个波长"的 occupancy matrix。
4. 对每条链路计算每个 active wavelength 的 NSR。
5. 沿着每条 lightpath 把链路 NSR 累加，再用 Shannon 公式算吞吐。

对应代码入口：

- 数据结构定义：`PreparedCFMInputs`
- 预处理：`_prepare_cfm_inputs()`
- 单链路 NSR：`calc_nsr_link_purejax()`
- topology 级入口：`compute_topology_throughput()`
- 批量跑多个 topology：`run_parallel()` / `run_sequential()`

## 2. 模块依赖和外部函数

`cfm_jax.py` 自己并没有把整个 CFM 物理层从零重写，它仍然调用了原始模块 [compute_throughput_cfm.py](/Volumes/uceebd1/pycharm_projects/network_simulations/thesis/throughput_upperbound/simulations/compute_throughput_cfm.py) 里的很多辅助函数。

### 2.1 依赖的 `cfm` 函数

这里 `import compute_throughput_cfm as cfm`，主要复用了：

- `cfm.BAND_CONFIGS`
  用于解释 band selection，例如 `C / CL / SCL / ESCL / OESCL`。

- `cfm._build_multiband_channel_grid(active_bands)`
  构造整个频谱信道网格，返回：
  - `ch_lambda`
  - `channel_idx`
  - `ch_idx_oband_full`
  - `band_masks`
  - `ref_lambda`

- `cfm._build_per_channel_params(...)`
  根据各 band 的配置生成：
  - 每个 channel 的 launch power
  - noise figure
  - transceiver SNR

- `cfm._build_setup(...)`
  构造 ONG/EGN 模型用的 `setup` 对象。

- `cfm._build_rwa_occupancy(graph, rwa, num_active)`
  把数据库里的 RWA 解析成：
  - `edges`
  - `edge_to_row`
  - `span_count_per_edge`
  - `occupancy_matrix`
  - `rwa_edge_paths`
  - `wavelength_to_col`

- `cfm.raman_solver.solve_isrs_evolution(...)`
  解 Raman/ISRS 的功率沿程演化。

- `cfm.get_power_profile_fit(...)`
  把功率沿程演化拟合成后面 SPM/XPM/FWM 所需的闭式参数。

- `cfm.egn.calc_Pase(...)`
  计算 ASE 噪声功率。

- `cfm.idB(...)`
  把 dB 形式的 transceiver SNR 转成线性尺度。

- `cfm._INDICES`、`cfm._INDICES_COI`、`cfm._INDICES_FWM`
  是原模型里用来枚举某些级数项的索引集合，`cfm_jax.py` 直接沿用。

### 2.2 数据库和图相关函数

- `nt.Database.read_data(...)`
- `nt.Database.read_topology_dataset_list(...)`
- `nt.Tools.read_database_dict(...)`
- `nt.Database.update_data_with_id(...)`

这些负责从 MongoDB 读取 topology / RWA，并把最终吞吐写回数据库。

### 2.3 仍然不是"100% 纯 JAX"的地方

`cfm_jax.py` 的"纯 JAX"主要是指：

- 热路径不再直接访问 `setup.spans[j].beta2` 这种懒属性
- 关键单链路公式改成了数组输入

但它仍然调用了：

- `cfm.raman_solver.solve_isrs_evolution(...)`
- `cfm.get_power_profile_fit(...)`

所以它更准确地说是：

**"纯数组输入的 JAX-friendly NSR kernel + 继续复用现有 Raman/fit 物理层子程序"**

## 3. `PreparedCFMInputs`：为什么要有这个数据结构

在 [cfm_jax.py](/Volumes/uceebd1/pycharm_projects/network_simulations/thesis/throughput_upperbound/simulations/cfm_jax.py) 里，`PreparedCFMInputs` 是最核心的重构点。

原版 `compute_throughput_cfm.py` 的问题是：

- 单链路计算时反复从 `setup` 里取值
- `setup` 里很多属性是懒计算的
- 这些懒计算内部还会混用 NumPy / Python object 逻辑
- JAX 很难稳定追踪并编译这一层

所以 `PreparedCFMInputs` 的作用是：

- 把后面单链路 NSR kernel 所需的量一次性展开成普通数组
- 让 `calc_nsr_link_purejax()` 只接收：
  - 标量
  - NumPy/JAX 数组
  - 不再直接碰复杂对象

### 3.1 里面装了什么

可以把字段分成几类：

#### A. 信道网格相关

- `channel_idx`
- `active_slot_ix`
- `ch_lambda_active`
- `ch_centre_active`
- `ch_centre_ij_active`
- `ch_bandwidth_active`
- `ch_bandwidth_ij_active`

这些描述"哪些 channel 是 active 的，它们的中心频率/波长/带宽是多少"。

#### B. 功率与器件参数

- `ch_power_W_active`
- `ch_power_W_ij_active`
- `nf_active`
- `snr_trx_active`

这些决定：

- 发射功率
- ASE 计算的 NF
- 最终 NSR 里的 transceiver penalty

#### C. 光纤参数

- `attenuation_active`
- `attenuation_ij_active`
- `gamma_active`
- `aeff_active`
- `beta2`
- `beta3`
- `beta4`
- `length_j`
- `length_scalar`
- `raman_profile`
- `raman_gain_slope_j`
- `ref_lambda`
- `zspan`

这些是单 span 物理层计算真正依赖的参数。

#### D. FWM 预计算结构

- `ch_idx_oband_active`
- `fwm_idx_pad`
- `fwm_valid`
- `fwm_idx_pad_oband`
- `fwm_valid_oband`

它们是为了避免在每条链路上重复构建 FWM 索引组合。

## 4. `_prepare_cfm_inputs()` 在做什么

这个函数就是把上面的 `PreparedCFMInputs` 填好。

### 4.1 它提取了第一条 span 的静态参数

代码里有：

```python
span = setup.spans[0]
```

随后提取：

- `beta2 / beta3 / beta4`
- `gamma_active`
- `aeff_active`

这意味着当前实现默认：

- 每条 span 使用同一套 span 类型参数
- 后面单链路 kernel 是按"统一 span 模型"写的

这和原 `calc_NSR_link()` 里固定 `j = 0` 的做法是一致的。

### 4.2 它把 O-band FWM 索引提前建好

这一步是相当重要的性能优化。

原版 [compute_throughput_cfm.py](/Volumes/uceebd1/pycharm_projects/network_simulations/thesis/throughput_upperbound/simulations/compute_throughput_cfm.py) 里，`_FWM_idx(f)` 是写在 `calc_NSR_link()` 内部的，所以每条链路都可能重新建一遍。

而 `cfm_jax.py` 在 `_prepare_cfm_inputs()` 里先做：

- 全 active channel 的 FWM 索引
- 如果 O-band 存在，再单独做一套 O-band 子集索引

这意味着：

- FWM index builder 从"每条链路重复执行"
- 变成了"每次 topology / band setup 只做一次"

这不是全部优化，但已经去掉了一个非常明显的重复开销。

## 5. `_build_fwm_idx()`：它到底在构造什么

这个函数是在频率网格上寻找满足 FWM mixing 条件的四元组。

对于某个受害频率 `f_i`，它寻找：

`f_m = f_j + f_k - f_i`

并要求这个 `f_m` 也落在已有 channel 网格上。

最终每个有效组合会形成一个索引四元组：

- `(i, j, k, m)`

这里它还做了一些去重和合法性约束，例如：

- `j_idx != i`
- `k_idx != i`
- `j_idx != m_idx`
- `k_idx != m_idx`
- `j_idx <= k_idx`

作用是：

- 去掉无意义或重复组合
- 让后面的 FWM kernel 可以直接根据索引访问 `f / P / B / alpha` 等数组

最后它把不同 `i` 对应的组合数量 pad 到相同长度，得到：

- `idx_pad`
- `valid_mask`

这样后面 JAX kernel 就能在规则张量上运行。

## 6. `calc_nsr_link_purejax()`：单链路 NSR 的主计算

这是整个文件最核心的函数。

输入：

- `prep`
  也就是预展开的静态数组集合
- `nspans`
  这条链路跨越多少个 span
- `mask`
  这条链路上哪些 active channels 被占用

输出：

- 这条链路上每个 active channel 的 `NSR`

### 6.1 第一步：按 occupancy 生成本链路的发射功率向量

```python
ch_power_w_i = prep.ch_power_W_ij_active * mask
```

这一步非常关键，因为它说明：

- 不同链路的差别主要由 `mask` 决定
- 也就是说，不同链路上 Raman / fit 之所以不能完全共用，是因为链路占用了不同的 wavelength 子集

### 6.2 第二步：求 Raman / ISRS 功率沿程演化

```python
_, power_evo = cfm.raman_solver.solve_isrs_evolution(...)
```

这一步算的是：

- 每个 active wavelength 沿着 span 长度 `zspan` 的功率演化

它考虑了：

- attenuation
- A_eff
- Raman profile
- channel powers

这是整个单链路计算里很重的一块。

### 6.3 第三步：把功率沿程演化拟合成闭式参数

```python
fit_params = cfm.get_power_profile_fit(...)
```

得到：

- `a`
- `a_bar`
- `cr`

这些量后面会进入 SPM/XPM/FWM 的闭式表达式。

直观上可以理解为：

- 用较少参数近似功率沿程行为
- 避免后面的非线性噪声计算直接在整个 `z` 方向做数值积分

### 6.4 第四步：构造频率和相位失配量

函数中构造了：

- `phi_i`
- `phi_ik`

大致对应：

- SPM 的相位失配项
- XPM/FWM 的频率差导致的相位失配项

它们依赖：

- `beta2`
- `beta3`
- `beta4`
- 各 channel 中心频率 `f_i, f_k`

### 6.5 第五步：构造 ISRS 修正量

代码里有：

- `tf_i`
- `tf_k`
- `t_i`
- `t_k`

它们来自：

- `ptot`
- `cr`
- `a_bar`
- `f`

这部分是在把 Raman/ISRS 拟合结果转化为后续 NLI 公式需要的修正项。

## 7. SPM / XPM / FWM 三块分别在算什么

最后的 NLI 分成 3 块：

- `eta_spm`
- `eta_xpm`
- `eta_fwm`

### 7.1 `eta_gn_spm()`

这是 self-phase modulation 的等效系数。

它对 `cfm._INDICES` 做 `vmap`，本质上是在累加某个级数展开中的项。里面反复出现：

- `alpha`
- `alpha_tilde`
- `kappa`
- `t_tilde`

这些量都来自：

- 拟合得到的 `a / a_bar`
- Raman 修正量 `tf / t`
- span 长度 `L`

公式结构上可以概括为：

- 一个和 `gamma^2` 成正比的非线性项
- 乘上与损耗/有效损耗有关的修正
- 再乘上与相位失配 `phi_i` 有关的反双曲函数项

### 7.2 `eta_gn_xpm()`

这是 cross-phase modulation 的等效系数。

和 SPM 相比，它会涉及：

- 受害信道 `i`
- 干扰信道 `k`
- 功率比 `(P_k / P_i)^2`
- channel bandwidth `B_i / B_k`
- 相位失配 `phi_ik`

同样用到了：

- `alpha_tilde`
- `kappa`
- `t_tilde`

最终得到每个 channel 受其他 channel 影响的 XPM 等效系数。

### 7.3 `eta_gn_fwm()`

这是 O-band 下最重的一块。

函数会对每个 channel `i`：

1. 取出所有预先构造好的 `(i, j, k, m)` FWM 组合
2. 对每个组合计算：
   - `phi_jk`
   - `dphi_f1`
   - `dphi_f2`
3. 再进入 island 积分近似

这个实现里主要保留了 `m_v == i_v` 的 COI 型分支：

```python
island_sum = jax.lax.cond(
    m_v == i_v,
    lambda _: jax.vmap(_island_coi)(indices_coi).sum(axis=0),
    lambda _: 0.0,
    operand=None,
)
```

这点很重要，因为它意味着：

- `cfm_jax.py` 里的 FWM 实现是一个"收敛到主干结构"的版本
- 它没有完整照搬原版里更复杂的 FWM 分支

所以它的目标更像是：

- 先把主要结构 JAX 化
- 而不是先做到每个细枝末节都 1:1 重现原实现

## 8. ASE 和最终 NSR 怎么合成

在 NLI 三项算完之后，代码继续做：

```python
ratio_in_out = power_evo[0, :] / power_evo[-1, :]
gain_lin = max(ratio_in_out, 1 + eps)
p_ase = cfm.egn.calc_Pase(...) * nspans
```

意思是：

- 用功率演化的输入/输出比得到每个 channel 的线性 gain
- 再根据 NF 和带宽去算 ASE

最后总 NSR 组成为：

`NSR = NLI + ASE + trx_penalty`

在代码里对应：

- `eta_sum = eta_spm + eta_xpm + eta_fwm`
- `term_nli = nspans * P^2 * eta_sum`
- `term_ase = p_ase / P`
- `trx_pen = 1 / idB(snr_trx)`

然后：

```python
nsr_active = term_nli + term_ase + trx_pen
```

如果某个 wavelength 在这条链路上没被占用，则：

```python
return inf
```

这样后面 lightpath 汇总时，空闲 channel 不会被误算成可用。

## 9. `_compute_topology_arrays()`：把 topology 级静态部分搭好

这个函数把 topology 级预处理封装成一步：

1. 根据 `band_selection` 取 band 配置
2. 调 `_build_multiband_channel_grid()`
3. 调 `_build_per_channel_params()`
4. 调 `_build_setup()`
5. 调 `_build_rwa_occupancy()`
6. 调 `_prepare_cfm_inputs()`

它的输出其实就是后面 topology 级计算所需的全部静态上下文：

- `setup`
- `prep`
- 边列表和行号映射
- span 数
- occupancy matrix
- RWA edge paths
- wavelength 到矩阵列号的映射

这里的设计思想是：

**band/grid/setup/RWA 解析只做一次，单链路 kernel 只做真正必须重复的那部分。**

## 10. `compute_topology_throughput()`：整张图怎么算吞吐

这是 topology 级主入口。

### 10.1 读取输入

它先从 MongoDB 读：

- topology 文档
- band-aware RWA key，例如：
  - `"kSP-FF C RWA"`
  - `"kSP-FF OESCL RWA"`

然后把 graph relabel 成整数节点，并把数据库中的字典格式 RWA 转回内存对象。

### 10.2 逐条链路计算 NSR

接着调用 `_compute_topology_arrays()`，拿到：

- `span_count_per_edge`
- `occupancy_matrix`

然后循环每条 undirected link：

```python
for i in range(n_links):
    link_nsr = calc_nsr_link_purejax(...)
```

这里 `nsr_link_channel[row, col]` 的含义是：

- 第 `row` 条链路
- 第 `col` 个 active wavelength
- 对应的单链路 NSR

### 10.3 沿 lightpath 累加 NSR

拿到每条链路每个 wavelength 的 NSR 后，再遍历 `rwa_edge_paths`：

- 对某个 lightpath
- 查它用的是哪个 wavelength `w`
- 找到该 wavelength 在 active channels 里的列号 `col`
- 把这条路径经过的每条边的 `nsr_link_channel[row, col]` 加起来

也就是：

`path_nsr = sum(link_nsr over all links on the lightpath)`

### 10.4 用 Shannon 公式算每条 lightpath 容量

若：

`SNR = 1 / path_nsr`

则容量：

`rate = 2 * B * log2(1 + SNR)`

代码里：

```python
rate_bps = 2 * ch_bw_hz * np.log2(1.0 + snr)
```

最后整网吞吐就是所有 lightpath 速率求和。

## 11. `run_parallel()` 和 `run_sequential()`

这两部分只是调度层。

### 11.1 `run_sequential()`

- 从数据库里把同名 topology 全部读出来
- 逐个 `_id` 调 `compute_topology_throughput()`

适合：

- 调试
- 看日志
- 定位某个 topology 慢在哪条链路

### 11.2 `run_parallel()`

- 用 Ray 把不同 `_id` 的 topology 分发给远程 worker

注意这里是：

- topology 之间并行
- 不是单个 topology 内部的链路并行

所以如果单张图很慢，Ray 只能帮助"多张图一起跑"，不能直接解决"一张图内部 OESCL 太慢"的问题。

## 12. 和原 `compute_throughput_cfm.py` 相比，`cfm_jax.py` 改了什么

### 12.1 改善点

最重要的是：

- 原版把很多 `setup` 属性访问放在热路径内
- `cfm_jax.py` 用 `PreparedCFMInputs` 把这些量先展开

这对 JAX 很重要，因为它避免了：

- 运行时反复触发 Python 对象访问
- 懒属性里再去调用 NumPy
- tracer 和 Python object 混在一起

此外：

- FWM index 预计算也提前到了 `_prepare_cfm_inputs()`

### 12.2 仍然没有彻底解决的慢点

尽管文件名叫 `cfm_jax.py`，但重的部分并没有全部消失。

每条链路仍然要重新做：

- `solve_isrs_evolution()`
- `get_power_profile_fit()`

而且在 O-band 存在时，还要进 FWM。

所以 OESCL 慢的主要原因依旧会保留：

- active channels 多
- Raman/fit 重
- FWM 重

## 13. 为什么 OESCL 特别慢

这份文件里，OESCL 慢主要是三件事叠加：

1. OESCL 的 active channels 最多。
2. 每条链路都要重新做 Raman 演化和 fit。
3. O-band 打开后 FWM 分支会被启用。

尤其第三点最容易让运行时间突然上一个量级。

## 14. 从"可复用量"的角度怎么理解这个文件

可以把整个计算拆成三层：

### 第一层：完全静态，可一次性预计算

这些与 topology 的 band/grid/setup 有关，但与具体链路 occupancy 无关：

- `beta2 / beta3 / beta4`
- `gamma_active`
- `aeff_active`
- `channel frequencies`
- `channel bandwidths`
- `NF`
- `snr_trx`
- FWM index pad / valid

这一层已经主要被 `PreparedCFMInputs` 吃掉了。

### 第二层：和 occupancy 相关，但与 `nspans` 无关

这些量是单链路最值得缓存的：

- `power_evo`
- `fit_params -> a, a_bar, cr`
- `gain_lin`
- `one-span noise`

如果两条链路的 occupancy mask 一样，它们这层结果其实可以复用。

### 第三层：和 `nspans` 线性相关

在当前公式里：

- NLI 项乘 `nspans`
- ASE 项乘 `nspans`
- transceiver penalty 不乘

所以如果是 scaling 问题，最自然的优化是：

- 先算一次 one-span noise
- 后面按 `nspans` 线性放大

这正是 `cfm_scaling_jax.py` 想利用的结构。

## 15. 这份文件最适合怎么读

如果你要真正吃透这份代码，我建议按下面顺序看：

1. 先看 `compute_topology_throughput()`
   这是整张图怎么跑完的总入口。

2. 再看 `_compute_topology_arrays()`
   这是 topology 静态数据如何搭起来的。

3. 再看 `_prepare_cfm_inputs()`
   这是 JAX 重构的关键。

4. 然后集中看 `calc_nsr_link_purejax()`
   这里才是单链路物理层主公式。

5. 最后看 `_build_fwm_idx()`
   这是 O-band/FWM 复杂度来源之一。

## 16. 一句话总结

`cfm_jax.py` 的核心思想不是"把所有物理层都彻底 GPU 化"，而是：

**先把 topology/band/setup/RWA 的静态部分和懒属性访问从热路径里剥离出来，再用数组化的 JAX kernel 去计算单链路 NSR，最后沿 lightpath 汇总吞吐。**

如果继续优化，最值得下手的方向通常不是再加一层 `jit`，而是：

- 对相同 occupancy 的链路复用 Raman / fit 结果
- 对 scaling 场景复用 one-span noise
- 把 FWM 从"全 O-band 组合"改成"实际 occupied O-band 组合"

## 17. OESCL NSFNET 运行时间分析

### 17.1 实测数据

NSFNET 拓扑有 21 条无向链路，OESCL 配置下有 878 个活跃信道。运行输出：

```
Link 0  ((1, 2),  spans=19): 30.3s   <- 含 JIT 编译
Link 1  ((1, 3),  spans=27): 10.1s
Link 2  ((1, 13), spans=45):  9.7s
Link 3  ((2, 3),  spans=14):  9.5s
...
Link 20 ((11,12), spans=5):   9.0s
```

- 第一条链路 30.3s（含 JAX JIT 首次编译开销约 20s）
- 后续每条链路约 9-11s
- 21 条链路合计约 220s

### 17.2 单条链路的计算步骤和耗时拆分

每条链路的 `calc_nsr_link_purejax()` 内部包含以下步骤：

| 步骤 | 复杂度 | 预估耗时 | 说明 |
|------|--------|---------|------|
| 1. ISRS ODE 求解 | O(N^2 x z_steps) | ~3-4s | 878x878 Raman 增益矩阵 x 80 个 z 步，Kvaerno5 隐式求解器每步还需牛顿迭代 |
| 2. 功率演化曲线拟合 | O(N x LM_iters x z_steps) | ~2-3s | 对 878 个通道 vmap 并行做 Levenberg-Marquardt（每通道约 10-20 次迭代） |
| 3. SPM 计算 | O(N) | ~0.1s | 逐通道向量化，trivial |
| 4. XPM 计算 | O(N^2) | ~2-3s | 878x878 矩阵运算（phi_ik, arctan 等） |
| 5. FWM 计算（O-band） | O(N_O^2 x M) | ~2-3s | N_O 约 344 个 O-band 通道，每通道扫描 FWM 四元组 |
| 6. ASE + NSR 合成 | O(N) | ~0.1s | 简单向量运算 |

其中步骤 1、2、4、5 是主要耗时来源，合计占每条链路 ~9-10s 的绝大部分。

### 17.3 为什么 OESCL 比单 C-band 慢这么多

对比 OESCL (878 channels) vs C-band only (~90 channels)：

| 计算步骤 | C-band (N=90) | OESCL (N=878) | 倍数 |
|----------|---------------|---------------|------|
| ISRS ODE（矩阵乘法 N^2） | 90^2=8,100 | 878^2=770,884 | **~95x** |
| XPM（NxN 矩阵） | 8,100 元素 | 770,884 元素 | **~95x** |
| 功率拟合（N 通道） | 90 通道 | 878 通道 | **~10x** |
| FWM（O-band 独有） | 无 | 344 通道 | **从无到有** |

关键点：

1. **ISRS ODE 是 O(N^2)**：每个 ODE 步需要计算 `g_Ra @ y`（878x878 矩阵乘 878 向量），这是整个热路径里最重的矩阵运算。从 90 通道扩展到 878 通道后，矩阵大小增长了 95 倍。

2. **XPM 也是 O(N^2)**：`phi_ik`、`arctan` 等都是 878x878 的矩阵运算。C-band 下只有 90x90。

3. **FWM 是 O-band 独有的**：C/S/E/L band 不触发 FWM 分支。O-band 有约 344 个通道，FWM 需要枚举满足 `f_m = f_j + f_k - f_i` 的四元组。虽然 `cfm_jax.py` 已经把 FWM 索引预计算到 `PreparedCFMInputs` 里（只做一次），但每条链路仍然要执行 FWM kernel。

4. **float64 精度翻倍**：`cfm_jax.py` 使用 `jax_enable_x64 = True`，所有数组从 32 位变成 64 位，GPU 显存和计算量都翻倍。

### 17.4 O-band 是不是主要原因

是的，O-band 是 OESCL 慢于 ESCL 的主要原因，但不是 OESCL 慢于 C-band 的唯一原因。

- **ESCL（无 O-band，约 534 通道）** 比 C-band 慢的主要原因是通道数增加导致 ISRS ODE 和 XPM 的 O(N^2) 开销增长 (534^2 / 90^2 = 35x)。
- **OESCL（有 O-band，878 通道）** 在 ESCL 基础上进一步慢的原因有两个：
  - 通道数从 534 增加到 878，O(N^2) 项再增长 (878^2 / 534^2 = 2.7x)
  - **新增 FWM 分支**：每条链路对 344 个 O-band 通道执行 FWM kernel，这是一个 O(N_O^2 x M) 的计算，在纯 C/E/S/L 配置下完全不存在

所以 O-band 的影响是双重的：既增加了 N（使所有 O(N^2) 步骤变慢），又触发了 FWM（新增一个 O(N_O^2) 步骤）。

## 18. OESCL NSFNET 部分链路 SNR 为负数的分析

### 18.1 问题现象

运行 `cfm_jax.py` 计算 NSFNET OESCL 时，21 条链路中有多条出现负 SNR：

```
Link 5  ((3, 6),  spans=33): mean SNR=  0.0 dB
Link 7  ((4, 8),  spans=37): mean SNR= -0.5 dB
Link 9  ((5, 7),  spans=14): mean SNR=-38.1 dB  <- 极端
Link 11 ((6, 14), spans=19): mean SNR=  0.4 dB
Link 12 ((7, 13), spans=14): mean SNR= -9.5 dB
Link 15 ((9, 10), spans= 7): mean SNR=-67.4 dB  <- 极端
```

Link 9 和 Link 15 的 SNR 极端（-38 和 -67 dB），即使考虑多 span 累积也不应该这么低。

### 18.2 根本原因：`get_power_profile_fit` 的频率与 kickstart 策略

`cfm_jax.py` 和 `compute_throughput_cfm.py` 都调用同一个 `get_power_profile_fit()` 函数（来自 `ong.models.raman_fitting`）。这个函数内部的拟合策略存在数值脆弱性：

**拟合模型**（Daniel Semrau 半解析解）：

```
rho(z) = (1 + Ti) * exp(-a*z) - Ti * exp(-(a+a_bar)*z)
其中 Ti = P * Cr * f_i / a_bar
```

**kickstart 策略**：函数先选中间通道（`valid_idx[size//2]`）拟合出初始参数，再用 `jax.vmap` 以这个初始参数并行拟合所有通道。

**问题**：`cfm_jax.py` 传入的是正确的活跃通道频率（`ch_centre_ij_active`），kickstart 通道落在 E-band（第 439 个活跃通道），其频率偏移 `f_i` 约 -2.65 THz，接近参考波长 1419.4 nm。由于 `Ti` 正比于 `Cr * f_i`，当 `f_i` 接近 0 时 `Ti` 接近 0，模型退化为简单指数衰减 `exp(-a*z)`，`Cr` 参数完全不受约束。

不受约束的 `Cr` 作为初始值传给所有通道后，频率偏移大的通道（O-band: f 约 +10 到 +27 THz）的初始 `Ti` 可能偏差极大，导致 Levenberg-Marquardt 收敛到错误的局部极小值，产出垃圾 `a, a_bar, Cr` 值。

### 18.3 为什么只有部分链路受影响

并非所有链路都出现负 SNR。受影响的链路通常有以下特征：

1. **特定的 occupancy 模式**：某些链路上只有少量波长被占用（`mask` 中大部分为 0），导致 ISRS ODE 求解时功率向量稀疏，power evolution 曲线更不规则，拟合更容易失败。

2. **占用波长分布不均**：如果被占用的波长集中在频谱一端（如只有 O-band 或只有 L-band），Raman 功率转移效应变得不对称，三参数模型更难准确拟合。

3. **拟合残差在 GN 模型中被放大**：GN 模型中的 `alpha_tilde`、`kappa`、`T_tilde` 等量包含指数和分式运算，小的参数误差可以导致 NLI 估计偏差几个数量级。

### 18.4 Link 15 为什么特别极端（-67.4 dB）

Link 15 `(9, 10)` 只有 7 个 span（最短链路之一），SNR 却最差（-67.4 dB）。这反直觉（短链路应该更好），说明问题不是物理性的，而是数值性的。

可能原因是这条链路上的 occupancy 模式恰好触发了拟合的最坏情况：被占用的波长位于频率偏移大的区域，kickstart 参数对这些波长的初始猜测极差，Levenberg-Marquardt 收敛到错误的局部极小值，使得 NLI 项变成极大正数，淹没信号。

### 18.5 `cfm_jax.py` 与 `compute_throughput_cfm.py` 的差异

两个文件调用 `get_power_profile_fit` 时传入的 `ch_centre_ij` 不同：

| | `compute_throughput_cfm.py` | `cfm_jax.py` |
|---|---|---|
| 传入的 `ch_centre_ij` | `setup.ch_centre_ij`（1111 行，全网格） | `ch_centre_ij_active`（878 行，仅活跃通道） |
| 频率是否正确 | 否（波段间隙导致偏移） | 是 |
| kickstart 通道频率 | (555-555) x 50GHz = 0 Hz | 约 -2.65 THz |
| Cr 约束程度 | 不受约束（f=0） | 弱约束（f 小） |
| 拟合稳定性 | 较好（所有通道 \|f\| < 16 THz，近似单指数） | 较差（O-band 通道 \|f\| 达 27 THz，双指数形状复杂） |
| 拟合结果准确性 | Cr 不准但因频率也错，误差部分抵消 | Cr 不准时直接暴露，无法抵消 |

两者都不是完美的解决方案。`compute_throughput_cfm.py` 靠"两个错误互相抵消"得到了看起来合理的结果；`cfm_jax.py` 用了正确的频率但暴露了拟合的脆弱性。

### 18.6 总结

| 问题 | 根因 | 影响范围 |
|------|------|---------|
| 计算慢 | O(N^2) 的 ISRS 和 XPM + O-band FWM，N=878 | 所有 OESCL 计算 |
| 部分链路 SNR 负数 | `get_power_profile_fit` 的 kickstart 策略在正确频率下 Cr 不受约束 | 特定 occupancy 模式的链路 |
| float64 进一步加慢 | `jax_enable_x64 = True` 使所有运算翻倍 | 所有计算 |
