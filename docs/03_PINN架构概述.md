# PINN 架构概述

> 本文说明当前多材料导热 PINN 的网络组织方式，重点解释
> `src/networks.py` 中基础网络 `MLP` 的设计思路，以及它与分域温度场、
> 控制方程损失、界面损失和边界损失之间的关系。

---

## 1. 总体架构

当前代码没有使用一个全局网络直接表示整个计算域的温度，而是采用
“分材料域温度网络 + 显式物理损失耦合”的结构：

```text
物理坐标 (x, y)
      │
      ▼
TemperatureField
      │
      ├── wall_l ── MLP ── θ_wall_l
      ├── wall_r ── MLP ── θ_wall_r
      ├── wall_b ── MLP ── θ_wall_b
      ├── wall_t ── MLP ── θ_wall_t
      ├── air    ── MLP ── θ_air
      ├── dev1   ── MLP ── θ_dev1
      ├── dev2   ── MLP ── θ_dev2
      └── dev3   ── MLP ── θ_dev3
      │
      ▼
PDE、界面、边界和能量预算损失
```

采用分域网络的主要原因是不同材料的导热系数跨度很大：

$$
\frac{k_{Al}}{k_f}=\frac{167}{0.026}\approx6423.
$$

材料界面两侧的温度连续，但温度梯度通常不连续。如果让一个全局光滑网络
同时拟合全部材料域，它需要在界面附近表示明显的导数跳变，训练难度较高。
当前架构让各区域分别学习自己的温度函数，再通过温度连续和热流连续损失
将它们耦合起来。

---

## 2. `MLP` 的职责

`MLP` 是每个材料域温度分支使用的基础网络。它只负责学习一个连续、平滑、
可通过自动微分求二阶导数的函数，不直接处理材料参数、几何边界或控制方程。

当前实现为：

```python
class MLP(nn.Module):
    def __init__(self, n_in, n_out, width, depth, fourier_sigma=0.0,
                 fourier_dim=64):
        super().__init__()
        self.fourier_dim = 0
        if fourier_sigma > 0:
            B = torch.randn(n_in, fourier_dim) * fourier_sigma
            self.register_buffer("fourier_B", B)
            self.fourier_dim = fourier_dim
            n_in = n_in + 2 * fourier_dim

        layers = [nn.Linear(n_in, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers += [nn.Linear(width, n_out)]
        self.net = nn.Sequential(*layers)

        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        if self.fourier_dim:
            proj = 2 * torch.pi * x @ self.fourier_B
            x = torch.cat([x, torch.sin(proj), torch.cos(proj)], dim=-1)
        return self.net(x)
```

默认情况下：

- `n_in=2`，输入二维坐标 $(x,y)$；
- `n_out=1`，输出无量纲温度 $\theta$；
- `width` 控制每个隐藏层的神经元数量；
- `depth` 控制隐藏层数量；
- `fourier_sigma=0`，即默认关闭 Fourier 特征。

---

## 3. 为什么主体采用 `Tanh MLP`

各材料域需要满足 Laplace 或 Poisson 方程：

$$
\nabla^2\theta=0,
$$

或

$$
\nabla^2\theta+S_k=0.
$$

训练时需要通过 PyTorch 自动微分计算：

$$
\frac{\partial^2\theta}{\partial x^2},
\qquad
\frac{\partial^2\theta}{\partial y^2}.
$$

`Tanh` 连续且具有平滑的高阶导数，因此适合包含二阶空间导数的 PINN。
相比之下，ReLU 的二阶导数几乎处处为零，直接用于当前导热 PDE 时不合适。

网络结构可以写成：

```text
(x, y)
   ↓
Linear + Tanh
   ↓
若干个 Linear + Tanh
   ↓
Linear
   ↓
θ(x, y)
```

最后一层不使用激活函数，因为无量纲温度不应被限制在 `[-1,1]`。例如在尚未
加入辐射的静止空气模型中，固定圆形设备附近可能出现远高于参考温差范围的
温度，网络输出需要允许超过 1。

---

## 4. Fourier 特征的设计

### 4.1 引入原因

普通 `Tanh MLP` 通常优先学习低频、缓慢变化的函数，即存在频谱偏置。
当前问题的整体温度场较平滑，但圆形设备附近的空气层可能出现较陡的局部
温度梯度。为提高网络表示局部变化的能力，`MLP` 预留了随机 Fourier 特征。

随机频率矩阵定义为：

$$
B\sim\mathcal N(0,\sigma_f^2),
$$

对应代码：

```python
B = torch.randn(n_in, fourier_dim) * fourier_sigma
```

前向传播时，原始输入被扩展为：

$$
\phi(\mathbf x)=
\left[
\mathbf x,
\sin(2\pi\mathbf xB),
\cos(2\pi\mathbf xB)
\right].
$$

代码实现为：

```python
proj = 2 * torch.pi * x @ self.fourier_B
x = torch.cat([x, torch.sin(proj), torch.cos(proj)], dim=-1)
```

保留原始坐标是为了让网络同时获得：

- 原始坐标所表达的低频和整体温度趋势；
- 正弦、余弦特征所表达的局部和较高频变化。

启用 Fourier 特征后，输入维数从 $n_{in}$ 变为：

$$
n_{in}+2N_F,
$$

其中 $N_F$ 为 `fourier_dim`。

### 4.2 随机矩阵 $B$ 的物理和几何含义

在二维问题中，若 `n_in=2`、`fourier_dim=N_F`，则：

$$
B\in\mathbb R^{2\times N_F}.
$$

$B$ 的每一列都是一个二维空间频率向量：

$$
\mathbf b_j=
\begin{bmatrix}
b_{x,j}\\
b_{y,j}
\end{bmatrix}.
$$

对坐标 $\mathbf x=(x,y)$，第 $j$ 个投影为：

$$
p_j=2\pi\mathbf x\cdot\mathbf b_j
=2\pi(xb_{x,j}+yb_{y,j}).
$$

因此，每一列同时定义了一个空间振荡方向和一个空间尺度。例如：

- $\mathbf b_j=(5,0)$：特征只沿水平方向变化；
- $\mathbf b_j=(0,8)$：特征只沿竖直方向变化；
- $\mathbf b_j=(3,4)$：特征沿斜向变化。

频率向量的长度

$$
\|\mathbf b_j\|=\sqrt{b_{x,j}^2+b_{y,j}^2}
$$

决定振荡速度，其代表性波长近似满足：

$$
\lambda_j\sim\frac{1}{\|\mathbf b_j\|}.
$$

因此，较短的频率向量对应大尺度、缓慢变化的温度结构；较长的频率向量对应
局部热点、薄热层或陡峭温度梯度。

$B$ 的随机性并不表示最终温度场是随机的。它只是为网络随机准备一组候选
空间基函数，后续 MLP 权重仍由 PDE、界面、边界和能量守恒损失确定。可以将
这一过程理解为：

```text
B：随机生成一套具有不同方向和波长的空间基函数字典
MLP：根据物理损失学习如何选择和组合这些基函数
```

采用随机采样而不是人工指定频率，是因为训练前通常不知道热点的准确宽度和
主要变化方向。高斯随机采样可以用较低成本覆盖不同方向和频率尺度。由于各
坐标方向采用相同的零均值高斯分布，它在统计上不偏向某个特定方向。

从随机 Fourier 特征理论看，对足够多的随机频率进行正余弦映射，可以近似某类
平移不变核。高斯采样对应 RBF/高斯类型的空间相关先验：较小的频率尺度偏向
宽而平滑的相关结构，较大的频率尺度偏向短程、局部的相关结构。

### 4.3 `fourier_sigma` 和 `fourier_dim`

`fourier_sigma` 控制随机空间频率的尺度：

- `fourier_sigma=0`：关闭 Fourier 特征，使用普通 MLP；
- 较小的 `fourier_sigma`：偏向缓慢的空间变化；
- 较大的 `fourier_sigma`：能够表达更局部、更陡的变化，但优化难度也可能增加。

`fourier_dim` 控制随机频率数量：

- 数量越多，频率覆盖越丰富；
- 第一层输入维数、显存占用和计算量也相应增加；
- 当前默认值为 64，启用后增加 64 个正弦和 64 个余弦特征。

当前各材料域的默认 `fourier_sigma` 均为零。也就是说，Milestone 1 首先使用
普通 `Tanh MLP`；只有独立验证表明局部温度梯度拟合不足时，才启用 Fourier
特征。它属于数值表达增强手段，不是控制方程的一部分。

第一层参数量近似为：

$$
\left(n_{in}+2\,fourier_{dim}\right)\times width.
$$

因此，`fourier_sigma` 决定频率有多高，`fourier_dim` 决定准备多少组频率：

| 参数 | 控制内容 | 增大后的作用 | 主要代价或风险 |
|---|---|---|---|
| `fourier_sigma` | 频率尺度 | 表达更快、更局部的变化 | 振荡和二阶导数放大 |
| `fourier_dim` | 频率数量 | 覆盖更多方向和尺度 | 参数量、显存和计算量增加 |

### 4.4 对 PINN 导数和训练稳定性的影响

Fourier 特征的一阶导数与频率成正比：

$$
\frac{d}{dx}\sin(2\pi bx)
=2\pi b\cos(2\pi bx),
$$

二阶导数与频率平方成正比：

$$
\frac{d^2}{dx^2}\sin(2\pi bx)
=-(2\pi b)^2\sin(2\pi bx).
$$

因此，过大的 `fourier_sigma` 会显著放大导热 PDE 中的二阶导数，可能造成：

- PDE 残差尺度过大；
- 梯度振荡或训练不稳定；
- 配点之间出现没有物理依据的伪振荡；
- 网络在高频成分上消耗过多表达能力。

另一个重要细节是：当前 $B$ 作用于各材料域归一化后的坐标，而不是原始物理
坐标。相同的 `fourier_sigma` 在 1 cm 铝墙和接近 1 m 的空气域中对应不同的
物理波长。因此，如果后续启用 Fourier 特征，应该按材料域分别选择频率尺度，
不宜默认让所有分支共用同一个数值。

### 4.5 随机种子与可复现性

当 `fourier_dim` 有限时，不同的随机 $B$ 会提供不同的频率集合，因此可能影响
收敛速度、局部温度误差和最终 PDE 残差。当前训练入口固定了 PyTorch 随机
种子，所以相同配置可以生成相同的 $B$。

正式启用 Fourier 特征后，建议至少比较三个随机种子，并同时检查：

- dev3 附近的局部温度误差；
- 空气域相对 L2 误差；
- 设备—空气界面热流误差；
- 总能量守恒；
- 独立配点上的 PDE 残差。

如果结果对随机种子非常敏感，通常意味着 `fourier_dim` 太小、
`fourier_sigma` 与物理尺度不匹配，或者当前问题并不需要 Fourier 增强。

### 4.6 为什么将频率矩阵注册为 buffer

```python
self.register_buffer("fourier_B", B)
```

随机矩阵 $B$ 只用于固定的坐标编码，不应该由优化器更新，因此没有定义为
`nn.Parameter`。注册为 buffer 后，它能够：

- 随模型自动移动到 CPU 或 GPU；
- 保存到 `state_dict` 和 checkpoint；
- 加载模型时恢复相同的随机频率；
- 不出现在待优化参数列表中。

这样可以避免加载 checkpoint 后重新生成 $B$，从而保证相同网络权重对应相同
的温度函数。

### 4.7 方法来源

当前实现最直接参考 Tancik 等人提出的坐标 Fourier 特征方法。该工作用
Fourier 坐标映射缓解低维坐标 MLP 的频谱偏置。原论文通常以正弦、余弦映射
作为输入；当前代码额外保留原始坐标，是兼顾低频趋势的工程化变体。

随机 Fourier 特征的更早理论基础来自 Rahimi 和 Recht，其出发点是用显式随机
特征近似平移不变核。Wang、Wang 和 Perdikaris 随后从 NTK 角度研究了 Fourier
特征在 PINN 和多尺度 PDE 中的作用。

参考文献：

1. Rahimi, A., Recht, B. *Random Features for Large-Scale Kernel Machines*.
   NIPS, 2007.  
   <https://papers.nips.cc/paper/2007/hash/013a006f03dbc5392effeb8f18fda755-Abstract.html>
2. Tancik, M., et al. *Fourier Features Let Networks Learn High Frequency
   Functions in Low Dimensional Domains*. NeurIPS, 2020.  
   <https://proceedings.neurips.cc/paper_files/paper/2020/hash/55053683268957697aa39fba6f231c68-Abstract.html>
3. Wang, S., Wang, H., Perdikaris, P. *On the Eigenvector Bias of Fourier
   Feature Networks: From Regression to Solving Multi-scale PDEs with
   Physics-informed Neural Networks*. CMAME, 2021.  
   <https://arxiv.org/abs/2012.10047>

---

## 5. 参数初始化

每个线性层采用 Xavier 正态初始化，偏置初始化为零：

```python
nn.init.xavier_normal_(m.weight)
nn.init.zeros_(m.bias)
```

`Tanh` 在输入绝对值较大时会进入饱和区：

$$
\tanh(z)\rightarrow\pm1.
$$

进入饱和区后，一阶和二阶导数都会减小，不利于 PDE 残差反向传播。Xavier
初始化根据相邻层的输入、输出维数控制初始权重尺度，使隐藏层在训练初期尽量
工作于 `Tanh` 的非饱和区域，从而改善：

- 前向输出的尺度稳定性；
- 一阶梯度传播；
- 二阶 PDE 导数的有效性。

零偏置让每个基础 MLP 的偏置初值保持对称。若需要指定正的统一温度水平，则由
外层 `TemperatureField` 的 `theta_init` 逻辑重置各分支最后一层。当前代码只有
在 `theta_init > 0` 时才执行这项重置；默认 `theta_init=0` 不代表强制冷端平场，
而是保留完整的 Xavier 随机网络初值。

---

## 6. `TemperatureField`：分域温度场装配器

### 6.1 核心思路

`MLP` 只负责学习一个平滑的标量函数，不知道输入属于哪种材料，也不知道设备
位置、温度归一化方式和物理边界条件。`TemperatureField` 则负责把多个 MLP
组织成完整的多材料温度场。

当前问题满足：

$$
T_a=T_b,
$$

但材料界面的守恒条件为：

$$
k_a\frac{\partial T_a}{\partial n}
=k_b\frac{\partial T_b}{\partial n}.
$$

因为 $k_{Al}/k_f\approx6423$，温度虽然连续，界面两侧的温度梯度一般存在很大
跳变。一个全局光滑网络很难同时表示这种跨材料导数变化，因此当前代码让每个
物理区域使用独立温度分支：

```text
TemperatureField
├── wall_l → MLP_wall_l
├── wall_r → MLP_wall_r
├── wall_b → MLP_wall_b
├── wall_t → MLP_wall_t
├── air    → MLP_air
├── dev1   → MLP_dev1
├── dev2   → MLP_dev2
└── dev3   → MLP_dev3
```

默认采用气凝胶一维热阻，因此共有八个温度分支；若设置
`USE_TBL_1D=False`，则额外增加显式气凝胶分支 `tbl`。

显式 `tbl` 目前只是保留的对比路径，并非与默认模式同等完整：域网络、PDE、
`tbl_wall` 界面、左端 Dirichlet、上下绝热和监控分支都已存在，但
`EnergyConservationLoss` 的全局左边界积分仍固定调用 `wall_l`。因此在修正该处
为读取 `bnd_samples["left"]["dom"]` 之前，`USE_TBL_1D=False` 不应直接用于正式
训练结论；当前 Milestone 1 正式基线是 `USE_TBL_1D=True`。

### 6.2 使用 `ModuleDict` 管理子网络

所有区域网络保存在：

```python
self.nets = nn.ModuleDict(
    {d: MLP(2, 1, width, depth, sig[d], fourier_dim)
     for d in self.DOMAINS}
)
```

调用时显式给出区域名称：

```python
theta_air = field("air", points)
theta_dev1 = field("dev1", points)
theta_wall = field("wall_b", points)
```

使用 `nn.ModuleDict` 而不是普通字典，可以保证：

- 所有子网络参数出现在 `field.parameters()` 中；
- `.to(device)` 会移动全部分支及其 buffer；
- `state_dict()` 和 checkpoint 会保存全部子网络；
- Adam、L-BFGS 可以一次联合更新所有温度分支。

因此当前实现是“一次训练多个子网络”，而不是依次训练八个互不相关的模型。

### 6.3 与 XPINN/cPINN 的关系

这种“一个子域对应一个网络，再用界面损失耦合”的组织方式与 XPINN 很接近。
墙体被拆成 `wall_l/wall_r/wall_b/wall_t`，尤其属于典型的数值区域分解：四个
子域材料相同、PDE 相同，只是为了改善薄框架区域的输入条件而拆分，再通过四个
墙角界面连接。

当前实现又强调物理热流守恒：

$$
k_a\partial_nT_a=k_b\partial_nT_b,
$$

所以也具有 cPINN 的保守耦合特征。更准确的名称是：

> XPINN/cPINN 风格的多材料分域 PINN。

它还不是完整通用 XPINN 框架，因为当前：

- 所有分支使用一个总损失联合反向传播；
- 没有每个子域独立优化器或多 GPU 并行；
- 子域由材料和几何预先确定，不会自适应拆分；
- 界面约束物理温度与热流，不额外强制不同 PDE 残差相等。

### 6.4 分区域输入归一化

各区域的几何尺度差异很大：铝墙宽度只有 0.01 m，而空气域宽度约为
0.98 m。如果所有网络直接接收物理坐标，薄墙分支在某个方向上只看到非常窄的
输入范围。

因此每个区域都具有自己的中心和尺度，前向传播首先执行：

```python
z = (pts - center) / scale
```

即：

$$
\mathbf z_d=
\frac{\mathbf x-\mathbf c_d}{\mathbf s_d}.
$$

`_INPUT_NORM` 中每一项的结构为：

```python
domain: ((center_x, center_y), (scale_x, scale_y))
```

即第一组数是区域参考中心 $\mathbf c_d=(c_x,c_y)$，第二组数是两个坐标方向的
特征半尺度 $\mathbf s_d=(s_x,s_y)$。它们不是均值和标准差，也不参与训练。

按当前 v4 参数展开后，各区域取值为：

| 区域 | `center=(cx,cy)` [m] | `scale=(sx,sy)` [m] | 设计含义 |
|---|---:|---:|---|
| `tbl` | $(-0.0025,0.5)$ | $(0.0025,0.5)$ | 5 mm 气凝胶显式域；默认一维热阻模式下不创建该网络 |
| `wall_l` | $(0.005,0.5)$ | $(0.005,0.5)$ | 左侧 1 cm 薄墙映射到局部标准尺度 |
| `wall_r` | $(0.995,0.5)$ | $(0.005,0.5)$ | 右侧 1 cm 薄墙映射到局部标准尺度 |
| `wall_b` | $(0.5,0.005)$ | $(0.49,0.005)$ | 下墙横向覆盖 $[0.01,0.99]$，厚度为 1 cm |
| `wall_t` | $(0.5,0.995)$ | $(0.49,0.005)$ | 上墙横向覆盖 $[0.01,0.99]$，厚度为 1 cm |
| `air` | $(0.5,0.5)$ | $(0.49,0.49)$ | 腔内区域整体映射到约 $[-1,1]^2$ |
| `dev1` | $(0.5,0.11)$ | $(0.5,0.1)$ | 竖向按设备半高缩放，横向保留移动位置信息 |
| `dev2` | $(0.5,0.815)$ | $(0.5,0.175)$ | 竖向按设备半高缩放，横向保留移动位置信息 |
| `dev3` | $(0.5,0.4)$ | $(0.5,0.1)$ | 以固定圆心为中心；当前横、纵尺度采用非等比经验设置 |

对于墙体和空气等固定矩形区域，`center` 通常取几何中心，`scale` 通常取对应
半尺寸，因此区域端点接近映射到 $-1$ 和 $1$。

dev1/dev2 的水平中心会随设计变量 $x_1/x_2$ 移动，但输入归一化的水平中心固定
为 0.5、尺度固定为 0.5。这样网络输入仍保留设备当前的全局水平位置，不会因
每次把设备重新居中而丢失位置信息，也避免归一化参数随优化变量变化。竖直方向
位置固定，所以使用设备自身半高进行局部缩放。

dev3 当前采用 $(s_x,s_y)=(0.5,0.1)$，因此圆域在归一化坐标中不是等比圆：
$z_x$ 约在 $[-0.2,0.2]$，$z_y$ 约在 $[-1,1]$。这是现有经验设置，不是物理
要求；后续应与 $(0.1,0.1)$ 的等比局部缩放做消融比较。

例如左墙 $x\in[0,0.01]$，使用 $c_x=0.005$、$s_x=0.005$ 后，可映射到
$[-1,1]$。空气域也被映射到相近的数值范围，使各子网络面对相似尺度的输入。

坐标变换写在网络内部，因此自动微分会自动保留链式法则：

$$
\frac{\partial\theta}{\partial x}
=\frac{1}{s_x}\frac{\partial\theta}{\partial z_x},
\qquad
\frac{\partial^2\theta}{\partial x^2}
=\frac{1}{s_x^2}\frac{\partial^2\theta}{\partial z_x^2}.
$$

`ConductionLoss` 对物理坐标 `pts` 求导时，得到的仍是物理坐标下的一阶、二阶
导数，不需要在损失模块中再次手工换算。

各区域的 `center` 和 `scale` 注册为 buffer，因为它们需要随模型移动和保存，
但不应参与训练。

它们后续参与的计算链为：

```text
物理配点 pts
    ↓  _INPUT_NORM
局部坐标 z=(pts-center)/scale
    ↓  可选 Fourier 编码
区域 MLP 输出
    ↓  _OUTPUT_SCALE
无量纲温度 θ
    ↓  自动微分 / 温度还原
PDE、界面、边界、能量预算和验收指标
```

由于 `_INPUT_NORM` 在模块导入时根据 `config.py` 计算，修改几何参数后需要重启
Python 进程。它们又被注册为 buffer 并写入 checkpoint，所以不能用旧几何
checkpoint 静默覆盖新配置下的归一化参数。

### 6.5 分区域输出尺度

MLP 原始输出还会乘以区域输出尺度：

```python
out = outscale * self.nets[domain](z)
```

即：

$$
\theta_d(\mathbf x)
=s_{\theta,d}NN_d(\mathbf z_d).
$$

`_OUTPUT_SCALE` 是优化层面的预条件，不是温度上下限。例如 dev3 使用较大的
输出尺度，是因为无辐射的静止空气模型中它可能形成较高温升。这样可以让 MLP
内部输出保持在较自然的数量级，减少最后一层必须产生很大权重的压力。

当前各区域取值为：

| 区域 | `outscale` | 含义 |
|---|---:|---|
| `tbl` | 2.0 | 显式气凝胶分支的无量纲输出预尺度 |
| `wall_l/right/bottom/top` | 2.0 | 四个铝墙分支使用统一输出尺度 |
| `air` | 3.0 | 允许空气域以较自然的网络幅值表示较高温升 |
| `dev1` | 2.0 | 小方形设备输出尺度 |
| `dev2` | 2.5 | 大功率方形设备使用稍大的输出尺度 |
| `dev3` | 4.0 | 无固体热桥的圆设备预计温升最高，使用最大输出尺度 |

这些数值是根据确定性参考解数量级设置的粗略优化先验，不是导热系数、功率、
温度上限或设备权重。假设 `dev3` 的 MLP 原始输出为 0.8，则返回的无量纲温度为：

$$
\theta_{dev3}=4.0\times0.8=3.2.
$$

还原后的绝对温度为：

$$
T_{dev3}=T_c+\Delta T\theta_{dev3}
=218.15+85\times3.2
=490.15\ \mathrm K.
$$

输出尺度同样自动进入导数：

$$
\frac{\partial\theta_d}{\partial x}
=\frac{s_{\theta,d}}{s_{x,d}}
\frac{\partial NN_d}{\partial z_x},
$$

$$
\frac{\partial^2\theta_d}{\partial x^2}
=\frac{s_{\theta,d}}{s_{x,d}^2}
\frac{\partial^2NN_d}{\partial z_x^2}.
$$

因此 `_OUTPUT_SCALE` 不只改变前向温度幅值，也改变 PDE、界面热流和边界导数
相对于网络权重的梯度尺度。它不会改变连续方程的理论解空间，但会明显影响优化
条件和收敛速度。

需要区分三种尺度：

| 尺度 | 作用 |
|---|---|
| 全局温度归一化 | 将开尔文温度变换为无量纲 $\theta$ |
| 分域输入归一化 | 处理各材料域的几何尺寸差异 |
| 分域输出尺度 | 改善不同区域温度幅值的优化条件 |

输出层保持线性，所以 `_OUTPUT_SCALE` 不会限制最终温度范围。

后续模块对输出的具体使用如下：

| 后续位置 | 使用方式 |
|---|---|
| `ConductionLoss` | 对缩放后的 $\theta$ 求二阶导，计算各域 Laplace/Poisson 残差 |
| `InterfaceLoss` | 比较两分支的 $\theta$，并用 $k\nabla\theta$ 计算守恒热流 |
| `BoundaryLoss` | 用 $\theta$ 施加 Dirichlet，用其梯度施加 Robin/绝热条件 |
| `EnergyConservationLoss` | 对缩放后温度的物理梯度积分，计算设备和全局排热量 |
| `temperature_K()` | 用 $T=T_c+\Delta T\theta$ 还原开尔文温度 |
| `monitors.py` | 计算设备最高温度、边界热流和界面误差 |

当且仅当 `theta_init > 0` 时，初始化逻辑会把最后一层权重置零，并把偏置除以
对应 `outscale`，确保所有区域乘回不同输出尺度后得到相同的初始无量纲温度。
默认 `theta_init=0` 不进入该分支，各层仍保留 Xavier 初始化。若改变
`_OUTPUT_SCALE`，应重新训练或重新初始化，不能把它当作只影响显示结果的后处理
系数。

### 6.6 空气域不使用解析增强

空气域现在与其他区域一样，完全由自己的 MLP 表示：

$$
\theta_{air}(\mathbf x)=
s_{\theta,air}NN_{air}(\mathbf z_{air}).
$$

代码不再叠加围绕 dev3 的对数 halo，也不再包含 `halo_A`、`HALO_R_FAR` 或
`halo_profile()`。dev3 周围的局部温升、径向梯度和远场传播全部由空气 MLP
在下列物理约束下学习：

- 空气域 Laplace 方程；
- dev3—空气温度连续；
- dev3—空气守恒热流连续；
- dev3 输出功率等于 20 W 的积分能量预算；
- 全局输入功率与外边界排热守恒。

这种处理避免了经验截断半径和轴对称解析先验，使网络结构更通用。相应代价是
dev3 周围的二维对数型导热结构需要由 MLP 自行逼近，训练速度和局部精度必须
通过 FEM 对比、界面热流误差及能量守恒进行验证。若普通 MLP 表达不足，应优先
尝试增加局部配点、调整损失尺度、使用 Fourier 特征或提高网络容量。

### 6.7 统一温度初值 `theta_init`

指定正值 `theta_init > 0` 时，代码把各分支最后一层权重设为零，并将偏置设置为：

$$
b_d=\frac{\theta_{init}}{s_{\theta,d}}.
$$

乘回各区域输出尺度后：

$$
\theta_d=s_{\theta,d}b_d=\theta_{init}.
$$

因此所有材料域从同一常温度场开始，初始界面温度跳跃接近零。这个初值不能
满足设备 Poisson 方程、热流连续和能量守恒，只是降低训练初期的界面冲突。

需要特别注意当前条件判断：

```python
if theta_init is not None and theta_init > 0:
```

所以默认配置中的 `theta_init=0.0` 会跳过统一平场初始化，网络从 Xavier 随机场
开始。若确实需要 $	heta=0$ 的严格统一冷端初值，当前代码需要把判断改为仅检查
`theta_init is not None`；仅在命令行传入 `--theta-init 0` 不会改变现有行为。

### 6.8 前向传播与温度还原

所有材料域采用相同形式的前向过程：

```text
物理坐标 pts
    ↓
读取该区域 center / scale
    ↓
变换为局部归一化坐标 z
    ↓
调用 nets[domain](z)
    ↓
乘以 outscale_domain
    ↓
返回无量纲温度 θ_domain
```

`forward()` 返回无量纲温度，`temperature_K()` 使用：

$$
T=T_c+\Delta T\,\theta
$$

还原绝对温度。PDE 训练主要使用无量纲温度；最高温度、FEM 对比以及后续辐射
$T^4$ 计算必须使用还原后的绝对温度。

### 6.9 与物理损失的职责划分

`TemperatureField` 只产生候选温度函数，物理解由损失模块确定：

| 层级 | 职责 |
|---|---|
| `MLP` | 学习连续、平滑、可二阶微分的标量函数 |
| `TemperatureField` | 管理分域网络、输入归一化和输出尺度 |
| `losses/conduction.py` | 对各域施加 Laplace/Poisson 方程 |
| `losses/interface.py` | 连接各分支，施加温度和热流连续条件 |
| `losses/boundary.py` | 施加气凝胶热阻、右侧热沉和上下绝热条件 |
| `losses/energy_conservation.py` | 施加设备级与全局积分能量守恒 |

这种设计能够自然表达材料界面的导数跳变，并允许针对不同区域使用不同网络
尺度和特征配置。代价是子网络参数量较多，界面连续完全依赖软约束，而且
`_OUTPUT_SCALE` 带有一定经验性，必须通过独立 FEM 和消融实验验证。

### 6.10 SDF 几何表达与采样模块

几何定义与随机配点采用两个独立模块：

| 模块 | 职责 |
|---|---|
| `geometry.py` | 定义几何尺寸、SDF、区域归属、界面拓扑、法向和界面长度 |
| `sampling.py` | 根据几何信息生成内部域、材料界面和外边界配点 |

`geometry.py` 中的隐式几何统一采用以下符号约定：

$$
F(x,y)<0\quad\text{区域内部},\qquad
F(x,y)=0\quad\text{几何边界},\qquad
F(x,y)>0\quad\text{区域外部}.
$$

例如圆设备的有符号距离函数为：

$$
F_{dev3}(x,y)=
\sqrt{(x-x_c)^2+(y-y_c)^2}-R_3.
$$

矩形墙体和方块设备使用轴对齐矩形的精确欧氏 SDF。空气域则通过 CSG 差集
表示为：

$$
F_{air}=\max\left(
F_{cavity},-F_{dev1},-F_{dev2},-F_{dev3}
\right).
$$

因此空气区域等于腔体减去三个设备。`domain_sdf()` 是按域名取得隐式场的统一
入口；`mask_domain()` 和原有 `mask_*()` 都由 $F\le0$ 派生，不再重复编写坐标
大小比较。`label_points()` 也根据各域 SDF 的符号确定点的材料归属。矩形角点和
CSG 的 `max` 等距交会处不可微，这是标准 SDF 表达的正常性质；除这些零测度
位置外，移动设备 SDF 对 $x_1,x_2$ 保留自动微分关系。

`sampling.py` 当前采用以下配点方法：

- 矩形内部：在轴对齐包围盒内均匀随机采样；
- 空气内部：在腔体中生成候选点，再用 $F_{air}\le0$ 做拒绝采样；
- 圆设备内部：令 $r=R\sqrt{u}$、$\varphi=2\pi v$，保证按面积均匀，而不是让
  点过度集中在圆心；
- 直线界面：沿实际线段均匀随机采样；不连续线段并集按各段长度分配概率；
- 圆周界面：对极角均匀采样，同时返回径向单位法向；
- 外边界：分别生成左、右、上、下边界点，并附带所属温度子域和方向信息。

采样器返回的是损失函数所需的离散配点；几何模块提供的是连续的隐式形状和
拓扑信息。这样的拆分使 SDF 可同时服务于点分类、拒绝采样、可行性判断和后续
自适应加密，而不会把随机采样逻辑重新混入几何定义。

---

## 7. 积分能量守恒损失

### 7.1 为什么逐点 PDE 之外还需要能量约束

`losses/energy_conservation.py` 定义了设备级和系统级积分能量守恒约束。理论上，当各域
PDE、界面热流连续和外边界条件都精确满足时，积分能量守恒会自动成立；但在
PINN 训练过程中，这些条件只被近似满足。

尤其空气与铝的导热系数比为：

$$
\frac{k_f}{k_{Al}}
=\frac{0.026}{167}
\approx1.56\times10^{-4}.
$$

网络可能先得到一个局部 PDE 残差较小的平滑场，却没有建立正确的空气温度梯度
和总热流幅值。积分约束直接向优化器提供以下低频、全局信息：

```text
dev1 必须输出 270 W
dev2 必须输出 540 W
dev3 必须输出 20 W
左右边界合计必须排出 830 W
```

因此它是逐点 PDE 和界面损失的补充，不能替代它们。

### 7.2 与高斯–格林公式的关系

设备区域 $\Omega_k$ 内满足稳态导热方程：

$$
\nabla\cdot(k\nabla T)+\dot q'''_k=0.
$$

定义傅里叶热流：

$$
\mathbf q=-k\nabla T,
$$

则局部守恒方程可写为：

$$
\nabla\cdot\mathbf q=\dot q'''_k.
$$

对整个设备区域积分：

$$
\int_{\Omega_k}\nabla\cdot\mathbf q\,dV
=\int_{\Omega_k}\dot q'''_k\,dV
=P_k.
$$

利用高斯–格林公式：

$$
\int_{\Omega_k}\nabla\cdot\mathbf q\,dV
=\int_{\partial\Omega_k}\mathbf q\cdot\mathbf n_k\,dA,
$$

得到：

$$
\boxed{
\int_{\partial\Omega_k}
-k\nabla T\cdot\mathbf n_k\,dA
=P_k
}.
$$

因此，设备侧能量预算确实是把区域内部体积热源的积分转换为设备边界法向热流
通量的积分。这里使用的更准确名称是散度定理或高斯–格林公式，而不是利用
格林函数构造边界积分解。

这也可以从有限元弱形式理解。将 PDE 乘测试函数 $v$ 并分部积分：

$$
-\int_{\Omega_k}k\nabla v\cdot\nabla T\,dV
+\int_{\partial\Omega_k}v,k\nabla T\cdot\mathbf n_k\,dA
+\int_{\Omega_k}v\dot q'''_k\,dV
=0.
$$

当取常数测试函数 $v=1$ 时，$\nabla v=0$，体内梯度项消失，剩下的就是上述
整体能量守恒。因此 `EnergyConservationLoss` 也可以理解为 PDE 在常数测试函数上的
全局弱形式约束。

需要强调：代码没有用该积分约束替代设备内部 Poisson 方程。两者职责不同：

```text
ConductionLoss：决定设备内部温度曲率和局部分布
EnergyConservationLoss：保证体积总发热与边界总排热相等
```

如果只保留边界积分，虽然能要求设备排出正确总功率，却不能保证内部温度场和
热点位置正确。

### 7.3 表面热流计算

对区域 `dom` 的界面点，先由温度网络得到无量纲温度并求梯度：

$$
\theta_d=TemperatureField(d,\mathbf x),
\qquad
\nabla\theta_d=autograd(\theta_d,\mathbf x).
$$

利用：

$$
T=T_c+\Delta T\theta,
$$

沿设备外法向 $\mathbf n_k$ 的物理导热流密度为：

$$
q''_{d\rightarrow n_k}
=-k_d\frac{\Delta T}{L_{ref}}
\nabla\theta_d\cdot\mathbf n_k.
$$

方块各表面的 $\mathbf n_k$ 是固定方向，圆形设备使用圆周径向单位向量。法向
始终按“设备向外”定义：计算设备侧时它就是设备外法向；计算接收侧时仍沿同一
方向投影，以检查墙体或空气是否承接了相同方向、相同数值的热流。

界面点按线长均匀采样，因此二维挤出模型中的表面热量近似为：

$$
Q_{k,j}
\approx
\overline{q''}_{k,j}\,L_{k,j}\,b,
$$

其中 $L_{k,j}$ 是二维界面线长，$b=1\,\mathrm m$ 是挤出厚度。圆设备使用
$L=2\pi R_3$，方块单边使用 $D_1$ 或 $D_2$。普通界面连续损失仍使用均匀
随机点；积分能量损失则使用独立、固定的等距中点求积点。二者分离后，Adam
每轮仍能覆盖不同位置，而能量目标不会因蒙特卡洛换点产生额外波动。

具体来说，若在第 $j$ 个界面上均匀独立采样 $N$ 个点，则
`_face_flux()` 先计算每个点的法向热流密度：

$$
q''_i=-k\frac{\Delta T}{L_{ref}}
\nabla\theta(\mathbf x_i)\cdot\mathbf n_i,
$$

再由 `q_out.mean()` 得到算术平均：

$$
\overline{q''}_{k,j}
=\frac{1}{N}\sum_{i=1}^{N}q''_i.
$$

于是代码中的

```python
qm = q_out.mean()
Q = Q + qm * length * C.B
```

对应蒙特卡洛积分估计：

$$
Q_{k,j}\approx
\frac{L_{k,j}b}{N}\sum_{i=1}^{N}q''(\mathbf x_i).
$$

各界面不是合并后直接平均，而是分别求平均、乘各自真实面积 $L_{k,j}b$，再对
设备所有表面求和。因此长界面对总热量的贡献自然大于短界面。

固定中点规则消除了训练期间的随机积分方差；对光滑热流，其确定性求积误差通常
也比同点数纯随机采样更小。增加 `n_energy` 只能降低积分离散误差，不能修正温度
网络、自动微分梯度、导热系数、法向或界面长度本身的误差。角点和热流陡变区域
仍可能需要局部加密。

### 7.4 每个设备的双侧预算

`geometry.DEV_IFACES` 给出每个设备的积分表面：

| 设备 | 纳入积分的界面 |
|---|---|
| dev1 | 安装底面，以及左、右、上三个空气接触面 |
| dev2 | 安装顶面，以及左、右、下三个空气接触面 |
| dev3 | 整个圆周空气接触面 |

每个设备分别计算两套总热量：

$$
Q_k^{dev}
=\sum_jQ_{k,j}^{dev},
$$

$$
Q_k^{recv}
=\sum_jQ_{k,j}^{recv}.
$$

- $Q_k^{dev}$ 使用设备自身温度网络的梯度；
- $Q_k^{recv}$ 在同一组界面点上使用相邻空气或墙体网络的梯度。

两侧都约束为设备功率：

$$
Q_k^{dev}=P_k,
\qquad
Q_k^{recv}=P_k.
$$

只约束设备侧时，设备网络可能已经输出正确总功率，但独立的空气/墙体网络尚未
形成相应承接热流。双侧预算为低导热接收分支提供直接的幅值驱动，并与逐点
`InterfaceLoss` 相互补充。

其中设备侧预算直接来自设备 Poisson 方程的高斯–格林积分；接收侧并不是对整个
空气域或墙体域再次应用同一个区域公式，因为接收域还具有其他边界。它更准确地
表示设备界面热流连续条件的积分形式：

$$
\int_{\partial\Omega_k}
\mathbf q_{recv}\cdot\mathbf n_k\,dA
=
\int_{\partial\Omega_k}
\mathbf q_{dev}\cdot\mathbf n_k\,dA
=P_k.
$$

设备残差按自身功率归一化：

$$
r_k^{side}
=\frac{Q_k^{side}-P_k}{P_k},
\qquad side\in\{dev,recv\}.
$$

这样 20 W 的 dev3 不会被 540 W 的 dev2 在绝对数值上掩盖。设备级损失为：

$$
L_{eng,device}
=\sum_{k=1}^{3}
\left[
\left(r_k^{dev}\right)^2
+\left(r_k^{recv}\right)^2
\right].
$$

### 7.5 全局左右边界预算

左侧计算边界位于 $x=0$ 的 `wall_l` 外表面，外法向为 $(-1,0)$，因此：

$$
Q_{left}
=k_{Al}\frac{\Delta T}{L_{ref}}
\int_0^1\frac{\partial\theta_{wall_l}}{\partial x}\,b\,dy.
$$

它代表热量从左铝墙经一维气凝胶热阻排向冷端。右侧外法向为 $(1,0)$：

$$
Q_{right}
=-k_{Al}\frac{\Delta T}{L_{ref}}
\int_0^1\frac{\partial\theta_{wall_r}}{\partial x}\,b\,dy.
$$

右侧无论采用 Dirichlet 基准还是 Robin 对比边界，实际排热量都由墙体温度梯度
积分得到。上下外边界按绝热处理，理论排热为零，因此不进入主全局预算；其数值
泄漏由 `monitors.py` 另行记录。

从整个共轭传热区域应用高斯–格林公式时，各内部材料界面成对出现。若界面热流
连续，且两侧外法向相反，则内部界面积分相互抵消，最终只剩系统外边界：

$$
Q_{left}+Q_{right}+Q_{top}+Q_{bottom}=P_{tot}.
$$

利用上下绝热条件，才进一步化为 $Q_{left}+Q_{right}=P_{tot}$。

全局相对残差为：

$$
r_g
=\frac{Q_{left}+Q_{right}-P_{tot}}
{P_{tot}},
\qquad
P_{tot}=830\ \mathrm W.
$$

对应损失为：

$$
L_{eng,global}=r_g^2.
$$

最终：

$$
L_{eng}
=L_{eng,device}+L_{eng,air}+L_{eng,global}.
$$

当前实现还加入无体热源空气域的积分弱守恒：

$$
r_{air}=\frac{1}{P_{tot}}
\int_{\partial\Omega_{air}}-k_f\nabla T\cdot\mathbf n_{air}\,dA,
\qquad L_{eng,air}=r_{air}^2.
$$

空气边界包括四段墙体接触面和全部设备接触面。该项把设备接收侧热流与墙体接收
热流直接联系起来，减少“设备已经输出正确总功率，但热量没有正确传递到外墙”
的中间丢失。

`main.py` 再通过权重 $\lambda_{eng}$ 将它加入总训练损失，当前默认
`w_eng=100`。

### 7.6 功率 continuation 与日志量

`EnergyConservationLoss.power_scale` 与 `ConductionLoss.power_scale` 同步。若调试时使用
0.1 倍功率，则 PDE 体热源和积分目标同时变为：

$$
(P_1,P_2,P_3)=(27,54,2)\ \mathrm W,
$$

避免控制方程与能量预算使用不同工况。

模块返回：

```text
total
details = {
    eng_dev1_dev,  eng_dev1_recv,
    eng_dev2_dev,  eng_dev2_recv,
    eng_dev3_dev,  eng_dev3_recv,
    eng_air, eng_global,
    eng_Q_left_W, eng_Q_right_W
}
```

`total` 保留计算图并参与反向传播；`details` 中的带符号相对残差以及固定求积
得到的左右排热量经过 `detach()`，只用于日志和诊断，并另存到
`energy_log.csv`。这样汇总的 `eng` 不会掩盖某个设备、空气域或全局预算错误。

### 7.7 与其他损失的关系及适用范围

四类损失的关系为：

```text
ConductionLoss：逐点满足各域 Laplace/Poisson 方程
InterfaceLoss：逐点满足温度和守恒热流连续
BoundaryLoss：逐点满足气凝胶热阻、右侧热沉和绝热条件
EnergyConservationLoss：积分校正设备功率和系统总排热幅值
```

积分预算只能保证“总量正确”，不能保证局部场正确。例如设备总排热等于额定
功率，并不意味着设备内部 PDE、各表面热流分配和界面温度已经正确。因此验收时
必须同时检查独立点 PDE 残差、界面误差和 FEM 温度场。

当前实现只适用于 Milestone 1 纯导热。加入腔内辐射后，接收侧预算不能继续
要求导热量单独等于全部设备功率，而应满足：

$$
P_k=Q_{cond,k}+Q_{rad,k}.
$$

届时需要把辐射热流纳入设备/接收侧积分，或只保留设备固体侧总输出约束并由
完整界面能量方程分配导热和辐射。阶段 B 加入自然对流后，也必须保证空气域
积分预算与对流—扩散能量方程一致。

---

## 8. `main.py` 的训练策略

### 8.1 总体流程

Milestone 1 的训练不是一次性把所有自由度同时释放，而是在固定几何布局下先求
稳定的多材料温度场：

```text
设置随机种子与计算设备
        ↓
建立分域 TemperatureField，冻结 x1、x2
        ↓
Adam + 每轮随机重采样 + 可选功率 continuation
        ↓
可选 L-BFGS 固定配点精修
        ↓
独立监控热流、能量平衡和设备最高温度
        ↓
保存 checkpoint 和 final_report.json
```

`DesignVars` 仍用 sigmoid 将 $x_1,x_2$ 限制在解析可行区间，但当前以
`trainable=False` 创建，因此优化器只接收 `field.parameters()`。这保证本阶段
先验证固定布局的正问题，不把温度场误差混入位置优化。

### 8.2 初始化、复现与启动模式

程序同时设置 PyTorch 与 NumPy 随机种子，并统一使用 `float32`。`--device auto`
优先选择 `cuda:0`，没有 CUDA 时退回 CPU。网络可通过正值 `theta_init` 采用统一
温度初值，并可按域使用 Fourier 特征；默认 `theta_init=0` 实际保留 Xavier 随机
初值。

三种启动方式为：

| 模式 | 行为 |
|---|---|
| fresh | 从新网络、新 Adam 和新调度器开始，不读取旧 3.1.2 checkpoint |
| `--resume` | 从当前案例的 `latest.pt` 恢复场网络、Adam、调度器和 epoch |
| `--resume --resume-from <路径>` | 从指定历史 checkpoint 恢复，用于跳过 `latest.pt` 中更晚的 L-BFGS 快照 |
| `--resume --resume-lr <值>` | 完整恢复后覆盖当前 Adam 学习率；保留动量，旧余弦周期耗尽时为剩余轮数新建调度周期 |
| `--init-from` | 只载入场网络权重，重新创建优化器和调度器，用于跨功率热启动 |

checkpoint 路径包含布局、案例版本、右边界类型和功率倍率，降低不同物理工况
之间误加载权重的风险。

推荐训练策略的其余参数集中在 `config.TRAIN` 中，因此正式启动通常只需指定
学习率、Adam epoch 数、布局和设备：

```powershell
python src/main.py --lr 1e-3 --epochs 30000 --layout center --device cuda:0
```

命令行仍保留全部高级选项，用于消融实验时临时覆盖配置，而不需要修改源码。

### 8.3 组合损失与权重

每批配点上的总损失为：

$$
L=\lambda_{pde}L_{pde}
+\lambda_{if,T}L_{if,T}
+\lambda_{if,q}L_{if,q}
+\lambda_{bc}L_{bc}
+\lambda_{eng}L_{eng}.
$$

当前默认权重为：

| 损失 | 默认权重 |
|---|---:|
| 域内 PDE | `w_pde=1` |
| 设备 Poisson 残差内部附加权重 | `w_pde_dev=100` |
| 界面温度连续 | `w_if_T=10` |
| 界面热流连续 | `w_if_q=100` |
| 外边界条件 | `w_bc=100` |
| 积分能量守恒 | `w_eng=100` |

`w_pde_dev` 在 `ConductionLoss` 内部强调设备热源方程，其余权重在
`compute_loss()` 中组合。能量守恒损失和控制方程共享同一个 `power_scale`，避免
体热源与目标排热量对应不同功率工况。

### 8.4 Adam、动态配点和学习率退火

第一阶段采用 Adam。每个 epoch 都重新生成：

- 各材料域内部 PDE 点；
- 全部材料/接触界面点；
- 外边界点。

因此 Adam 优化的是连续物理域上损失期望的随机近似，而不是记忆一组永久固定的
配点。默认空气域点数最多，薄壁、设备和各界面分别保留独立点数；具体数量由
`config.TRAIN` 控制。

学习率采用余弦退火：

$$
\eta_e=\eta_{min}
+\frac{1}{2}(\eta_0-\eta_{min})
\left[1+\cos\left(\frac{\pi e}{E}\right)\right],
\qquad \eta_{min}=0.01\eta_0.
$$

训练前期用较大学习率建立整体温度幅值，后期逐渐降低学习率以细化 PDE、界面和
边界残差。

### 8.5 功率 continuation

高功率解相对常温平场具有较大的温升和梯度，直接施加全部功率可能使网络难以
跨过初始损失屏障。因此代码允许把功率倍率从 $p_0$ 平滑增加到目标 $p_f$。
在 `ramp_frac * epochs` 之前，线性形式为：

$$
p(e)=p_0+(p_f-p_0)f,
$$

指数形式为：

$$
p(e)=p_0\left(\frac{p_f}{p_0}\right)^f,
\qquad
f=\min\left(\frac{e}{r_{frac}E},1\right).
$$

当前 `config.TRAIN` 默认从 $p_0=0.1$ 指数提升到 $p_f=1$，爬坡覆盖 Adam
阶段前 80%。将 `power_start` 设为与 `power_scale` 相同即可关闭实际爬坡。
continuation 同时缩放设备 Poisson 源项和能量守恒目标。

### 8.6 L-BFGS 精修与固定配点

Adam 完成后进入拟牛顿精修，默认步数由 `config.TRAIN["lbfgs_steps"]` 控制；
设为 0 可以关闭。设备内部曲率、界面热流和
低导热空气温度梯度会形成刚性的耦合损失谷；L-BFGS 利用历史参数差和梯度差
近似二阶曲率，并通过 strong-Wolfe 线搜索选择步长。

一次 `lb.step(closure)` 可能反复调用 `closure()`。为保证这些函数评估对应同一
目标，域内、界面和边界配点在一个 L-BFGS 步块内保持固定。参数
`lbfgs_resample` 控制每隔多少个外层 step 更新整组配点；设为 0 则全程固定。
这在局部优化稳定性与长期空间覆盖之间折中。

需要注意，PyTorch 的 `LBFGS.step()` 返回值通常是本次调用开始时的 closure
损失，不一定等于更新后的最终损失；代码随后重新计算的各分项损失更能反映当前
参数状态。L-BFGS 阶段生成的历史 checkpoint 也保存拟牛顿优化器状态；当前
`--resume` 入口仍以恢复场网络和 Adam/调度器为主，若要从某一 L-BFGS step
原样续接，还需在训练入口显式加载其中的 `lbfgs` 状态。

### 8.7 训练监控、保存与最终验收

Adam 每 `eval_every` 轮、L-BFGS 每 10 个外层 step 记录：

- 总损失以及 PDE、界面温度、界面热流、边界和能量守恒分项；
- 左右边界排热量与全局能量平衡误差；
- 右侧边界温度误差及 Robin 对照热流；
- 三个设备的估计最高温度；
- 学习率和单步耗时。

这些量写入 `loss_log.csv`。Adam 每 `save_every` 轮、L-BFGS 每 50 步及阶段末尾
同时保存按累计编号命名的 `epoch_XXXXXX.pt`，并刷新便于续训的 `latest.pt`。
L-BFGS 使用 `epoch_tag=epochs+step`，因此完成 30000 轮 Adam 和 200 步 L-BFGS
后，`latest.pt` 的 `epoch` 为 30200、`phase` 为 `lbfgs`。L-BFGS 历史快照额外
包含拟牛顿优化器状态，但当前入口不会恢复该状态继续同一段 L-BFGS。训练结束后
强制恢复目标全功率，
调用 `full_report()` 生成 `final_report.json`，最后提示运行 `validate.py` 与独立
FEM 结果比较。

训练损失下降只是必要条件。正式验收应同时检查 PDE、界面连续、边界条件、积分
能量守恒、设备最高温度和 FEM 场误差，不能仅凭加权总损失判断物理解已经可靠。

当前 `validate.py` 已自动计算 FEM 逐域/全局温度场误差、设备最高温度、左右总
能量、两个安装面的连续误差和气凝胶一维热阻；它尚未在独立配点上重新计算所有
域 PDE、全部17组界面及全部外边界残差。后者仍是验收计划中的缺口，不能把训练
日志中的随机批次残差称为完整的独立残差验证。

### 8.8 终端 `print`、CSV 与报告字段

Adam 阶段的终端输出格式为：

```text
ep  100  loss ...  pde ...  ifT ...  ifq ...  bc ...  eng ... |
Q_L ...  Q_R ...  bal ... | T1 ...C  T2 ...C  T3 ...C
```

L-BFGS 阶段内容相同，只把开头换成 `lb step`。各字段含义为：

| 字段 | 含义 | 单位/性质 |
|---|---|---|
| `ep` | Adam 的 epoch 编号 | 整数 |
| `lb` | L-BFGS 外层 step 编号 | 整数 |
| `loss` | 加权总损失 | 无量纲组合量 |
| `pde` | `ConductionLoss`：所有域的 Laplace/Poisson 残差 | 未乘 `w_pde` 的分项值 |
| `ifT` | 材料界面两侧的温度连续残差 | 未乘 `w_if_T` 的分项值 |
| `ifq` | 材料界面两侧的守恒热流连续残差 | 未乘 `w_if_q` 的分项值 |
| `bc` | 左侧气凝胶热阻、右侧边界及上下绝热残差 | 未乘 `w_bc` 的分项值 |
| `eng` | 每台设备双侧及全局积分能量守恒残差 | 未乘 `w_eng` 的分项值 |
| `Q_L` | 通过左外边界向计算域外排出的热量 | W |
| `Q_R` | 通过右外边界向计算域外排出的热量 | W |
| `bal` | 左右排热与当前总输入功率的相对不平衡 | 无量纲 |
| `T1` | dev1 内监控网格上的最高温度 | ℃ |
| `T2` | dev2 内监控网格上的最高温度 | ℃ |
| `T3` | dev3 内监控网格上的最高温度 | ℃ |

其中终端的总损失满足：

$$
L=\lambda_{pde}L_{pde}
+\lambda_{if,T}L_{if,T}
+\lambda_{if,q}L_{if,q}
+\lambda_{bc}L_{bc}
+\lambda_{eng}L_{eng},
$$

但 `pde/ifT/ifq/bc/eng` 打印的是乘外层权重之前的原始分项，便于分别判断各类
物理约束是否改善。因此不能把几个打印分项直接相加来复原 `loss`，必须先乘对应
权重；`ConductionLoss` 内部的设备附加权重 `w_pde_dev` 已经包含在 `pde` 中。

能量平衡字段定义为：

$$
\mathrm{bal}
=\frac{|P_{in}-Q_L-Q_R|}{P_{in}}.
$$

它使用独立的规则边界监控点计算，不直接复用随机训练批次。当前上下边界应绝热，
所以终端只用左右排热进行主平衡；上下泄漏量仍会进入完整能量报告。`bal=0.01`
表示左右边界排热与输入功率相差 1%。

`loss_log.csv` 比终端多保存以下字段：

| CSV 字段 | 含义 |
|---|---|
| `Q_right_robin` | 按右壁平均温度和 Robin 公式计算的对照排热量；Dirichlet 模式可为空 |
| `right_T_rms_err_K` | 右侧温度相对目标温度的 RMS 误差，单位 K |
| `lr` | 当前 Adam 学习率；L-BFGS 日志中写为 `-1` |
| `sec` | 本次记录对应步骤的耗时，单位 s |

`energy_log.csv` 另外记录三台设备双侧残差、`eng_air`、`eng_global`，以及固定
求积得到的 `eng_Q_left_W/eng_Q_right_W`。这些字段用于判断汇总 `eng` 的主导
来源，并与 `loss_log.csv` 中独立监控网格计算的左右热流交叉检查。

CSV 中还保存与终端相同的 epoch、损失、热量和最高温度字段。Adam 的 `loss` 是
该 epoch 参数更新前在当前随机批次上算得的值；PyTorch 的 L-BFGS 返回值通常是
该次 `step()` 首次 closure 的损失，因此 L-BFGS 行中的总 `loss` 不保证是线搜索
完成后的值，而随后重新计算的五个分项更接近更新后的当前状态。

训练结束时，终端还会打印完整 `final_report.json`。其主要层级包括：

- `energy`：输入功率、四侧热流、左右 Robin 对照、右壁温度误差、全局平衡和
  上下绝热泄漏；
- `T_max_K/T_max_C`：三个设备的最高温度；
- `aerogel`：气凝胶等效一维热阻检查；
- `mount`：设备安装界面的温度与热流连续误差；
- `constraints`：dev1、dev2 是否低于 $70\,^{\circ}\mathrm C$；
- `feasible_dev12_70C` 与 `objective_Tmax3_C`：当前布局的可行性和 dev3 指标；
- 布局、案例版本、边界类型、$x_1,x_2$、功率倍率和总运行时间。

这些监控量不全部参与反向传播。`loss` 负责训练，`Q_L/Q_R/bal/T1--T3` 和最终
报告负责判断训练得到的场是否具有正确的工程物理意义。

### 8.9 模型训练说明与参数调节指南

本节面向 Milestone 1 固定布局正问题。目标不是获得最低的加权 `loss` 数字，
而是让域内 PDE、材料界面、外边界、积分能量和独立 FEM 场误差同时满足要求。

#### 8.9.1 推荐基线与启动命令

当前推荐基线集中在 `config.TRAIN`：

| 参数组 | 默认设置 |
|---|---|
| 网络 | `width=96`，`depth=5`，`theta_init=0`（按当前判断表示保留 Xavier 随机初值） |
| Adam | `lr=1e-3`，`epochs=30000` |
| 功率 | `power_start=0.1`，`power_scale=1`，指数爬坡覆盖前 80% Adam |
| L-BFGS | 200 个外层 step，`max_iter=20`，`history=50`，每100步换点 |
| 域内点 | air 6000；四墙1000--1500；三设备1200--1500 |
| 边界点 | 每组界面256点，每段外边界256点 |
| 损失权重 | PDE 1，设备 PDE 100，界面温度10、通量100，边界100，能量100 |
| 记录 | 每100轮评估，每1000轮保存 checkpoint |

通常只在命令行指定四个常用参数：

```powershell
D:\ANACONDA\envs\Pytorch\python.exe src\main.py `
  --lr 1e-3 --epochs 30000 --layout center --device cuda:0
```

第一次完整运行应保留这组基线。不要在尚未得到基线日志前同时修改网络、采样、
损失权重和功率课程，否则无法判断是哪一项带来改善或退化。

#### 8.9.2 推荐调节顺序

参数建议按下列顺序处理：

1. 先检查几何、材料参数、功率和边界条件是否正确；
2. 用短训练确认损失有限、梯度不出现 `NaN/Inf`、各监控量方向合理；
3. 调整学习率和功率爬坡，使 Adam 能稳定建立温度与热流幅值；
4. 根据独立分项调整采样数量，优先增加误差集中的区域，而不是全域等比例加点；
5. 再调整损失权重，使某一物理条件不会长期被其他项掩盖；
6. Adam 已进入平台后才判断 L-BFGS 精修效果；
7. 普通 MLP 容量不足时再增加宽度/深度或启用 Fourier 特征；
8. 用未参与训练的规则网格和 FEM 结果验收，而不是用训练批次自证精度。

每次实验原则上只改变一组参数，并保留随机种子、布局和物理工况。重要结论至少
使用多个随机种子复核，避免把一次有利初始化误认为稳定收敛规律。

#### 8.9.3 `lr`、`epochs`、`width` 和 `depth`

| 参数 | 增大后的主要作用 | 过大风险 | 何时调整 |
|---|---|---|---|
| `lr` | 更快离开初始平场 | 损失振荡、热量换向、出现非有限值 | 所有分项同步剧烈波动时减小；下降过慢且梯度稳定时小幅增大 |
| `epochs` | 给 Adam 更多动态配点更新 | 时间增加，不能修复错误权重或错误物理 | 多项仍稳定下降且尚未平台时增加 |
| `width` | 增加每层特征容量 | 显存、时间和过拟合固定配点风险上升 | 多区域都欠拟合时优先增加 |
| `depth` | 增加函数组合能力 | `Tanh` 梯度传播更难、二阶导数训练更敏感 | 宽度增加仍不能表达复杂场时再增加 |

学习率建议按倍率调整，而不是做很细的十进制搜索。例如从 $10^{-3}$ 依次比较
$5\times10^{-4}$ 和 $2\times10^{-4}$。若只有某个物理分项不下降，通常应先查该项
的尺度和采样，不应立即把全局学习率降低几个数量级。

增加网络容量是否有效，应以相同训练预算下的独立验证误差判断。训练损失降低而
验证网格误差不变，说明瓶颈更可能在采样、损失尺度或优化过程，而非网络容量。

#### 8.9.4 功率 continuation 参数

| 参数 | 含义 | 调节方式 |
|---|---|---|
| `power_scale` | 最终功率倍率 | 正式 v4 验证保持1；小倍率只用于调试或分阶段热启动 |
| `power_start` | Adam 初始功率倍率 | 初期热流/温度剧烈发散时降低；初始阶段已稳定时可提高以缩短训练 |
| `ramp` | `none/linear/exp` | `exp` 在低功率停留更久；`linear` 更快进入中高功率 |
| `ramp_frac` | 达到最终功率占 Adam 总轮数的比例 | 高功率阶段不稳定时增大；很早已稳定时减小 |

必须保证 `ConductionLoss.power_scale` 和 `EnergyConservationLoss.power_scale`
同步。否则设备体热源与边界总排热目标不同，会产生不存在的冲突损失。

如果使用低功率 checkpoint 热启动全功率，优先使用 `--init-from`，只继承场网络
而重新建立优化器；只有同一工况、同一训练计划的中断恢复才使用 `--resume`。

同一工况续训但需要降低当前学习率时，使用 `--resume-lr`。例如从已有 epoch
继续到总计 30000 epoch，并把恢复后的当前学习率改为 $5\times10^{-4}$：

```powershell
D:\ANACONDA\envs\Pytorch\python.exe src\main.py `
  --epochs 30000 --layout center --device cuda:0 `
  --resume --resume-lr 5e-4
```

这里的 `--epochs 30000` 表示 Adam 阶段的累计目标 epoch，而不是再增加 30000
轮。`--resume-lr` 不清空 Adam 的一、二阶动量，也不把余弦调度器退回第 0 轮；
它只重设恢复时刻的学习率。若旧余弦周期尚未结束，则保留已有调度进度；若旧
周期已经结束且累计目标 epoch 更大，则为剩余 Adam 轮数建立新的余弦退火周期。

若 30000 轮 Adam 后已经执行了 L-BFGS，`latest.pt` 会指向 L-BFGS 快照。要从
Adam 第 30000 轮继续到累计第 60000 轮，应明确恢复 Adam 历史快照，并保持全
功率，避免 continuation 回退：

```powershell
D:\ANACONDA\envs\Pytorch\python.exe src\main.py `
  --epochs 60000 --layout center --device cuda:0 `
  --resume --resume-from checkpoint_m1\center_v4_dirichlet\epoch_030000.pt `
  --resume-lr 1e-4 --power-start 1 --power-scale 1 --ramp none --no-plot
```

该命令仍使用原来的 checkpoint 和输出目录，并从 Adam epoch 30001 训练到
epoch 60000，之后再执行配置中的 L-BFGS 精修。由于目录中已经存在第一次运行的
30010--30200 L-BFGS 日志行，新的 Adam 日志会继续追加在文件末尾，CSV 的 epoch
列在这一小段内不是严格递增；同名的 `epoch_030050.pt`、`epoch_030100.pt` 等
历史快照也会被后续 Adam 保存覆盖。`epoch_030000.pt` 本身不会被覆盖。若必须
完整保留第一次 L-BFGS 历史，应在续训前另行复制旧目录；这是“保持原文件夹”
带来的记录层面代价。

#### 8.9.5 配点数量与分布

| 参数 | 控制对象 | 应增加的典型现象 |
|---|---|---|
| `n_dom[d]` | 域 $d$ 的 PDE 点数 | 该域独立 PDE 或 FEM 场误差明显高于其他域 |
| `n_iface` | 每组材料/接触界面点数 | `ifT/ifq` 波动大，或界面验证误差集中在局部 |
| `n_bnd` | 每段外边界点数 | `bc`、右壁温度或绝热泄漏估计不稳定 |
| `n_energy` | 固定能量中点求积数 | 求积加密后能量积分仍明显变化时增加；默认512 |

空气域面积最大且包含圆形设备周围的陡峭梯度，因此默认点数最多。薄壁虽然面积
小，但尺度很薄，不能简单按面积比例削减点数。设备—墙安装面、圆周和角点附近
若出现局部误差，应采用局部加密或残差自适应采样；全域点数翻倍通常成本更高，
也可能仍然错过窄误差区。

动态随机采样会使分项存在正常批次噪声。判断点数不足应观察多个评估周期的均值
和方差，以及固定验证网格误差，不能仅凭相邻两个 epoch 的升降。

#### 8.9.6 损失权重调节

总损失的外层权重只决定优化器关注比例，不改变真实物理定律。调权的目标是让
各项都能下降，而不是用某个大权重把其他错误隐藏起来。

| 参数 | 对应约束 | 增大的条件 | 过大时的表现 |
|---|---|---|---|
| `w_pde` | 全部域内 PDE | PDE 长期不降而界面/边界已较低 | 边界和界面条件被牺牲 |
| `w_pde_dev` | 三个设备 Poisson 源项 | 设备内部曲率不足、发热解趋向平场 | 设备 PDE 支配总梯度，空气和界面跟不上 |
| `w_if_T` | 界面温度连续 | `ifT` 或安装温差长期偏高 | 可产生温度连续但通量错误的解 |
| `w_if_q` | 界面热流连续 | `ifq`、界面通量验证误差长期偏高 | 强导热侧梯度主导，温度/PDE 收敛变慢 |
| `w_bc` | 外边界条件 | 右壁温度、气凝胶热阻或绝热泄漏不合格 | 网络过度拟合外边界而忽略内部热源 |
| `w_eng` | 设备和全局积分守恒 | `eng`、`bal` 长期偏高，排热幅值不足 | 只保证总量，局部 PDE 和热流分配可能错误 |

推荐先按 2 倍左右做粗调，并比较完整曲线。例如 `w_if_q: 10→20`，而不是一次
提高到 $10^3$。某项的打印值大不一定表示它权重不足，因为各分项归一化方式不同；
应结合该项的物理验收量判断。

特别注意以下两种假收敛：

- `eng` 很低但 PDE/界面仍高：总排热碰巧正确，局部温度场不可信；
- `ifT` 很低但 `ifq` 很高：界面温度接上了，但能量没有正确跨材料传递。

#### 8.9.7 L-BFGS 参数

| 参数 | 增大后的作用 | 风险与建议 |
|---|---|---|
| `lbfgs_steps` | 增加外层精修次数 | 分项仍下降时可增加；长期不变时继续增加收益有限 |
| `lbfgs_max_iter` | 每次 step 内允许更多线搜索/拟牛顿迭代 | 单步时间和 closure 次数增加；显存/时间不足时降低 |
| `lbfgs_history` | 使用更多历史曲率信息 | 占用更多内存；网络较大时优先控制该值 |
| `lbfgs_resample` | 控制多少外层 step 后换配点 | 值大更利于稳定曲率估计，值小提高空间覆盖但会扰动目标 |

如果 L-BFGS 损失明显跳动，先延长固定配点区间，即增大 `lbfgs_resample` 或设为
0 做一次固定点诊断；不要首先增加 `max_iter`。如果固定点损失下降而独立验证误差
变差，说明对该批配点过拟合，应恢复周期重采样或增加验证导向的局部点。

#### 8.9.8 Fourier 与温度初值

`fourier_sigma=None` 表示使用 `networks.py` 的分域默认值；当前各域默认关闭。
设置全局 `fourier_sigma=0` 也会明确关闭 Fourier 特征。普通 MLP 无法表达经过
验证的局部陡峭梯度时，才尝试从较小 `fourier_sigma` 开始，并保持
`fourier_dim=64` 做第一组对照。

若增加 `fourier_sigma` 后 PDE 二阶导数残差剧烈振荡，应减小频率尺度，而不是只
增加 PDE 权重。若多个随机种子差异很大，可增加 `fourier_dim`，但会增加首层参数
量、显存和计算时间。

正值 `theta_init` 是所有分域的统一初始无量纲温度。当前实现中默认0不会构造
冷端平场，而是跳过重置并保留 Xavier 随机初值。已有相近工况时优先用 checkpoint
热启动；手工设置正的 `theta_init` 只能改变初始常温水平，不能提供正确的温度
梯度和界面热流。

#### 8.9.9 记录、保存与可复现性

| 参数 | 调节建议 |
|---|---|
| `eval_every` | 调试时减小以观察异常；长训练时增大以降低监控开销 |
| `save_every` | 应小于可接受的最大重算轮数；过小会产生大量 checkpoint |
| `plot_every` | 实时温度场窗口刷新间隔；默认每100个累计训练步更新 |
| `plot_resolution` | 温度图每个方向的网格数；增大会提高图像细节和推理开销 |
| `seed` | 基线固定；稳定性研究时改变并保存每次结果 |

`live_plot=True` 时，训练会打开一个非阻塞 Matplotlib 窗口，将各材料分支拼成
完整温度场，并叠加两个方块和圆设备边界。窗口更新不会参与反向传播，每次更新
还会覆盖保存 `<outdir>/temperature_live.png`。远程服务器或无图形界面运行时可
使用 `--no-plot`，训练和数值结果不受影响。当前绘图异常没有被训练循环捕获；在
Windows 的 TkAgg 后端出现窗口句柄或 `PyCapsule_New` 错误时会中断进程，因此长
训练建议使用 `--no-plot`。绘图调用位于同一 epoch 的 checkpoint 保存之前，若
窗口刷新报错，该轮通常不会保存，但此前的 `latest.pt` 不受影响。

训练会保留 `epoch_XXXXXX.pt` 并刷新 `latest.pt`。开始新的消融实验时应使用新的
`ckptdir/outdir` 或能区分配置的案例标签，避免不同参数实验共享日志和历史文件。
`loss_log.csv`、命令行参数、Git 版本和 `config.TRAIN` 应共同归档，单独保存网络
权重不足以完整复现实验。

#### 8.9.10 现象诊断速查表

| 观察到的现象 | 优先检查 | 推荐动作 |
|---|---|---|
| `loss`、所有分项共同振荡 | 学习率、输入/输出尺度 | 降低 `lr`，检查是否出现非有限梯度 |
| 初期长期停在近常温平场 | 功率课程、设备 PDE、能量信号 | 检查 `power_scale` 同步，适当提高 `w_pde_dev/w_eng` |
| `pde` 高、其他项低 | 域内点和 PDE 权重 | 找出具体高误差域，增加该域点数后再调 `w_pde` |
| `ifT` 高、`ifq` 低 | 温度连续约束 | 增加界面点或小幅提高 `w_if_T` |
| `ifT` 低、`ifq` 高 | 通量尺度、导热系数、方向 | 先核对法向和量纲，再调 `w_if_q` |
| `bc` 高 | 边界采样和边界类型 | 核对 Dirichlet/Robin 配置，增加 `n_bnd` 或 `w_bc` |
| `eng` 高且 `bal` 高 | 总热流幅值 | 放慢功率爬坡，检查 `w_eng` 和边界排热方向 |
| `eng` 低但 `bal` 高 | 设备预算与外边界预算不一致 | 检查界面传热、边界条件和监控积分，不盲目调权 |
| `bal` 低但 FEM 误差高 | 局部场未收敛 | 增加局部点，检查 PDE、界面和最高温度；总量正确不代表场正确 |
| Adam 平台、各项仍缓慢下降 | 一阶优化进入刚性谷 | 启用或增加 L-BFGS 精修 |
| L-BFGS 固定点下降、验证变差 | 固定配点过拟合 | 缩短重采样间隔或增加独立/局部验证点 |
| 设备温度出现明显非物理值 | 符号、尺度、功率、边界 | 先查物理实现，不用权重把异常压下去 |

#### 8.9.11 停止训练与验收

可以停止增加训练预算的条件不是“总损失看起来很小”，而是：

1. 多个评估周期内各分项和监控量已经稳定；
2. 全功率下左右边界总能量不平衡达到第04文档规定的门槛，当前建议小于1%；
3. 上下绝热泄漏、右边界条件和气凝胶热阻检查合格；
4. 当前 `validate.py` 自动检查的两个设备安装面误差合格；全部材料界面还需增加
   独立验证器后再作为自动硬门槛；
5. 三个设备最高温度在加密监控网格上稳定；
6. `validate.py` 的逐域及全局 FEM 相对误差满足复合精度要求，并明确记录当前
   尚未覆盖的独立 PDE/全界面/全边界残差；
7. 改变随机种子或增加验证点后，结论没有明显改变。

如果某项不合格，应根据速查表定位物理环节。不得通过任意放大某个权重让加权总
损失变小，也不得只凭能量平衡或最高温度单项宣布 PINN 已经收敛。

---

## 9. 设计总结

当前 `MLP` 的设计原则可以概括为：

1. 使用平滑的 `Tanh` 网络满足 PINN 对二阶自动微分的要求；
2. 输出层保持线性，不人为截断可能超过参考范围的温度；
3. 默认采用简单 MLP，优先保证纯导热基线的训练稳定性；
4. 通过固定随机 Fourier 特征预留局部陡峭温度场的增强能力；
5. 使用 Xavier 初始化降低 `Tanh` 初始饱和和导数消失风险；
6. 将基础函数逼近与材料域、几何和物理约束分离，保持代码职责清晰。

一句话概括：`MLP` 是面向二阶导热方程的平滑函数逼近器，
`TemperatureField` 将多个 `MLP` 组织成分域温度场，而各类物理损失负责把这些
独立温度分支重新耦合成完整的多材料共轭导热问题。
