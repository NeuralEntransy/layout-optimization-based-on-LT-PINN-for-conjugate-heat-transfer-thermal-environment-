# SDF-XPINN 架构概述

> 本文说明 `src_sdf_xpinn/` 中五域多材料导热模型的网络组织方式，重点解释
> SDF/CSG 几何、XPINN 风格分域温度网络、显式界面耦合、SDF 边界层采样和
> PDE 残差自适应随机加密（RAR）之间的关系。
>
> 该目录是独立实验实现，不覆盖原有 `src/`。当前版本号为
> `sdf_xpinn_v1`，用于固定布局下的 Milestone 1 纯导热正问题。

---

## 1. 方法定位

当前实现结合了两类方法思想：

1. **LT-PINN 风格的显式可微几何**：墙体、设备和空气域由 SDF/CSG 及少量
   几何参数描述；
2. **XPINN/cPINN 风格的分域求解**：每个物理区域使用独立温度网络，再通过
   温度连续和守恒热流连续损失耦合。

因此更准确的名称是：

> 基于显式 SDF 几何的五域 XPINN/cPINN 共轭导热模型。

它不是 LT-PINN 原论文算例的逐项复现，也不是包含子域并行优化器的完整通用
XPINN 框架。当前所有分支仍由一个总目标联合反向传播。

总体结构为：

```text
显式几何参数 x1、x2
        │
        ▼
SDF / CSG 几何
        │
        ├── wall SDF：外框减空腔
        ├── dev1 SDF：移动方形设备
        ├── dev2 SDF：移动方形设备
        ├── dev3 SDF：固定圆形设备
        └── air SDF：空腔减三个设备
        │
        ▼
五个独立温度分支
        │
        ├── wall ── MLP ── θ_wall
        ├── air  ── MLP ── θ_air
        ├── dev1 ── MLP ── θ_dev1
        ├── dev2 ── MLP ── θ_dev2
        └── dev3 ── MLP ── θ_dev3
        │
        ▼
PDE + 界面 + 外边界 + 积分能量损失
```

---

## 2. 为什么从九分支改为五分支

旧版 `src/` 把墙框拆分为：

```text
wall_l / wall_r / wall_b / wall_t
```

这样每个墙分支都是输入条件较好的薄矩形，但四段墙之间必须增加四个墙角人工
接口。墙角接口需要额外满足：

$$
T_a=T_b,
\qquad
k_{Al}\partial_nT_a=k_{Al}\partial_nT_b.
$$

新架构使用一个 `wall` 网络表示整个连续铝框，因此：

- 四个墙角不再是数值接口；
- 删除四组墙角 `ifT/ifq`；
- 左右、上下外边界都由同一个墙体温度函数提供；
- 墙体整体热流可以直接积分；
- 墙体几何由一个非凸 SDF 定义。

设备仍使用独立分支，原因包括：

- 每台设备具有独立体热源；
- 需要分别监控温度和输出功率；
- 设备—墙与设备—空气接触面需要独立验收；
- 后续可能释放设备位置参数。

最终区域为：

| 区域 | 材料 | 控制方程 | 网络 |
|---|---|---|---|
| `wall` | 铝 | Laplace | `MLP_wall` |
| `air` | 静止空气 | Laplace | `MLP_air` |
| `dev1` | 铝 | Poisson | `MLP_dev1` |
| `dev2` | 铝 | Poisson | `MLP_dev2` |
| `dev3` | 铝 | Poisson | `MLP_dev3` |

---

## 3. SDF 与 CSG 几何

### 3.1 统一符号约定

所有 SDF 使用：

$$
F(\mathbf x)<0 \quad \text{区域内部},
$$

$$
F(\mathbf x)=0 \quad \text{区域边界},
$$

$$
F(\mathbf x)>0 \quad \text{区域外部}.
$$

矩形使用轴对齐矩形的精确欧氏 SDF，圆形设备使用：

$$
F_{dev3}(x,y)
=\sqrt{(x-x_c)^2+(y-y_c)^2}-R_3.
$$

### 3.2 墙体 SDF

外框为单位正方形，内腔为：

$$
[w,1-w]\times[w,1-w],
\qquad w=0.01\ \mathrm m.
$$

墙体等于外框减去内腔，对应 CSG 差集：

$$
\boxed{
F_{wall}
=\max(F_{outer},-F_{cavity})
}.
$$

该零水平集同时包含墙体外边界和墙—空腔内边界。墙体是一个连续非凸框架，
不再拆分为四个矩形网络。

### 3.3 设备 SDF

dev1、dev2 为轴对齐矩形：

$$
F_{dev1}=SDF_{box}(x_1,D_1),
$$

$$
F_{dev2}=SDF_{box}(x_2,D_2).
$$

其中 $x_1,x_2$ 是设备水平中心。`DesignVars` 使用 sigmoid 将它们限制在可行
区间内，因此 SDF 对位置参数保持自动微分关系。

dev3 为固定圆形设备：

$$
F_{dev3}=SDF_{circle}(\mathbf c_3,R_3).
$$

### 3.4 空气域 SDF

设备并集的隐式场为：

$$
F_{devices}
=\min(F_{dev1},F_{dev2},F_{dev3}).
$$

空气等于内腔减去设备并集：

$$
\boxed{
F_{air}
=\max(F_{cavity},-F_{devices})
}.
$$

因此 `air` 自动表示包含矩形孔洞和圆形孔洞的复杂区域。设备移动时，空气域
边界随 $x_1,x_2$ 同步变化。

### 3.5 SDF 当前参与哪些计算

SDF 当前用于：

- `domain_sdf()`：取得指定区域隐式场；
- `mask_domain()`：判断点是否位于某域；
- `label_points()`：拼装完整温度场；
- 空气域拒绝采样；
- SDF 内侧边界层加密；
- 几何测试和接口零水平集检查；
- `smooth_indicator()`：为后续 LT-PINN 风格软可见性预留接口。

当前训练仍是固定布局。虽然 SDF 对 $x_1,x_2$ 可微，但随机采样器会将位置转换
为 Python 浮点数，因此还不能通过采样路径对位置联合反向传播。释放几何变量前
必须将移动边界点改成重参数化采样。

---

## 4. 五分支温度网络

### 4.1 基础 MLP

每个区域使用独立的平滑 `Tanh MLP`：

```text
物理坐标 (x,y)
      ↓
分域输入归一化
      ↓
可选随机 Fourier 特征
      ↓
Linear + Tanh × depth
      ↓
线性输出层
      ↓
分域输出尺度
      ↓
无量纲温度 θ_d
```

默认网络参数为：

| 参数 | 默认值 |
|---|---:|
| 输入维数 | 2 |
| 输出维数 | 1 |
| `width` | 96 |
| `depth` | 5 |
| `fourier_sigma` | 0 |
| `fourier_dim` | 64 |

`Tanh` 具有平滑的一阶和二阶导数，适合通过自动微分计算 Laplace/Poisson
残差。输出层保持线性，不限制温度范围。

### 4.2 为什么不使用单一全局温度网络

不同材料界面满足：

$$
T_a=T_b,
\qquad
k_a\frac{\partial T_a}{\partial n}
=k_b\frac{\partial T_b}{\partial n}.
$$

由于：

$$
\frac{k_{Al}}{k_f}\approx6423,
$$

铝—空气界面两侧的法向温度梯度可能相差数千倍。一个全局光滑网络倾向于给出
连续梯度，不适合表示这种物理跳变。

也不建议直接用平滑 SDF 指示函数构造：

$$
k(\mathbf x)
=k_f+(k_{Al}-k_f)\delta(F(\mathbf x)),
$$

再用单一强形式 PINN 求解。平滑指示函数会产生人工过渡层，且 $\nabla k$ 在
窄界面附近可能非常大，使二阶自动微分更加刚性。

### 4.3 分域输入归一化

当前归一化为：

$$
\mathbf z_d
=\frac{\mathbf x-\mathbf c_d}{\mathbf s_d}.
$$

| 区域 | `center` | `scale` |
|---|---:|---:|
| `wall` | $(0.5,0.5)$ | $(0.5,0.5)$ |
| `air` | $(0.5,0.5)$ | $(0.49,0.49)$ |
| `dev1` | $(0.5,0.11)$ | $(0.5,0.1)$ |
| `dev2` | $(0.5,0.815)$ | $(0.5,0.175)$ |
| `dev3` | $(0.5,0.4)$ | $(0.1,0.1)$ |

dev1/dev2 的水平中心固定为 0.5，使网络保留设备的全局水平位置信息。dev3 使用
等比局部尺度，使圆形设备在归一化空间中仍保持圆形。

墙体是非凸薄框，因此其全局归一化不再像旧版四个薄矩形分支那样把厚度映射到
完整 $[-1,1]$。这是减少人工接口与改善薄域输入条件之间的折中，必须通过训练
结果和 FEM 对比验证。

### 4.4 分域输出尺度

网络输出为：

$$
\theta_d(\mathbf x)
=s_{\theta,d}NN_d(\mathbf z_d).
$$

当前输出尺度为：

| 区域 | 输出尺度 |
|---|---:|
| `wall` | 2.0 |
| `air` | 3.0 |
| `dev1` | 2.0 |
| `dev2` | 2.5 |
| `dev3` | 4.0 |

输出尺度是优化预条件，不是温度上限。它会同时影响温度、梯度、PDE 残差和
界面热流相对于网络权重的梯度尺度。

### 4.5 统一温度初始化

所有分支最后一层权重初始化为零，偏置设置为：

$$
b_d=\frac{\theta_{init}}{s_{\theta,d}}.
$$

当前默认 $\theta_{init}=0$，因此所有分支从冷端参考温度开始：

$$
T=T_c+\Delta T\theta=T_c.
$$

这样初始界面温差和热流接近零，设备 Poisson 源项和能量损失再逐步建立温度
曲率及排热路径。

---

## 5. 控制方程损失

### 5.1 无热源区域

墙体和空气域满足：

$$
\nabla^2\theta=0.
$$

材料导热系数没有显式出现在域内 Laplace 方程中，因为每个常导热系数区域的
控制方程已经除以本域 $k_d$。

### 5.2 发热设备

每台设备满足：

$$
\nabla^2\theta+S_kp=0,
$$

其中 $p$ 为 `power_scale`，源项为：

$$
S_k
=\frac{\dot q'''_kL_{ref}^2}{k_{Al}\Delta T}.
$$

设备源项使用正号，是因为稳态导热方程写成：

$$
\nabla\cdot(k\nabla T)+\dot q'''=0.
$$

设备 PDE 通过 `w_pde_dev` 获得附加强调。当前默认：

```text
w_pde = 1.5
w_pde_dev = 600
```

`w_pde_dev` 已包含在打印的设备 PDE 聚合值中。

---

## 6. 显式界面耦合

### 6.1 物理条件

每个界面使用同一组物理点和同一导数方向，约束：

$$
r_T=\theta_a-\theta_b,
$$

$$
r_q
=\left(
k_a\partial_\xi\theta_a
-k_b\partial_\xi\theta_b
\right)
\frac{\Delta T}{L_{ref}Q_{IF}}.
$$

损失为：

$$
L_{if,T}=\operatorname{mean}(r_T^2),
$$

$$
L_{if,q}=\operatorname{mean}(r_q^2).
$$

当前外层权重为：

```text
w_if_T = 20
w_if_q = 20
```

当前 v1 使用普通平方罚损失，尚未实现增广拉格朗日乘子更新。

### 6.2 当前接口拓扑

墙—空气接口：

```text
wall_air_left
wall_air_right
wall_air_bottom
wall_air_top
```

设备接口：

```text
dev1_wall
dev1_air_left
dev1_air_right
dev1_air_top

dev2_wall
dev2_air_left
dev2_air_right
dev2_air_bottom

dev3_air
```

共 13 个显式接口。几何测试会检查每个接口采样点同时位于相邻两个区域的零水平
集上。

### 6.3 相同材料接口为什么仍保留

dev1/dev2 与墙体都是铝，但当前仍使用独立 MLP，因此设备—墙安装面在数值上
仍是分域接口。对于理想同材料接触，需要同时满足温度和梯度连续；因为两侧
$k$ 相同，热流连续等价于法向温度梯度连续。

如果未来确认设备与墙是一体成型，可以考虑将 dev1/dev2 与 wall 合并为统一
铝域，并使用设备 SDF 仅打开体热源。但当前五域设计保留独立设备温度和功率
监控，因此继续显式约束安装面。

---

## 7. 外边界条件

### 7.1 左侧气凝胶等效热阻

气凝胶没有作为第六个显式网络求解，而是退化为一维 Robin 热阻：

$$
q''_{left}
=h_{TBL}(T-T_{cold}),
\qquad
h_{TBL}=\frac{k_{TBL}}{L_{TBL}}.
$$

左侧外法向为 $(-1,0)$，代码比较墙体导热流和气凝胶等效排热量。

### 7.2 右侧定温边界

当前右边界使用：

$$
\theta_{wall}=\theta_{inf}=1.
$$

新入口暂未提供旧版 `right-bc=robin` 切换选项。

### 7.3 上下绝热边界

上下边界满足：

$$
-k_{Al}\nabla T\cdot\mathbf n=0.
$$

当前边界损失中，上下绝热项还会额外乘 `w_adiabatic=8`，再统一乘外层
`w_bc=75`。

---

## 8. 积分能量约束

### 8.1 设备双侧预算

对每台设备，分别积分设备自身网络和接收侧网络的法向热流：

$$
Q_k^{dev}
=\int_{\partial\Omega_k}
-k_{Al}\nabla T_{dev}\cdot\mathbf n_k\,dA,
$$

$$
Q_k^{recv}
=\int_{\partial\Omega_k}
-k_{recv}\nabla T_{recv}\cdot\mathbf n_k\,dA.
$$

两侧都约束为设备额定功率：

$$
Q_k^{dev}=P_k,
\qquad
Q_k^{recv}=P_k.
$$

每个设备面还加入双侧积分通量差：

$$
Q_{k,j}^{dev}=Q_{k,j}^{recv}.
$$

这不能替代逐点 `InterfaceLoss`，但能向各接收分支提供低频总量信号。

### 8.2 左右排热预算与上下独立绝热预算

外边界热流均按计算域外法向取正。物理上只有左、右边界允许排出设备产生的总热量，
因此左右预算定义为：

$$
r_{LR}
=\frac{Q_L+Q_R-P_{tot}}{P_{tot}},
\qquad
P_{tot}=P_1+P_2+P_3.
$$

代码不规定 $Q_L$ 和 $Q_R$ 各自承担多少，只约束二者之和。两侧分配由左侧气凝胶
热阻、右侧定温条件以及内部温度场共同决定。

上下边界分别绝热，不能合并成一个净泄漏条件，因此定义：

$$
r_T=\frac{Q_T}{P_{tot}},
\qquad
r_B=\frac{Q_B}{P_{tot}}.
$$

外边界积分损失为：

$$
L_{eng,outer}
=r_{LR}^2+r_T^2+r_B^2.
$$

此前使用的四侧合并预算
$Q_L+Q_R+Q_T+Q_B=P_{tot}$ 已删除，因为它允许非物理的上下漏热与左右排热误差
相互抵消。这里的 $r_T^2+r_B^2$ 是整侧积分约束；第 7.3 节的边界损失仍在采样点
上逐点约束法向热流为零，两者分别控制局部泄漏和整侧净泄漏。

当前 SDF-XPINN 尚未加入旧版针对整个空气域和整个墙体域的独立积分预算，也没有
把上下边界进一步拆成多个线段预算。

### 8.3 固定确定性求积

能量积分使用 `n_energy=1024` 个确定性中点，与随机训练界面点分离。这样可以
降低蒙特卡洛噪声，使能量目标在相邻 epoch 之间更稳定。

---

## 9. 域内采样策略

每个 epoch 的域内 PDE 点由三部分组成：

```text
均匀随机域内点
    +
SDF 内侧边界层点
    +
PDE 残差加权 RAR 点
```

### 9.1 均匀域内点

当前基础点数为：

| 区域 | 点数/epoch |
|---|---:|
| `wall` | 12000 |
| `air` | 15000 |
| `dev1` | 3000 |
| `dev2` | 14000 |
| `dev3` | 2500 |

墙体通过四个互不重叠矩形条带按面积加权采样，但所有点都训练同一个 `wall`
网络。空气域在内腔包围盒中生成候选点，再用 `F_air<=0` 拒绝设备内部点。

圆设备采用：

$$
r=R\sqrt{u},
\qquad
\varphi=2\pi v,
$$

保证按面积均匀采样。

### 9.2 SDF 内侧边界层采样

边界层点满足：

$$
-\delta_d\le F_d(\mathbf x)\le0.
$$

当前附加点数为：

| 区域 | 点数 | 宽度 [m] |
|---|---:|---:|
| `wall` | 4000 | 0.002 |
| `air` | 6000 | 0.01 |
| `dev1` | 1000 | 0.01 |
| `dev2` | 3000 | 0.01 |
| `dev3` | 500 | 0.01 |

这种方法不需要为每种矩形面单独编写局部采样器，复杂空气边界和圆周也能使用
同一 SDF 距离规则。

---

## 10. RAR 残差自适应随机加密

### 10.1 目的

均匀采样和固定边界层加密依赖预先判断误差位置。RAR 根据当前网络的逐点 PDE
残差，自动把更多点分配到尚未满足控制方程的位置。

### 10.2 候选与抽样

对区域 $d$，先生成 $M$ 个均匀候选点，再计算：

$$
s_i=r_i^2,
$$

其中 $r_i$ 是未乘训练权重的强形式 PDE 残差。抽样概率为：

$$
p_i=(1-\epsilon)
\frac{(s_i/\bar s)^\alpha}
{\sum_j(s_j/\bar s)^\alpha}
+\frac{\epsilon}{M}.
$$

当前：

```text
candidate_factor = 4
rar_power = 1
rar_uniform_mix = 0.05
```

5% 的均匀混合项维持探索性。RAR 采用无放回随机抽样，每个 epoch 重新生成，
并追加到基础点和 SDF 边界层点中。

### 10.3 当前 RAR 数量

| 区域 | RAR 点数 | 候选点数 |
|---|---:|---:|
| `wall` | 2000 | 8000 |
| `air` | 3000 | 12000 |
| `dev1` | 500 | 2000 |
| `dev2` | 1000 | 4000 |
| `dev3` | 500 | 2000 |

### 10.4 RAR 日志

训练日志记录：

```text
rar_<domain>_candidate_mean
rar_<domain>_selected_mean
```

如果 RAR 正在偏向困难区域，通常有：

$$
\overline{r^2}_{selected}
>\overline{r^2}_{candidate}.
$$

初始统一常温场中，设备源项残差在设备内部近似常数，因此设备 RAR 的候选均值
和入选均值可能相同。温度曲率形成后才会产生空间偏置，这是正常现象。

RAR 评分采用独立 microbatch 并 `detach()`，不会向模型参数写入正式训练梯度。
命令行 `--no-rar` 可以关闭 RAR，供消融实验使用。

---

## 11. 梯度积累与单步更新

二阶 PDE 自动微分占用显存较高。当前一个 epoch 内采用梯度积累：

```text
optimizer.zero_grad()
        ↓
各域 PDE microbatch backward
        ↓
各界面 microbatch backward
        ↓
四组外边界 backward
        ↓
积分能量 backward
        ↓
optimizer.step()
```

当前 microbatch 参数为：

```text
pde_microbatch = 2000
interface_microbatch = 512
rar_score_microbatch = 2000
```

PDE 和界面 microbatch 损失乘：

$$
\frac{N_{chunk}}{N_{total}},
$$

保证拆分后梯度等价于完整批次平均损失。每个 epoch 只有一次
`optimizer.step()`，不是把每个子域依次当作独立优化步骤。

---

## 12. 总损失

当前训练目标为：

$$
L
=w_{pde}L_{pde}
+w_{if,T}L_{if,T}
+w_{if,q}L_{if,q}
+w_{bc}L_{bc}
+w_{eng}L_{eng}.
$$

默认权重为：

| 损失 | 权重 |
|---|---:|
| 域内 PDE | `w_pde=1.5` |
| 设备 PDE 内部附加 | `w_pde_dev=600` |
| 界面温度 | `w_if_T=20` |
| 界面热流 | `w_if_q=20` |
| 外边界 | `w_bc=75` |
| 上下绝热内部附加 | `w_adiabatic=8` |
| 积分能量 | `w_eng=100` |

各聚合分项的当前精确定义如下。

### 12.1 PDE 聚合

令 $D=\{wall,air,dev1,dev2,dev3\}$，则：

$$
L_{pde}
=\sum_{d\in D}\alpha_d
\operatorname{mean}_{\mathbf x\in X_d}
\left[r_{pde,d}(\mathbf x)^2\right],
$$

其中设备域的 $\alpha_d=w_{pde,dev}=600$，墙体和空气的 $\alpha_d=1$。每个域先
独立取均值，再把五个域相加；因此增加某个域的采样点主要降低估计方差，不会因为
点数更多就自动提高该域在总损失中的系数。外层再乘 `w_pde=1.5`。

### 12.2 界面聚合

对 13 个显式界面集合 $I$：

$$
L_{if,T}=\sum_{j\in I}\operatorname{mean}(r_{T,j}^2),
\qquad
L_{if,q}=\sum_{j\in I}\operatorname{mean}(r_{q,j}^2).
$$

每个界面也是先独立取均值再求和，所以界面点数控制离散精度，不直接改变该界面的
损失系数。两项分别乘 `w_if_T=20` 和 `w_if_q=20`。

### 12.3 外边界聚合

令 $L_L,L_R,L_T,L_B$ 分别为 `boundary_one()` 返回的四侧逐点均方损失，则：

$$
L_{bc}=L_L+L_R+w_{adiabatic}(L_T+L_B),
\qquad w_{adiabatic}=8.
$$

其中左侧为 Robin 热阻残差，右侧为 Dirichlet 温度残差，上下为逐点绝热热流残差；
整个 $L_{bc}$ 再乘 `w_bc=75`。

### 12.4 能量聚合

设备双侧预算为：

$$
L_{eng,device}
=\sum_{k=1}^{3}
\left[
\left(\frac{Q_k^{dev}-P_k}{P_k}\right)^2
+\left(\frac{Q_k^{recv}-P_k}{P_k}\right)^2
\right].
$$

设备各接触面的双侧积分通量差为：

$$
L_{eng,face}
=\sum_k\sum_{j\in\partial\Omega_k}
\left(\frac{Q_{k,j}^{dev}-Q_{k,j}^{recv}}{P_k}\right)^2.
$$

结合第 8.2 节的外边界项，当前能量损失为：

$$
L_{eng}
=L_{eng,device}+L_{eng,face}
+r_{LR}^2+r_T^2+r_B^2.
$$

这四类能量分量当前没有额外的内层权重，整体统一乘 `w_eng=100`。

打印的 `pde/ifT/ifq/bc/eng` 是上述聚合分项，其中设备 PDE 和逐点绝热内部附加
权重已经分别包含在 `pde` 和 `bc` 中；外层权重仍需乘回才能复原 `loss`。RAR
候选评分只用于选点并已 `detach()`，不单独进入总损失或 `backward()`。

---

## 13. 训练入口与 checkpoint

### 13.1 当前训练流程

`main.py` 的流程为：

```text
设置随机种子和 device
        ↓
建立五域 TemperatureField
        ↓
固定 center 布局的 x1、x2
        ↓
Adam + 每轮动态采样 + RAR
        ↓
定期记录 loss、热量、温度和 RAR 指标
        ↓
保存 epoch_xxxxxx.pt 与 latest.pt
        ↓
生成 temperature.png 和 run_report.json
```

当前新入口尚未实现：

- 学习率调度器选择；
- 功率 continuation；
- `--init-from` 部分权重热启动；
- L-BFGS 阶段；
- 增广拉格朗日界面约束；
- 可训练位置联合优化；
- 旧版能量损失中的空气域、墙体域和分段绝热积分预算。当前已经包含左右总排热及
  上、下整侧独立绝热积分预算。

不能把旧版 `src/main.py` 已实现的功能默认认为新入口也已具备。

### 13.2 checkpoint 内容

checkpoint 保存：

```text
epoch
field state_dict
Adam optimizer state_dict
命令行参数 args
case version
```

使用 `--resume` 时，会从当前 `ckptdir/latest.pt` 恢复网络和 Adam 状态，并从：

$$
epoch_{start}=epoch_{checkpoint}+1
$$

继续到命令行指定的累计 `--epochs`。

由于墙体网络结构已经变化，旧 `src/` 的 checkpoint 不能直接完整加载到五域
模型。当前新入口也没有实现旧分支的选择性迁移。

---

## 14. 日志字段

当前 `loss_log.csv` 记录：

| 字段 | 含义 |
|---|---|
| `epoch` | 累计 Adam epoch |
| `loss` | 当前加权总损失估计 |
| `pde` | 五域 PDE 聚合值 |
| `ifT` | 13 个界面温度连续损失之和 |
| `ifq` | 13 个界面热流连续损失之和 |
| `bc` | 外边界聚合损失，含绝热内部附加权重 |
| `eng` | 设备双侧、逐面、左右总排热及上下独立绝热积分损失 |
| `lr` | 当前 Adam 学习率 |
| `Q_left/right/top/bottom` | 四侧外边界带符号排热量 [W] |
| `T1_C/T2_C/T3_C` | 三台设备网格最高温度 [°C] |
| `rar_*_candidate_mean` | 对应域 RAR 候选平均残差平方 |
| `rar_*_selected_mean` | 对应域 RAR 入选平均残差平方 |

除聚合日志外，当前入口还会在每次评估时写入：

- `physics_log.csv`：五个域的 PDE、13 个界面的逐项 `ifT/ifq`、左/右/上/下
  四组原始边界损失；
- `energy_log.csv`：三台设备的固体侧/接收侧相对残差、每个设备面的两侧积分
  热量及差值、四侧外边界热量、左右排热残差 `eng_lr`、上下整侧残差
  `eng_adiabatic_top/bottom`，以及
  `eng_loss_device/face/lr/adiabatic/outer/total`。

终端只打印最大 PDE、最大温度/热流接口误差和设备/外边界能量摘要，完整字段保留
在两个 CSV 中。需要强调，这些日志只拆分当前实际损失；尚未加入的空气域、墙体域
及分段绝热积分预算不会出现在 `eng_loss_total` 中。

---

## 15. 验证与测试

### 15.1 几何测试

`test_geometry.py` 检查：

- 代表性点被正确标记为五个区域；
- dev1 SDF 对 $x_1$ 存在非零梯度；
- 13 个界面采样点同时位于相邻两域的零水平集。

### 15.2 RAR 测试

`test_rar.py` 检查：

- RAR 返回点全部位于指定域内；
- 入选点平均 PDE 残差高于候选池平均值；
- RAR 评分不会写入网络参数 `.grad`。

### 15.3 已完成的运行检查

当前已通过：

- Python 静态编译；
- CPU SDF/接口测试；
- CUDA 二阶导数和 backward；
- checkpoint 保存与 `--resume`；
- 无交互温度图输出。

这些检查只证明代码链可以运行，不代表物理解已经收敛。正式验收仍需：

- 逐域 PDE；
- 逐界面温度和热流；
- 设备功率；
- QL/QR/QT/QB；
- 加密网格最高温度；
- 独立 FEM 场误差。

---

## 16. 推荐训练命令

从头训练 20000 个 Adam epoch，并默认启用 RAR：

```powershell
Set-Location -LiteralPath 'F:\LT-PINN'

& 'D:\ANACONDA\envs\Pytorch\python.exe' .\src_sdf_xpinn\main.py `
  --layout center `
  --epochs 20000 `
  --lr 2e-7 `
  --device cuda:0 `
  --width 96 `
  --depth 5 `
  --power-scale 1 `
  --ckptdir '.\checkpoint_m1\sdf_xpinn_v1_rar' `
  --outdir '.\results\sdf_xpinn_v1_rar' `
  --eval-every 100 `
  --save-every 1000 `
  --no-plot
```

进行无 RAR 消融时增加：

```text
--no-rar
```

从当前 `latest.pt` 继续训练到累计 30000 epoch：

```powershell
& 'D:\ANACONDA\envs\Pytorch\python.exe' .\src_sdf_xpinn\main.py `
  --layout center `
  --epochs 30000 `
  --lr 2e-7 `
  --device cuda:0 `
  --width 96 `
  --depth 5 `
  --power-scale 1 `
  --ckptdir '.\checkpoint_m1\sdf_xpinn_v1_rar' `
  --outdir '.\results\sdf_xpinn_v1_rar' `
  --eval-every 100 `
  --save-every 1000 `
  --no-plot `
  --resume
```

`--epochs` 表示目标累计 epoch，不是额外训练步数。

---

## 17. 当前优势、风险与下一步

### 17.1 当前优势

1. 墙体成为真实的连续非凸域，删除四个数值墙角接口；
2. 五个分支仍允许异质材料界面的温度梯度跳变；
3. SDF 统一服务于区域判定、复杂空气域和局部加密；
4. RAR 能自动寻找域内 PDE 难点；
5. 梯度积累允许较高密度采样而不改变一步更新语义；
6. 几何表达为后续位置优化保留了可微基础。

### 17.2 主要风险

1. 单一 `wall` 网络的全局输入尺度可能不利于 1 cm 薄框；
2. 新入口的能量预算和日志比旧版简化，诊断能力暂时下降；
3. RAR 增加候选点二阶导数开销；
4. 固定 penalty 仍可能造成 PDE 与界面约束竞争；
5. 设备位置目前没有通过采样路径联合反传；
6. 新架构无法直接继承旧墙体 checkpoint。

### 17.3 推荐下一步

建议按以下顺序推进：

1. 完成五域 SDF-XPINN 从头训练基线；
2. 根据 `physics_log` 和 `energy_log` 定位逐域、逐界面误差；
3. 与旧四墙模型比较 wall PDE、安装面误差和 QL/QR；
4. 比较启用/关闭 RAR 的精度和单步成本；
5. 根据训练诊断决定是否继续补充空气域、墙体域和分段绝热积分预算；
6. 若安装面固定罚损失仍停滞，再试逐界面权重或增广拉格朗日；
7. 基线通过后，再实现固定点 L-BFGS；
8. 最后将移动界面采样重参数化，释放 $x_1,x_2$ 联合优化。

---

## 18. 设计总结

当前架构的设计原则可以概括为：

1. 用显式 SDF/CSG 描述墙、设备和复杂空气域；
2. 用五个独立 `Tanh MLP` 表示各区域温度；
3. 用温度连续和守恒法向热流连接分域网络；
4. 用均匀点保持全域覆盖；
5. 用 SDF 边界层点加强已知界面附近的 PDE；
6. 用 RAR 自动发现未知高残差区域；
7. 用积分能量约束校正设备功率和外边界排热幅值；
8. 用 microbatch 梯度积累控制二阶自动微分显存。

一句话概括：

> SDF 提供清晰、可微的物理几何，XPINN 分支提供跨材料梯度跳变的表达能力，
> 显式界面损失重新连接各温度场，SDF 边界层与 RAR 则共同提高复杂区域中的
> PDE 配点效率。
