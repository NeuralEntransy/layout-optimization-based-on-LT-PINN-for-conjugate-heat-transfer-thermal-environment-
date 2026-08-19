# Milestone 0 / 1 复现结果

> 对应复现方案 `04_复现方案.md` 第 6 节的 Milestone 0（确定性参考解）与 Milestone 1（固体热桥 + 纯导热）。
> 运行环境：WSL2；FEM 使用 conda env `fenicsx-env`（dolfinx 0.10.0 + gmsh），PINN 使用 `fortran_Torch_cp3.9_cuda12x`（torch 2.8.0+cu128，RTX 3080）。

---

## 1. Milestone 0：FEniCSx 确定性参考解

### 1.1 实现

脚本：`fem/milestone0_reference.py`（gmsh OCC 布尔切分 → 共形多材料网格，CG2 单元，约 13.4 万自由度；PETSc CG+GAMG，rtol=1e-12）。

- 区域：气凝胶 `[-0.05,0]×[0,1]`（k=0.018）、铝墙框 `[0,1]²\(0.05,0.95)²`（k=167）、腔内静止空气（k=0.026）、三个设备（k=167，体积热源 7500/4898/637 W/m³）；
- 边界：左 x=-0.05 Dirichlet 218.15 K；右 x=1 Robin（h=20，T∞=303.15 K）；上/下绝热（自然边界）；
- 材料界面由共形网格保证温度/热流精确连续；
- 三个固定布局：偏左 (x1=0.25, x2=0.35)、居中 (0.5, 0.5)、偏右 (0.75, 0.65)。

运行：

```bash
~/miniforge/envs/fenicsx-env/bin/python fem/milestone0_reference.py results/milestone0_v2
~/miniforge/envs/fortran_Torch_cp3.9_cuda12x/bin/python fem/plot_fields.py results/milestone0_v2
```

输出：`results/milestone0/summary_{left,center,right}.json`、`all_layouts.json`、`Tfield_*.csv`（251×251 网格温度场，供 PINN 验证）、`T_*.xdmf`、`Tfield_all_layouts.png`。

### 1.2 结果（v2 工况：5 cm 气凝胶 + 5 cm 壁厚 + 圆 20 W + h=20；网格 h=0.006/0.003，约 13.4 万自由度）

| 布局 | dev1 T_max | dev2 T_max | dev3 T_max | Q_left | Q_right | 能量守恒误差 |
|---|---:|---:|---:|---:|---:|---:|
| 偏左  | 104.3 ℃ | 115.4 ℃ | 234.8 ℃ | 58.6 W | 861.4 W | 1.7e-5 |
| 居中  | 94.1 ℃ | 106.9 ℃ | 192.8 ℃ | 55.2 W | 864.8 W | 2.2e-5 |
| 偏右  | 83.5 ℃ | 98.3 ℃ | 222.7 ℃ | 51.7 W | 868.3 W | 1.1e-5 |

能量守恒误差均远小于 1% 硬门槛 ✓。右壁平均温度 73.1–73.4 ℃，气凝胶一维热阻退化误差 1.3e-5 ✓。
结果文件：`results/milestone0_v2/`（summary_*.json、all_layouts.json、Tfield_*.csv、T_*.xdmf、Tfield_all_layouts.png）。

### 1.3 结论

1. **静止空气纯导热下不存在满足 70 ℃ 约束的可行布局**：三个布局中固定圆设备（20 W，无固体热桥）T_max 达 193–235 ℃，两个带热桥的方块也达 84–115 ℃。与方案 7.1 预判一致（排热上限 45+800=845 W < 920 W）。**该结论是阶段 A 降阶模型的属性**——Milestone 2 辐射（T⁴）与阶段 B 自然对流会显著改变温度水平，届时再判断可行性。
2. 布局影响符合预期：设备偏右时 T_max 更低（靠近右侧主热沉），偏左时 Q_left 略增；右侧 Robin 承担约 94% 排热（865 W/920 W）。
3. 气凝胶层精确退化为一维热阻（相对误差 1.3e-5），因此 PINN 侧按方案 Milestone 1 的要求直接将其退化处理（见 2.1）。

### 1.4 附：v1 工况（5 mm 气凝胶 + 1 cm 壁厚 + 圆 150 W + h=15）旧结果存档

v1 结果在 `results/milestone0/`（能量守恒 0.3–0.4%）：圆设备 T_max 高达 1075–1270 ℃（150 W 经静止空气的极端热点），方块 107–184 ℃。v1 已被 v2 工况取代，仅作存档。

---

## 2. Milestone 1：PINN 多材料纯导热（固体热桥）

### 2.1 代码结构（按方案 8.2 重组）

```text
src/
  config.py        # 有量纲/无量纲参数、布局、训练超参
  geometry.py      # 区域掩码/采样、界面采样（带固定方向）、界面元数据、DesignVars(x1,x2 sigmoid, 本里程碑冻结)
  networks.py      # MLP + TemperatureField（分域分支，输入仿射归一化、输出线性尺度、dev3 对数 halo 富集）
  losses/
    conduction.py     # 分域导热残差（各方程除以自身 k；设备含 S_k 源项，源项加权 --w-pde-dev）
    interface.py      # 材料界面的温度连续 + 热流连续（唯一含导热系数比的位置）
    boundary.py       # 左 x=0 气凝胶一维 Robin、上/下绝热、右 Robin（物理热流形式/Q_REF 归一）
    energy_budget.py  # 每设备 + 全局能量预算残差（积分型约束，驱动排热/热点形成）
  monitors.py      # 边界排热、能量守恒、设备 T_max、安装界面连续性、气凝胶一维化检验
  main.py          # Milestone 1 训练入口（重写）：Adam 阶段 + L-BFGS 阶段
  validate.py      # 与 Milestone 0 FEM 参考解对比
```

归一化严格按方案第 4 节：θ=(T−218.15)/85，Bi_w=0.1198，S₁=0.528，S₂=0.345，S₃=0.0449；热流类残差以 Q_REF=500 W/m²、界面通量以 Q_IF=150 W/m² 归一。

**关键数值处理（调试过程记录见 git 历史与本文 2.4 节）：**

1. **气凝胶退化为一维热阻**（方案 Milestone 1 明确要求验证该退化）：FEM 显式层与一维模型误差 1.3e-5，因此 PINN 中 5 cm 气凝胶不作为独立 PDE 域，而是作为 x=0 处墙体的 Robin 边界 k_Al·∂θ/∂x = h_TBL·θ（h_TBL=k_TBL/l=0.36 W/m²K）。这同时消除了薄层二阶导放大 ~160000× 的数值病态。对比验证仍可对 FEM 显式层做（一维剖面重建）。
2. **设备能量预算损失（决定性修复）**：纯逐点损失下，界面热流连续残差中空气侧梯度的系数是 k_F·DT/Q_IF≈0.015，比铝侧小 6400 倍，导致空气域几乎收不到学习信号、温度场卡在全场≈30℃ 的假稳态（即无源齐次解）。新增每设备/全局的**积分型能量预算损失**（收敛时与 PDE+界面损失冗余，训练中提供 O(1) 驱动）：每个设备表面对外排热必须等于其功率，全腔外边界排热等于 920 W。
3. **空气域对数 halo 富集**：θ_air 附加可学习幅值 A·ln(R_far/r)/ln(R_far/R₃)（ln r 在二维严格调和，halo 自身不产生 PDE 残差），为圆设备的热晕提供直接的全局幅值通道。
4. **Adam + L-BFGS 两阶段**：Adam 建立结构后，L-BFGS（拟牛顿）沿刚性耦合峡谷大步前进，把能量守恒从 ~50% 推到 ~2%。

---

## 3. 文件与备份

- 改动前快照：`backups/pre_milestone1_20260819_121500/`（src、checkpoint、docs 完整副本）；
- 本地 git 仓库（无远程，仅本地版本管理）：初始提交 `snapshot before milestone 0/1 reproduction`，里程碑完成后有对应提交；
- FEM 结果：`results/milestone0/`；PINN 结果：`results/milestone1/`；PINN checkpoint：`checkpoint_m1/`；
- 旧 3.1.2 代码（`src/loss.py.bak`、`src/NeuralNetwork.py`、`src/loss.py`、`src/visualize.py`）保留未删，新 main.py 不再引用。
