# 运行指令速查

> 环境：WSL2。FEM 用 conda 环境 `fenicsx-env`（dolfinx 0.10 + gmsh）；PINN / 绘图用 `fortran_Torch_cp3.9_cuda12x`（torch 2.8.0 + matplotlib）。
> 仓库根目录：`/mnt/f/LT-PINN`

## 参数位置

| 参数 | 文件 | 说明 |
|---|---|---|
| 设备功率 P1/P2/P3 | `fem/milestone0_reference.py` 第 45–47 行 | 改这里，辐射脚本自动跟随（共享 `P`） |
| 气凝胶厚度 `x_tbl` | 同上 第 44 行 | 当前 `-0.01`（1 cm） |
| 右侧换热 `h_ext` / `T_inf` | 同上 | 当前 20 / 303.15 K |
| 布局 x1/x2 | 同上 `LAYOUTS` | left / center / right |
| 右边界类型 | 命令行参数 | `robin`（对流）或 `dirichlet`（30℃ 恒温） |

## FEM：仅导热（无辐射）

```bash
cd /mnt/f/LT-PINN
~/miniforge/envs/fenicsx-env/bin/python fem/milestone0_reference.py <输出目录> <布局> <功率缩放> <右边界>
# 例：
~/miniforge/envs/fenicsx-env/bin/python fem/milestone0_reference.py results/milestone0_v3 left,center,right 1.0 robin
~/miniforge/envs/fenicsx-env/bin/python fem/milestone0_reference.py results/milestone0_v3_dirichlet left,center,right 1.0 dirichlet
```

## FEM：导热 + 辐射

```bash
cd /mnt/f/LT-PINN/fem   # 必须在 fem/ 里跑（import 同目录模块）
~/miniforge/envs/fenicsx-env/bin/python milestone0_radiation.py <输出目录> <布局> <右边界>
# 例：
~/miniforge/envs/fenicsx-env/bin/python milestone0_radiation.py ../results/milestone0_radiation_v3 left,center,right robin
~/miniforge/envs/fenicsx-env/bin/python milestone0_radiation.py ../results/milestone0_radiation_v3_dirichlet left,center,right dirichlet
```

## 画温度场图（-55~70℃ 色卡）

```bash
cd /mnt/f/LT-PINN
# 导热版文件名前缀 Tfield_，辐射版 Tfield_rad_
~/miniforge/envs/fortran_Torch_cp3.9_cuda12x/bin/python fem/plot_fields.py results/milestone0_radiation_v3_dirichlet left,center,right Tfield_rad_
```

## PINN（Milestone 1，torch 环境，用 GPU）

```bash
cd /mnt/f/LT-PINN
# 训练（默认 Adam + 可选 L-BFGS 抛光）
~/miniforge/envs/fortran_Torch_cp3.9_cuda12x/bin/python src/main.py --layout center --epochs 6000 --lbfgs-steps 800
# 断点续训
~/miniforge/envs/fortran_Torch_cp3.9_cuda12x/bin/python src/main.py --layout center --resume --epochs 12000
# 验证（对比 FEM 参考解）
~/miniforge/envs/fortran_Torch_cp3.9_cuda12x/bin/python src/validate.py --layout center
```

## 结果目录约定

| 目录 | 内容 |
|---|---|
| `results/milestone0_v3` | v3 工况（气凝胶1cm）仅导热 |
| `results/milestone0_radiation_v3` | v3 导热+辐射，Robin 右侧 |
| `results/milestone0_v3_dirichlet` | v3 仅导热，Dirichlet 右侧 |
| `results/milestone0_radiation_v3_dirichlet` | v3 导热+辐射，Dirichlet 右侧 |
| `results/milestone0/` `results/milestone0_v2/` | v1 / v2 存档 |

每个 FEM 输出目录含：`summary_*.json`（各布局指标）、`all_layouts.json`、`Tfield_*.csv`（251×251 温度场）、`T_*.xdmf`（ParaView 可视化）。
