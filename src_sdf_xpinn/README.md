# SDF-XPINN thermal solver

This directory is an independent five-domain implementation.  It does not
import or overwrite checkpoints from `src/`.

The geometry is represented by signed-distance/CSG fields:

- `wall`: outer enclosure minus the cavity;
- `dev1`, `dev2`, `dev3`: two boxes and one disk;
- `air`: cavity minus all devices.

Temperature is represented by five independent MLP branches.  Explicit
interface points enforce temperature continuity and normal heat-flux
continuity.  This combines the explicit, differentiable geometry idea of
LT-PINN with the subdomain-network coupling used by XPINN/cPINN.

Each epoch retains uniform and SDF-boundary-layer domain points and appends a
fresh residual-adaptive random (RAR) set.  RAR samples a larger uniform
candidate pool with probabilities based on the current squared PDE residual.
Use `--no-rar` only for an ablation run.

Run a smoke test from the repository root:

```powershell
& 'D:\ANACONDA\envs\Pytorch\python.exe' .\src_sdf_xpinn\main.py `
  --epochs 3 --device cuda:0 --no-plot
```
