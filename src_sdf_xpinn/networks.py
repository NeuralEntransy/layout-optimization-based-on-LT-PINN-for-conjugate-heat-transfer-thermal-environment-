"""Five independent temperature branches (XPINN-style decomposition)."""
import torch
import torch.nn as nn
import config as C


class MLP(nn.Module):
    def __init__(self, width, depth, sigma=0.0, fourier_dim=64):
        super().__init__()
        self.fourier_dim = 0
        n_in = 2
        if sigma > 0:
            self.register_buffer("fourier_B", torch.randn(2, fourier_dim) * sigma)
            self.fourier_dim = fourier_dim
            n_in += 2 * fourier_dim
        layers = [nn.Linear(n_in, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers += [nn.Linear(width, 1)]
        self.net = nn.Sequential(*layers)
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        if self.fourier_dim:
            proj = 2 * torch.pi * x @ self.fourier_B
            x = torch.cat((x, torch.sin(proj), torch.cos(proj)), dim=-1)
        return self.net(x)


NORMS = {
    "wall": ((.5, .5), (.5, .5)),
    "air": ((.5, .5), (.49, .49)),
    "dev1": ((.5, C.W_IN+C.D1/2), (.5, C.D1/2)),
    "dev2": ((.5, 1-C.W_IN-C.D2/2), (.5, C.D2/2)),
    "dev3": (C.C3, (C.R3, C.R3)),
}
OUT_SCALE = {"wall": 2., "air": 3., "dev1": 2., "dev2": 2.5, "dev3": 4.}


class TemperatureField(nn.Module):
    DOMAINS = C.DOMAINS
    def __init__(self, width=96, depth=5, fourier_sigma=0., fourier_dim=64,
                 theta_init=0.0):
        super().__init__()
        self.nets = nn.ModuleDict({d: MLP(width, depth, fourier_sigma,
                                          fourier_dim) for d in C.DOMAINS})
        for d in C.DOMAINS:
            center, scale = NORMS[d]
            self.register_buffer(f"center_{d}", torch.tensor(center).float())
            self.register_buffer(f"scale_{d}", torch.tensor(scale).float())
            self.register_buffer(f"outscale_{d}", torch.tensor(OUT_SCALE[d]))
            # A common constant field removes artificial initial interface
            # jumps.  PDE/source and energy gradients then build curvature.
            last = self.nets[d].net[-1]
            nn.init.zeros_(last.weight)
            nn.init.constant_(last.bias, theta_init / OUT_SCALE[d])

    def forward(self, domain, pts):
        z = (pts - getattr(self, f"center_{domain}")) / getattr(self, f"scale_{domain}")
        return getattr(self, f"outscale_{domain}") * self.nets[domain](z)

    def temperature_K(self, domain, pts):
        return C.T_C + C.DT * self(domain, pts)
