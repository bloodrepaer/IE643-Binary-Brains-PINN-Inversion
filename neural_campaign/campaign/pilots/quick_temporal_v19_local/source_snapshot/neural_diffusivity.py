"""Shared, bounded, continuous log-diffusivity for adjoint-NN and PINN.

Coordinates are physical x,y on the existing unit-square benchmark. The
network returns log D; diffusivity() returns positive physical D, not D*T.
"""
from dataclasses import asdict, dataclass
import math
import torch
from torch import nn


@dataclass
class CoefficientConfig:
    width: int = 32
    depth: int = 3
    d_min: float = 1e-3
    d_max: float = 10.0


class NeuralDiffusivity(nn.Module):
    def __init__(self, config=None, initial_D=0.25, seed=42):
        super().__init__()
        self.config = config or CoefficientConfig()
        c = self.config
        if c.width < 1 or c.depth < 1 or not 0 < c.d_min < initial_D < c.d_max:
            raise ValueError("Positive width/depth and d_min < initial_D < d_max required")
        # Local seed: coefficient initialisation is identical in both methods,
        # independent of how many random numbers the temperature net consumes.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            layers, n = [], 2
            for _ in range(c.depth):
                lin = nn.Linear(n, c.width, dtype=torch.float64)
                nn.init.xavier_uniform_(lin.weight); nn.init.zeros_(lin.bias)
                layers.extend([lin, nn.Tanh()]); n = c.width
            last = nn.Linear(n, 1, dtype=torch.float64)
            nn.init.zeros_(last.weight)
            p = (math.log(initial_D) - math.log(c.d_min)) / math.log(c.d_max / c.d_min)
            nn.init.constant_(last.bias, math.log(p / (1 - p)))
            self.net = nn.Sequential(*layers, last)

    def forward(self, xy):
        c = self.config
        return math.log(c.d_min) + math.log(c.d_max / c.d_min) * torch.sigmoid(self.net(2 * xy - 1))

    def diffusivity(self, xy):
        return self(xy).exp()

    def checkpoint(self):
        return {"config": asdict(self.config), "state_dict": self.state_dict()}

    @classmethod
    def from_checkpoint(cls, checkpoint):
        c = CoefficientConfig(**checkpoint["config"])
        model = cls(c, initial_D=math.sqrt(c.d_min * c.d_max))
        model.load_state_dict(checkpoint["state_dict"])
        return model
