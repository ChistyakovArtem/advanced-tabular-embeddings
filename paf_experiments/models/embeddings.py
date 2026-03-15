"""
Numerical feature embedding modules.

Implements:
  - OriginalPeriodic  : baseline from Gorishniy et al. (NeurIPS 2022)
                        c_i ~ N(0, sigma), sigma is a single HP
  - GridPeriodic      : c_i ~ N(0, sigma_i), where
                        sigma_i = exp(linspace(log(1e-3), log(100), k))
                        — replaces single-sigma HPO with a fixed frequency grid
  - Both are available with three "tail" variants:
        plain  : concat[sin(v), cos(v)]
        sc     : concat[sin(v), cos(v), x]           (skip-connection, append raw x)
        sc_af  : concat[sin(v), cos(v), relu(x)]     (skip-connection + nonlinearity)

Usage
-----
    emb = GridPeriodic(n_features=8, k=16, variant="sc_af")
    z = emb(x)   # x: (B, n_features)  ->  z: (B, n_features, 2*k + 1)  [sc/sc_af]
                 #                      ->  z: (B, n_features, 2*k)       [plain]
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from enum import Enum


class PeriodicVariant(str, Enum):
    plain  = "plain"   # concat[sin, cos]
    sc     = "sc"      # concat[sin, cos, x]
    sc_af  = "sc_af"   # concat[sin, cos, relu(x)]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _grid_sigmas(k: int, log_lo: float = math.log(1e-3),
                 log_hi: float = math.log(100.0)) -> torch.Tensor:
    """Return 1-D tensor of k sigma values spaced on log scale."""
    return torch.exp(torch.linspace(log_lo, log_hi, k))


def _output_dim(k: int, variant: PeriodicVariant) -> int:
    """Embedding dim per feature."""
    base = 2 * k  # sin + cos
    if variant in (PeriodicVariant.sc, PeriodicVariant.sc_af):
        return base + 1
    return base


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class _PeriodicBase(nn.Module):
    """
    Shared forward logic.  Subclasses only need to define self.coeffs
    as a Parameter of shape (n_features, k).
    """

    def __init__(self, n_features: int, k: int, variant: PeriodicVariant):
        super().__init__()
        self.n_features = n_features
        self.k = k
        self.variant = PeriodicVariant(variant)

    # subclasses must set self.coeffs: nn.Parameter  shape (n_features, k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args
            x : (B, n_features)   raw (pre-normalised) scalar features

        Returns
            z : (B, n_features, output_dim_per_feature)
        """
        # x: (B, F)  ->  (B, F, 1)
        x_ = x.unsqueeze(-1)

        # v_ij = 2π * c_ij * x_i   shape (B, F, k)
        v = 2.0 * math.pi * self.coeffs.unsqueeze(0) * x_  # broadcast over B

        out = torch.cat([torch.sin(v), torch.cos(v)], dim=-1)  # (B, F, 2k)

        if self.variant == PeriodicVariant.sc:
            out = torch.cat([out, x_], dim=-1)                 # (B, F, 2k+1)
        elif self.variant == PeriodicVariant.sc_af:
            out = torch.cat([out, F.relu(x_)], dim=-1)        # (B, F, 2k+1)

        return out

    @property
    def output_dim(self) -> int:
        return _output_dim(self.k, self.variant)


# ---------------------------------------------------------------------------
# OriginalPeriodic  (single sigma HP, matches paper exactly)
# ---------------------------------------------------------------------------

class OriginalPeriodic(_PeriodicBase):
    """
    Gorishniy et al. (2022) Periodic embedding.

    c_i ~ N(0, sigma)  where sigma is a tunable hyperparameter.
    coeffs are *trained* (not frozen) — matches the paper.

    Parameters
    ----------
    n_features : int
    k          : int    number of frequencies per feature
    sigma      : float  std for initialisation of coefficients
    variant    : str    one of "plain" | "sc" | "sc_af"
    """

    def __init__(self, n_features: int, k: int = 48,
                 sigma: float = 1.0,
                 variant: str = "plain"):
        super().__init__(n_features, k, variant)
        coeffs = torch.empty(n_features, k).normal_(0.0, sigma)
        self.coeffs = nn.Parameter(coeffs)

    def extra_repr(self) -> str:
        return (f"n_features={self.n_features}, k={self.k}, "
                f"sigma_init={self.coeffs.std().item():.4f}, variant={self.variant}")


# ---------------------------------------------------------------------------
# GridPeriodic  (grid of sigmas, no single-sigma HPO needed)
# ---------------------------------------------------------------------------

class GridPeriodic(_PeriodicBase):
    """
    Periodic embedding with a *grid* of initialisation sigmas.

    Instead of tuning a single sigma in [1e-3, 100], we tile the full
    frequency range by spreading k coefficients across
        sigma_i = exp(linspace(log(1e-3), log(100), k))

    This captures low- AND high-frequency patterns simultaneously,
    removing the need for sigma HPO.

    The coefficients remain *trainable* — the grid only determines
    the initial distribution.

    Parameters
    ----------
    n_features : int
    k          : int    number of frequencies per feature
    log_lo     : float  lower bound of log-sigma grid (default: log(1e-3))
    log_hi     : float  upper bound of log-sigma grid (default: log(100))
    variant    : str    one of "plain" | "sc" | "sc_af"
    """

    def __init__(self, n_features: int, k: int = 48,
                 log_lo: float = math.log(1e-3),
                 log_hi: float = math.log(100.0),
                 variant: str = "plain"):
        super().__init__(n_features, k, variant)
        self.log_lo = log_lo
        self.log_hi = log_hi

        sigmas = _grid_sigmas(k, log_lo, log_hi)  # (k,)
        # Each feature gets its own copy initialised from N(0, sigma_j)
        # coeffs[i, j] ~ N(0, sigmas[j])
        coeffs = torch.randn(n_features, k) * sigmas.unsqueeze(0)  # broadcast
        self.coeffs = nn.Parameter(coeffs)

    def extra_repr(self) -> str:
        return (f"n_features={self.n_features}, k={self.k}, "
                f"log_lo={self.log_lo:.2f}, log_hi={self.log_hi:.2f}, "
                f"variant={self.variant}")


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

EMBEDDING_REGISTRY: dict[str, type] = {
    "original": OriginalPeriodic,
    "grid":     GridPeriodic,
}


def build_embedding(name: str, n_features: int, k: int,
                    variant: str = "plain", **kwargs) -> _PeriodicBase:
    """
    Convenience factory.

    name    : "original" | "grid"
    variant : "plain" | "sc" | "sc_af"
    kwargs  : passed to the class (e.g. sigma= for OriginalPeriodic)
    """
    cls = EMBEDDING_REGISTRY[name]
    return cls(n_features=n_features, k=k, variant=variant, **kwargs)
