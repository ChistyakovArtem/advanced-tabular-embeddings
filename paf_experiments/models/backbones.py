"""
Backbone models.

MLP
    Standard MLP baseline.

PAFNet
    MLP where Linear→ReLU blocks are replaced by Linear→PAF blocks.
    PAF variants (controlled by `paf_variant`):
        "plain"  : concat[sin(v), cos(v)]          — no skip
        "sc"     : concat[sin(v), cos(v), x]       — skip with raw x
        "sc_af"  : concat[sin(v), cos(v), relu(x)] — skip with relu(x)

    The "sigma mode" (controlled by `sigma_mode`) determines how the
    periodic coefficients inside each hidden PAF layer are initialised:
        "const"  : all coeffs ~ N(0, sigma_init)  (single sigma HP)
        "grid"   : coeffs[j] ~ N(0, sigma_j),
                   sigma_j = exp(linspace(log(1e-3), log(100), k))

The 6 PAF-Net variants from the experiment plan are combinations of:
    paf_variant  in {"plain", "sc", "sc_af"}
    sigma_mode   in {"const", "grid"}

The 6 "embedding-only" variants (OriginalPeriodic / GridPeriodic ×
{plain, sc, sc_af}) are constructed externally via build_embedding() +
a plain MLP backbone — see experiments/runner.py.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .embeddings import _PeriodicBase, OriginalPeriodic, GridPeriodic, PeriodicVariant


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mlp_block(in_dim: int, out_dim: int,
                    dropout: float = 0.0) -> nn.Sequential:
    layers: list[nn.Module] = [nn.Linear(in_dim, out_dim), nn.ReLU()]
    if dropout > 0.0:
        layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# MLP  (standard baseline)
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    """
    Standard MLP.

    Architecture:
        Flatten → [Linear → ReLU → (Dropout)] × n_layers → Linear (head)

    Parameters
    ----------
    in_dim      : int     input dimensionality (after any upstream embedding)
    hidden_dim  : int
    n_layers    : int     number of hidden blocks
    out_dim     : int     output dimensionality (1 for regression / binary)
    dropout     : float
    """

    def __init__(self, in_dim: int, hidden_dim: int = 256,
                 n_layers: int = 3, out_dim: int = 1,
                 dropout: float = 0.0):
        super().__init__()
        dims = [in_dim] + [hidden_dim] * n_layers
        self.blocks = nn.Sequential(
            *[_make_mlp_block(dims[i], dims[i + 1], dropout)
              for i in range(n_layers)]
        )
        self.head = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.blocks(x))


# ---------------------------------------------------------------------------
# PAF hidden layer
# ---------------------------------------------------------------------------

class _PAFLayer(nn.Module):
    """
    One PAF hidden block: [LayerNorm →] Linear → PAF-activation.

    Architecture rationale
    ----------------------
    In the original paper Periodic is applied to raw (quantile-normalised) input
    scalars x_i, so the scale is controlled.  Inside a network the pre-activation
    values are not normalised, which makes the frequency content of sin/cos
    unpredictable.  We add an *optional* LayerNorm before the linear projection
    to restore a controlled input scale at each layer.

    The linear layer maps  in_dim → k  and produces pre-activation values h.
    We then compute  v = 2π * h  and apply sin/cos.  There is NO separate
    `coeffs` parameter — that would be redundant with the linear weights.
    The "frequency" is entirely determined by the learned linear projection.

    Output dim depends on variant:
        plain  : 2 * k
        sc     : 2 * k + in_dim   (concatenate pre-LayerNorm input x)
        sc_af  : 2 * k + in_dim   (concatenate relu(x))
    """

    def __init__(self, in_dim: int, k: int,
                 variant: PeriodicVariant,
                 use_layernorm: bool = True):
        super().__init__()
        self.variant = PeriodicVariant(variant)
        self.k = k
        self.in_dim = in_dim
        self.norm   = nn.LayerNorm(in_dim) if use_layernorm else nn.Identity()
        self.linear = nn.Linear(in_dim, k)

        # Initialise linear weights so that the effective frequencies at init
        # span a useful range — we use the same N(0,1) default but scale by
        # 1/sqrt(in_dim) (standard Kaiming), which keeps h ~ O(1).
        nn.init.normal_(self.linear.weight, 0.0, 1.0 / math.sqrt(in_dim))
        nn.init.zeros_(self.linear.bias)

    @property
    def output_dim(self) -> int:
        base = 2 * self.k
        if self.variant in (PeriodicVariant.sc, PeriodicVariant.sc_af):
            return base + self.in_dim
        return base

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, in_dim) -> (B, output_dim)"""
        x_norm = self.norm(x)                           # (B, in_dim)
        h = self.linear(x_norm)                         # (B, k)
        v = 2.0 * math.pi * h
        periodic = torch.cat([torch.sin(v), torch.cos(v)], dim=-1)  # (B, 2k)

        if self.variant == PeriodicVariant.sc:
            return torch.cat([periodic, x], dim=-1)     # skip: unnormed x
        if self.variant == PeriodicVariant.sc_af:
            return torch.cat([periodic, F.relu(x)], dim=-1)
        return periodic


# ---------------------------------------------------------------------------
# PAFNet
# ---------------------------------------------------------------------------

class PAFNet(nn.Module):
    """
    MLP where every hidden Linear→ReLU is replaced by Linear→PAF.

    Architecture:
        [LayerNorm → Linear → PAF] × n_layers → Linear (head)

    LayerNorm before each PAF layer normalises the activation scale, making
    periodic functions behave predictably inside the network (mirrors what
    quantile-transform does at the input level in the original paper).

    Parameters
    ----------
    in_dim        : int
    k             : int     frequencies per PAF layer  (~= hidden_dim / 2)
    n_layers      : int
    out_dim       : int
    paf_variant   : str     "plain" | "sc" | "sc_af"
    use_layernorm : bool    apply LayerNorm before each PAF (recommended: True)
    dropout       : float
    """

    def __init__(self, in_dim: int, k: int = 128,
                 n_layers: int = 3, out_dim: int = 1,
                 paf_variant: str = "plain",
                 use_layernorm: bool = True,
                 dropout: float = 0.0):
        super().__init__()
        variant = PeriodicVariant(paf_variant)

        layers: list[nn.Module] = []
        cur_dim = in_dim
        for _ in range(n_layers):
            layer = _PAFLayer(cur_dim, k, variant, use_layernorm=use_layernorm)
            layers.append(layer)
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            cur_dim = layer.output_dim

        self.blocks = nn.Sequential(*layers)
        self.head = nn.Linear(cur_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.blocks(x))


# ---------------------------------------------------------------------------
# MLP + upstream Periodic embedding  (the "embedding-only" variants)
# ---------------------------------------------------------------------------

class EmbeddingMLP(nn.Module):
    """
    Upstream Periodic embedding → flatten → MLP.

    This implements the 6 "embedding-only" variants:
        embedding_type × {original, grid}  ×  variant × {plain, sc, sc_af}

    Parameters
    ----------
    embedding   : _PeriodicBase   pre-built embedding module
    n_features  : int
    hidden_dim  : int
    n_layers    : int
    out_dim     : int
    dropout     : float
    """

    def __init__(self, embedding: _PeriodicBase,
                 n_features: int,
                 hidden_dim: int = 256,
                 n_layers: int = 3,
                 out_dim: int = 1,
                 dropout: float = 0.0):
        super().__init__()
        self.embedding = embedding
        flat_dim = n_features * embedding.output_dim
        self.backbone = MLP(flat_dim, hidden_dim, n_layers, out_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, n_features)
        z = self.embedding(x)           # (B, n_features, emb_dim)
        z = z.flatten(start_dim=1)      # (B, n_features * emb_dim)
        return self.backbone(z)
