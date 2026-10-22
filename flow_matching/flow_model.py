"""FlowPTv3: la red de velocidades v_θ(x_t, t) con Point Transformer V3 como backbone.

    x_t (B, N, 3), t (B,)
      │                       t ─► TimeEmbedding ─► temb (B, 256)
    [A] entrada                                       │
      │   atributos por punto = x_t (3)               │
      │                                               │
    [B] vóxeles                                       │
      │   los puntos de una misma celda se promedian: │
      │   PTv3 (spconv) supone un punto por celda     │
      │                                               │
    [C] PTv3 ◄───── temb en cada normalización ───────┤   (normalización adaptativa, como AdaLN de DiT)
      │   · 3 LayerNorm por bloque (CPE, antes de la  │
      │     atención, antes del MLP) × 22 bloques     │
      │   · BatchNorm del Embedding, de cada bajada   │
      │     y de cada subida de resolución            │
      │                                               │
    [D] cabeza ◄──────── FiLM(temb) ──────────────────┘
      │   característica de la celda + proyección del punto -> FiLM -> MLP -> 3
    v (B, N, 3)

Cada normalización de PTv3 hace  feat · (1 + escala(t)) + desplazamiento(t)  (PDNorm con
adaptive=True). Las capas que calculan escala y desplazamiento empiezan en cero
("adaLN-Zero", Peebles & Xie 2023): al inicio PTv3 se comporta como el original y el
tiempo se incorpora durante el entrenamiento.

Con time_in_blocks=False el tiempo solo entra por la cabeza y, si time_in_dim > 0, como
atributo de entrada (para comparar).
"""

import torch
import torch.nn as nn
from torch_scatter import scatter_mean

from .backbone import PDNORM_CONTEXT, PTV3_DEFAULTS, build_ptv3, load_ptv3_module
from .embeddings import FiLM, TimeEmbedding

CONDITION = "pottery"  # PDNorm exige un nombre de condición; hay una sola


class FlowPTv3(nn.Module):
    def __init__(self, time_dim=256, time_in_dim=0, time_in_blocks=True, grid_size=0.01, clip=4.0,
                 ptv3_kwargs=None):
        """
        time_dim        dimensión del embedding de tiempo
        time_in_dim     canales de tiempo pegados a cada punto en la entrada (0 = ninguno)
        time_in_blocks  modular con t todas las normalizaciones de PTv3
        grid_size       lado de la celda de PTv3 (la vasija cabe en la esfera unitaria)
        clip            x_t se recorta a ±clip para la grilla (el ruido N(0, I) casi no sale de ±4)
        ptv3_kwargs     cambios sobre la configuración original de PTv3 (backbone.PTV3_DEFAULTS)
        """
        super().__init__()
        self.grid_size, self.clip = grid_size, clip
        self.time_in_dim, self.time_in_blocks = time_in_dim, time_in_blocks
        in_channels = 3 + time_in_dim
        ptv3_kwargs = dict(ptv3_kwargs or {})
        if time_in_blocks:
            # normalización adaptativa de PTv3 en LayerNorm (bloques) y BatchNorm (embedding, bajadas, subidas)
            ptv3_kwargs.update(pdnorm_ln=True, pdnorm_bn=True, pdnorm_adaptive=True,
                               pdnorm_decouple=True, pdnorm_conditions=(CONDITION,))
        out_channels = tuple(ptv3_kwargs.get("dec_channels", PTV3_DEFAULTS["dec_channels"]))[0]

        self.time_embed = TimeEmbedding(time_dim)
        self.time_in = nn.Linear(time_dim, time_in_dim) if time_in_dim > 0 else None
        # PTv3 fija el tamaño del contexto de PDNorm en 256
        self.time_ctx = None
        if time_in_blocks:
            self.time_ctx = nn.Identity() if time_dim == PDNORM_CONTEXT else nn.Linear(time_dim, PDNORM_CONTEXT)
        self.backbone = build_ptv3(in_channels, **ptv3_kwargs)
        self.point_proj = nn.Linear(in_channels, out_channels)
        self.film = FiLM(time_dim, out_channels)
        self.head = nn.Sequential(
            nn.Linear(out_channels, 2 * out_channels),
            nn.GELU(),
            nn.Linear(2 * out_channels, 3),
        )
        nn.init.zeros_(self.head[-1].weight)  # al empezar predice velocidad 0
        nn.init.zeros_(self.head[-1].bias)
        self.n_time_norms = self._zero_init_modulation() if time_in_blocks else 0

    def _zero_init_modulation(self):
        """adaLN-Zero: escala y desplazamiento de cada PDNorm empiezan en 0. Devuelve cuántas hay."""
        pdnorm = load_ptv3_module().PDNorm
        count = 0
        for module in self.backbone.modules():
            if isinstance(module, pdnorm) and module.adaptive:
                nn.init.zeros_(module.modulation[-1].weight)
                nn.init.zeros_(module.modulation[-1].bias)
                count += 1
        return count

    def voxelize(self, coord, feat, batch):
        """Agrupa los puntos por (nube, celda). Devuelve el dict de PTv3 y el índice punto -> vóxel."""
        grid = torch.floor((coord + self.clip) / self.grid_size).long()  # origen fijo: todas las nubes en la misma grilla
        key = torch.cat([batch[:, None], grid], dim=1)
        uniq, inverse = torch.unique(key, dim=0, return_inverse=True)  # orden lexicográfico: agrupado por nube
        batch_vox = uniq[:, 0]
        offset = torch.cumsum(torch.bincount(batch_vox), dim=0)
        data = {
            "coord": scatter_mean(coord, inverse, dim=0),
            "grid_coord": uniq[:, 1:].int(),
            "feat": scatter_mean(feat, inverse, dim=0),
            "offset": offset,
            # se entrega `batch` hecho: si no, PTv3 lo crea con offset2batch bajo torch.inference_mode y
            # context[point.batch] (PDNorm) no puede guardarlo para el backward
            "batch": batch_vox,
            "grid_size": self.grid_size,
        }
        return data, inverse

    def forward(self, x, t):
        """x: (B, N, 3) puntos en el tiempo t; t: (B,) en [0, 1]. Devuelve la velocidad (B, N, 3)."""
        B, N, _ = x.shape
        temb = self.time_embed(t)                                    # (B, time_dim)
        batch = torch.arange(B, device=x.device).repeat_interleave(N)
        coord = x.reshape(-1, 3).clamp(-self.clip, self.clip)

        # [A] atributos por punto
        feat = coord if self.time_in is None else torch.cat([coord, self.time_in(temb)[batch]], dim=1)

        # [B] + [C] vóxeles y PTv3 (con t en cada normalización)
        data, inverse = self.voxelize(coord, feat, batch)
        if self.time_in_blocks:
            data["context"] = self.time_ctx(temb)                    # (B, 256), uno por nube
            data["condition"] = CONDITION
        point = self.backbone(data)                                  # point.feat: (vóxeles, C) en el orden de entrada

        # [D] cabeza por punto
        h = point.feat[inverse] + self.point_proj(feat)
        h = self.film(h, temb[batch])
        return self.head(h).reshape(B, N, 3)
