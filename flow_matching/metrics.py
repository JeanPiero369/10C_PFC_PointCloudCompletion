"""Métricas de generación de nubes de puntos (protocolo de PointFlow / LION).

Entre dos conjuntos de nubes, generadas G y reales R:
  MMD    promedio, sobre cada real, de la distancia a su generada más cercana  (calidad)
  COV    fracción de reales que son la más cercana de alguna generada        (diversidad)
  1-NNA  acierto de un clasificador 1-vecino entre G ∪ R; ideal 50 %          (ambas)
Con distancia de Chamfer (CD) y, opcionalmente, Earth Mover's (EMD).
"""

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


def chamfer(a, b):
    """CD entre pares de nubes: a (B, N, 3), b (B, M, 3) -> (B,). Suma de distancias al cuadrado medias."""
    d = torch.cdist(a, b)
    return d.min(dim=2).values.pow(2).mean(1) + d.min(dim=1).values.pow(2).mean(1)


def emd(a, b):
    """EMD exacta entre dos nubes del mismo tamaño (numpy (N, 3)): asignación óptima, distancia media."""
    cost = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    rows, cols = linear_sum_assignment(cost)
    return float(cost[rows, cols].mean())


def pairwise(X, Y, metric="cd", batch=32):
    """Matriz de distancias (len(X), len(Y)) entre dos conjuntos de nubes (tensores (n, N, 3))."""
    out = torch.empty(len(X), len(Y))
    for i in range(len(X)):
        if metric == "cd":
            for j in range(0, len(Y), batch):
                yj = Y[j:j + batch]
                out[i, j:j + len(yj)] = chamfer(X[i:i + 1].expand(len(yj), -1, -1), yj).cpu()
        else:
            xi = X[i].cpu().numpy()
            for j in range(len(Y)):
                out[i, j] = emd(xi, Y[j].cpu().numpy())
    return out


def generation_metrics(gen, ref, metric="cd"):
    """MMD, COV y 1-NNA entre nubes generadas `gen` y reales `ref` (tensores (n, N, 3))."""
    d_rg = pairwise(ref, gen, metric)
    d_rr = pairwise(ref, ref, metric)
    d_gg = pairwise(gen, gen, metric)
    mmd = d_rg.min(dim=1).values.mean().item()
    cov = len(torch.unique(d_rg.argmin(dim=0))) / len(ref)

    n_r, n_g = len(ref), len(gen)
    full = torch.cat([torch.cat([d_rr, d_rg], 1), torch.cat([d_rg.T, d_gg], 1)], 0)
    full.fill_diagonal_(float("inf"))
    labels = torch.cat([torch.ones(n_r), torch.zeros(n_g)])
    nna = (labels[full.argmin(dim=1)] == labels).float().mean().item()
    return {f"MMD-{metric.upper()}": mmd, f"COV-{metric.upper()}": cov, f"1-NNA-{metric.upper()}": nna}
