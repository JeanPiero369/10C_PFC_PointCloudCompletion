"""Conditional Flow Matching con camino recto (rectified flow).

  x_t = (1 - t) x0 + t x1,   u_t = x1 - x0,   pérdida = E || v_θ(x_t, t) - u_t ||²
"""

import torch


def sample_t(batch_size, device, kind="logit_normal", mu=0.0, sigma=1.0, generator=None):
    """Tiempos de entrenamiento.

    logit_normal  t = sigmoid(N(mu, sigma)): concentra el entrenamiento en los tiempos
                  intermedios, donde se decide la forma (Esser et al., 2024, SD3)
    uniform       t ~ U[0, 1]
    """
    if kind == "uniform":
        return torch.rand(batch_size, device=device, generator=generator)
    if kind == "logit_normal":
        return torch.sigmoid(mu + sigma * torch.randn(batch_size, device=device, generator=generator))
    raise ValueError(f"tipo de muestreo de t desconocido: {kind}")


def interpolate(x0, x1, t):
    """Punto del camino recto y su velocidad. x0, x1: (B, N, 3); t: (B,)."""
    tt = t.view(-1, 1, 1)
    return (1.0 - tt) * x0 + tt * x1, x1 - x0


def cfm_loss(model, x1, t_kind="logit_normal", generator=None):
    """Pérdida de flow matching para un lote de nubes x1 (B, N, 3). Devuelve (pérdida, t)."""
    x0 = torch.randn(x1.shape, device=x1.device, dtype=x1.dtype, generator=generator)
    t = sample_t(x1.shape[0], x1.device, kind=t_kind, generator=generator)
    xt, ut = interpolate(x0, x1, t)
    vt = model(xt, t)
    loss = (vt.float() - ut.float()).pow(2).sum(-1).mean()  # norma al cuadrado por punto, promedio
    return loss, t
