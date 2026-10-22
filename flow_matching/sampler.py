"""Integrador de Euler para dx/dt = v_θ(x, t), de t = 0 (ruido) a t = 1 (vasija)."""

import torch


def snapshot_steps(steps, frames, schedule="linear", min_noise=0.02):
    """Pasos de Euler (0 … steps) en los que guardar la nube, `frames` en total.

    linear  repartidos parejo en t
    log     el ruido que queda, 1 − t, baja a la misma razón entre cuadros (más cuadros al
            principio, donde la nube cambia mucho de tamaño) hasta `min_noise`, más t = 1
    """
    if frames <= 0:
        return set()
    if schedule == "linear" or frames < 3:
        return set(torch.linspace(0, steps, frames).round().long().tolist())
    i = torch.arange(frames - 1, dtype=torch.float64)
    t = 1.0 - min_noise ** (i / (frames - 2))
    return set((t * steps).round().long().clamp(0, steps).tolist()) | {steps}


@torch.no_grad()
def euler_sample(model, batch_size, n_points, steps=100, device="cuda", snapshots=0, generator=None, x0=None,
                 keep_steps=None):
    """Genera `batch_size` nubes de `n_points` puntos.

    x_{k+1} = x_k + h · v_θ(x_k, t_k),  t_k = k h,  h = 1 / steps

    snapshots > 0 devuelve además la trayectoria en ese número de instantes repartidos parejo
    entre t = 0 y t = 1; `keep_steps` (conjunto de pasos 0 … steps) la elige a mano.
    Devuelve (x1, trayectoria): x1 (B, N, 3); trayectoria = lista de (t, x) o [].
    """
    x = x0.to(device) if x0 is not None else torch.randn(batch_size, n_points, 3, device=device, generator=generator)
    times = torch.linspace(0.0, 1.0, steps + 1, device=device)
    keep = set(keep_steps) if keep_steps is not None else snapshot_steps(steps, snapshots)
    trajectory = [(0.0, x.cpu())] if 0 in keep else []
    for k in range(steps):
        t = times[k].expand(x.shape[0])
        x = x + (times[k + 1] - times[k]) * model(x, t).float()
        if k + 1 in keep:
            trajectory.append((float(times[k + 1]), x.cpu()))
    return x, trajectory
