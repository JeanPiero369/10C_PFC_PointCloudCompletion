"""GIF de la inferencia: los puntos viajan desde el ruido (t = 0) hasta la vasija (t = 1).

Dos formas de uso (desde la raíz del proyecto, entorno "Python (PFC)"):

  1. Desde un checkpoint: genera con Euler y anima cada vasija.
       python -m flow_matching.animate --ckpt runs/flow_ptv3_2048/ckpt_last.pt --n 2 --n-points 2048

  2. Desde una trayectoria ya guardada (carpeta con step_*.ply, de sample.py --snapshots o flow/):
       python -m flow_matching.animate --from-dir "flow/Y2532_POTTERY5"

Cómo se dibuja:
  · cada punto conserva su color durante todo el GIF: el de su altura en la vasija final, así
    se sigue su recorrido desde el ruido;
  · la cámara sigue a la nube mientras se contrae (--zoom follow) o queda fija (--zoom fixed);
  · --spin gira la vista a lo largo de la animación; el último cuadro se mantiene --hold ms.

Salida: <out>/sample_000.gif (y, desde checkpoint, también sample_000.ply).
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import trimesh  # noqa: E402
from PIL import Image  # noqa: E402


def load_trajectory_dir(folder):
    """Trayectoria [(t, puntos N×3)] desde step_KK_tT.ply, en orden."""
    steps = sorted(Path(folder).glob("step_*.ply"))
    if not steps:
        raise FileNotFoundError(f"no hay step_*.ply en {folder}")
    traj = []
    for p in steps:
        t = float(p.stem.split("_t")[-1])
        traj.append((t, np.asarray(trimesh.load(p).vertices, dtype=np.float32)))
    return traj


def render_frames(trajectory, size=520, elev=18.0, azim=35.0, spin=0.0, zoom="follow", fixed_limit=2.5,
                  point_size=1.2, cmap="copper", title=True):
    """Una imagen PIL por instante de la trayectoria [(t, N×3)]."""
    clouds = [np.asarray(x, dtype=np.float32) for _, x in trajectory]
    final = clouds[-1]
    y = final[:, 1]
    colors = plt.get_cmap(cmap)((y - y.min()) / max(np.ptp(y), 1e-9))   # color fijo por punto
    final_limit = float(np.abs(final).max()) * 1.05
    frames = []
    for k, ((t, _), x) in enumerate(zip(trajectory, clouds)):
        if zoom == "follow":   # encuadre: el 98 % de los puntos, nunca más cerca que la vasija final
            limit = max(float(np.quantile(np.abs(x), 0.98)) * 1.05, final_limit)
        else:
            limit = fixed_limit
        fig = plt.figure(figsize=(size / 100, size / 100), dpi=100)
        ax = fig.add_axes([0, 0, 1, 1], projection="3d")
        keep = (np.abs(x) <= limit).all(1)
        ax.scatter(x[keep, 0], x[keep, 2], x[keep, 1], s=point_size, c=colors[keep], linewidths=0)  # +Y arriba
        for set_lim in (ax.set_xlim, ax.set_ylim, ax.set_zlim):
            set_lim(-limit, limit)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev, azim + spin * k / max(len(clouds) - 1, 1))
        ax.axis("off")
        if title:
            fig.text(0.04, 0.94, f"t = {t:.2f}", fontsize=13, color="#333333")
        fig.canvas.draw()
        frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()))
        plt.close(fig)
    return frames


def save_gif(frames, path, ms=80, hold=1500):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    durations = [ms] * (len(frames) - 1) + [hold]
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=durations, loop=0, optimize=True)
    return path


def animate(trajectory, path, ms=80, hold=1500, **render_kwargs):
    """Trayectoria [(t, N×3)] -> GIF. Devuelve la ruta."""
    return save_gif(render_frames(trajectory, **render_kwargs), path, ms=ms, hold=hold)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--ckpt", help="checkpoint de FlowPTv3: genera y anima")
    src.add_argument("--from-dir", help="carpeta con step_*.ply ya guardados")
    ap.add_argument("--out", default=None, help="carpeta de salida (por defecto <ckpt>/gifs o la carpeta de origen)")
    ap.add_argument("--n", type=int, default=2, help="vasijas a generar (con --ckpt)")
    ap.add_argument("--n-points", type=int, default=2048, help="puntos por nube (los del entrenamiento)")
    ap.add_argument("--steps", type=int, default=100, help="pasos de Euler")
    ap.add_argument("--frames", type=int, default=60, help="cuadros del GIF")
    ap.add_argument("--frame-schedule", choices=["linear", "log"], default="linear",
                    help="cómo repartir los cuadros en t (log: más cuadros al principio)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-ema", action="store_true")
    ap.add_argument("--ms", type=int, default=80, help="duración de cada cuadro")
    ap.add_argument("--hold", type=int, default=1500, help="duración del último cuadro")
    ap.add_argument("--size", type=int, default=520, help="lado del GIF en píxeles")
    ap.add_argument("--zoom", choices=["follow", "fixed"], default="follow")
    ap.add_argument("--limit", type=float, default=2.5, help="medio lado del cubo con --zoom fixed")
    ap.add_argument("--spin", type=float, default=0.0, help="grados que gira la vista durante el GIF")
    ap.add_argument("--elev", type=float, default=18.0)
    ap.add_argument("--azim", type=float, default=35.0)
    args = ap.parse_args()
    render = dict(size=args.size, elev=args.elev, azim=args.azim, spin=args.spin, zoom=args.zoom,
                  fixed_limit=args.limit)

    if args.from_dir:
        folder = Path(args.from_dir)
        out = Path(args.out) if args.out else folder.parent
        path = animate(load_trajectory_dir(folder), out / f"{folder.name}.gif", ms=args.ms, hold=args.hold, **render)
        print("GIF ->", path)
        return

    from .sample import load_model, save_ply   # evita cargar PTv3 cuando solo se anima una carpeta
    from .sampler import euler_sample, snapshot_steps

    device = "cuda"
    model, _ = load_model(args.ckpt, device, use_ema=not args.no_ema)
    out = Path(args.out) if args.out else Path(args.ckpt).parent / "gifs"
    keep = snapshot_steps(args.steps, args.frames, args.frame_schedule)
    g = torch.Generator(device=device).manual_seed(args.seed)
    for i in range(args.n):   # de a una vasija: menos memoria
        x1, traj = euler_sample(model, 1, args.n_points, steps=args.steps, device=device, generator=g, keep_steps=keep)
        save_ply(x1[0].cpu().numpy(), out / f"sample_{i:03d}.ply")
        path = animate([(t, xt[0].numpy()) for t, xt in traj], out / f"sample_{i:03d}.gif",
                       ms=args.ms, hold=args.hold, **render)
        print(f"vasija {i}: {len(traj)} cuadros -> {path}")


if __name__ == "__main__":
    main()
