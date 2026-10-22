"""Muestreo con Euler desde un checkpoint de FlowPTv3 y evaluación opcional.

Uso (desde la raíz del proyecto, entorno "Python (PFC)"):
  python -m flow_matching.sample --ckpt runs/flow_ptv3/ckpt_last.pt --n 8
  python -m flow_matching.sample --ckpt ... --n 4 --snapshots 15        # guarda la trayectoria
  python -m flow_matching.sample --ckpt ... --n 4 --gif                 # y un GIF por vasija
  python -m flow_matching.sample --ckpt ... --eval --split test         # MMD / COV / 1-NNA

Salida en --out (por defecto <carpeta del checkpoint>/generated):
  sample_000.ply                     vasija generada (t = 1)
  sample_000/step_00_t0.000.ply ...  trayectoria, con --snapshots o --gif
  sample_000.gif                     los puntos moviéndose del ruido a la vasija, con --gif
  metrics.json                       con --eval
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import trimesh

from .data import CompleteClouds, split_clouds, list_clouds, GENERATED
from .flow_model import FlowPTv3
from .metrics import generation_metrics
from .sampler import euler_sample


def save_ply(points, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    trimesh.PointCloud(np.asarray(points, dtype=np.float32)).export(str(path))


def load_model(ckpt_path, device, use_ema=True):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = FlowPTv3(**ckpt["model_cfg"]).to(device)
    if use_ema and "ema" in ckpt:
        # AveragedModel guarda los pesos bajo "module." y un contador "n_averaged"
        state = {k[len("module."):]: v for k, v in ckpt["ema"].items() if k.startswith("module.")}
    else:
        state = ckpt["model"]
    model.load_state_dict(state)
    model.eval()
    return model, ckpt


def generate(model, n, n_points, steps, batch, device, snapshots, seed):
    g = torch.Generator(device=device).manual_seed(seed)
    clouds, trajectories = [], []
    for start in range(0, n, batch):
        b = min(batch, n - start)
        x, traj = euler_sample(model, b, n_points, steps=steps, device=device, snapshots=snapshots, generator=g)
        clouds.append(x.cpu())
        trajectories.extend([[(t, xt[i]) for t, xt in traj] for i in range(b)] if traj else [[] for _ in range(b)])
    return torch.cat(clouds), trajectories


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--n", type=int, default=8, help="vasijas a generar (con --eval: las del split)")
    ap.add_argument("--n-points", type=int, default=8192)
    ap.add_argument("--steps", type=int, default=100, help="pasos de Euler")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--snapshots", type=int, default=0, help="instantes de la trayectoria a guardar (0 = ninguno)")
    ap.add_argument("--gif", action="store_true", help="un GIF por vasija (si --snapshots es 0 usa 60 cuadros)")
    ap.add_argument("--gif-ms", type=int, default=80, help="duración de cada cuadro del GIF")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-ema", action="store_true", help="usar los pesos sin promediar")
    ap.add_argument("--eval", action="store_true", help="comparar contra las vasijas reales del split")
    ap.add_argument("--split", default="test")
    ap.add_argument("--eval-points", type=int, default=2048, help="puntos por nube para las métricas")
    ap.add_argument("--emd", action="store_true", help="calcular también EMD (lento)")
    args = ap.parse_args()

    device = "cuda"
    model, ckpt = load_model(args.ckpt, device, use_ema=not args.no_ema)
    out = Path(args.out) if args.out else Path(args.ckpt).parent / "generated"

    ref = None
    if args.eval:
        data_dir = Path(ckpt["args"]["data"])
        clouds_dir = ckpt["args"].get("clouds") or data_dir / "points_complete"  # las mismas nubes del entrenamiento
        paths = split_clouds(list_clouds(clouds_dir), data_dir / "manifest.csv", seed=ckpt["args"]["seed"])[args.split]
        ref = torch.stack([c for c in CompleteClouds(paths, args.eval_points, augment=False)])
        args.n = len(ref)

    if args.gif and args.snapshots == 0:
        args.snapshots = 60
    clouds, trajectories = generate(model, args.n, args.n_points, args.steps, args.batch, device,
                                    args.snapshots, args.seed)
    for i, (cloud, traj) in enumerate(zip(clouds.numpy(), trajectories)):
        save_ply(cloud, out / f"sample_{i:03d}.ply")
        for k, (t, xt) in enumerate(traj):
            save_ply(xt.numpy(), out / f"sample_{i:03d}" / f"step_{k:02d}_t{t:.3f}.ply")
        if args.gif and traj:
            from .animate import animate
            animate([(t, xt.numpy()) for t, xt in traj], out / f"sample_{i:03d}.gif", ms=args.gif_ms)
    print(f"{len(clouds)} vasijas generadas ({args.steps} pasos de Euler) -> {out}"
          + (" (con GIF)" if args.gif else ""))

    if args.eval:
        gen = torch.stack([c[torch.randperm(len(c))[: args.eval_points]] for c in clouds]).to(device)
        ref = ref.to(device)
        metrics = generation_metrics(gen, ref, "cd")
        if args.emd:
            metrics.update(generation_metrics(gen, ref, "emd"))
        metrics.update({"split": args.split, "n": len(ref), "euler_steps": args.steps})
        (out / "metrics.json").write_text(json.dumps(metrics, indent=1), encoding="utf-8")
        print(json.dumps(metrics, indent=1))


if __name__ == "__main__":
    main()
