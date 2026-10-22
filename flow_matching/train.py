"""Entrenamiento de FlowPTv3 con flow matching sobre las vasijas completas.

Uso (desde la raíz del proyecto, kernel/entorno "Python (PFC)"):
  python -m flow_matching.train --out runs/flow_ptv3
  python -m flow_matching.train --n-points 16384 --batch 8 --amp bf16
  python -m flow_matching.train --clouds datasets/points_complete_2048 --n-points 2048 --batch 2
  python -m flow_matching.train --resume runs/flow_ptv3/ckpt_last.pt

En la carpeta de salida:
  config.json      argumentos y configuración del modelo
  splits.json      vasijas de cada split
  log.csv          paso, pérdida, lr, norma del gradiente, pérdida de validación
  ckpt_last.pt     último checkpoint (modelo, EMA, optimizador, scheduler, paso)
  ckpt_<paso>.pt   checkpoints periódicos
  samples/<paso>/  vasijas generadas con Euler por la EMA, como .ply
"""

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import trimesh
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.utils.data import DataLoader

from .cfm import cfm_loss
from .data import GENERATED, CompleteClouds, list_clouds, split_clouds
from .flow_model import FlowPTv3
from .sampler import euler_sample

LAST_CKPT = "ckpt_last.pt"


def save_ply(points, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    trimesh.PointCloud(np.asarray(points, dtype=np.float32)).export(str(path))


def warmup_cosine(warmup, total):
    def factor(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = min(1.0, (step - warmup) / max(1, total - warmup))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return factor


@torch.no_grad()
def validate(model, loader, device, amp_dtype, t_kind):
    """Pérdida media en validación, con ruido y tiempos fijos para que sea comparable entre pasos."""
    model.eval()
    g = torch.Generator(device=device).manual_seed(0)
    total, count = 0.0, 0
    for x1 in loader:
        x1 = x1.to(device)
        with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
            loss, _ = cfm_loss(model, x1, t_kind=t_kind, generator=g)
        total += loss.item() * len(x1)
        count += len(x1)
    model.train()
    return total / max(count, 1)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(GENERATED),
                    help="carpeta datasets/generated: de su manifest.csv salen los splits por vasija")
    ap.add_argument("--clouds", default=None,
                    help="carpeta de nubes <Categoria>/<modelo>.ply (por defecto <data>/points_complete)")
    ap.add_argument("--out", default="runs/flow_ptv3")
    ap.add_argument("--n-points", type=int, default=8192,
                    help="puntos por nube en entrenamiento (como mucho los que tenga cada nube)")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--iters", type=int, default=100_000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--clip-grad", type=float, default=1.0)
    ap.add_argument("--t-kind", choices=["logit_normal", "uniform"], default="logit_normal")
    ap.add_argument("--amp", choices=["none", "bf16", "fp16"], default="none")
    ap.add_argument("--grid-size", type=float, default=0.01)
    ap.add_argument("--time-dim", type=int, default=256, help="dimensión del embedding de tiempo")
    ap.add_argument("--time-in-dim", type=int, default=0, help="canales de tiempo pegados a cada punto (0 = ninguno)")
    ap.add_argument("--no-time-in-blocks", action="store_true",
                    help="no modular las normalizaciones de PTv3 con t (solo entrada y cabeza; para comparar)")
    ap.add_argument("--drop-path", type=float, default=0.3)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--val-every", type=int, default=2000)
    ap.add_argument("--sample-every", type=int, default=5000)
    ap.add_argument("--sample-steps", type=int, default=100, help="pasos de Euler para las muestras")
    ap.add_argument("--save-every", type=int, default=5000)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", default=None)
    return ap.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = "cuda"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    amp_dtype = {"none": None, "bf16": torch.bfloat16, "fp16": torch.float16}[args.amp]

    # datos
    args.clouds = args.clouds or str(Path(args.data) / "points_complete")
    splits = split_clouds(list_clouds(args.clouds), Path(args.data) / "manifest.csv", seed=args.seed)
    (out / "splits.json").write_text(json.dumps({k: [str(p) for p in v] for k, v in splits.items()}, indent=1),
                                     encoding="utf-8")
    train_set = CompleteClouds(splits["train"], args.n_points, augment=True)
    val_set = CompleteClouds(splits["val"], args.n_points, augment=False)
    train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True, drop_last=True,
                              num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0)
    val_loader = DataLoader(val_set, batch_size=args.batch, shuffle=False, num_workers=0)
    print(f"vasijas: train {len(train_set)}, val {len(val_set)}, test {len(splits['test'])}")

    # modelo
    model_cfg = dict(time_dim=args.time_dim, time_in_dim=args.time_in_dim,
                     time_in_blocks=not args.no_time_in_blocks, grid_size=args.grid_size,
                     ptv3_kwargs=dict(drop_path=args.drop_path))
    model = FlowPTv3(**model_cfg).to(device)
    print(f"tiempo: {model.n_time_norms} normalizaciones de PTv3 moduladas, "
          f"{args.time_in_dim} canales en la entrada, FiLM en la cabeza")
    ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(args.ema), use_buffers=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, warmup_cosine(args.warmup, args.iters))
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype is torch.float16)
    print(f"parametros: {sum(p.numel() for p in model.parameters()) / 1e6:.1f} M")
    (out / "config.json").write_text(json.dumps({"args": vars(args), "model_cfg": model_cfg}, indent=1),
                                     encoding="utf-8")

    step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        ema.load_state_dict(ckpt["ema"])
        opt.load_state_dict(ckpt["opt"])
        sched.load_state_dict(ckpt["sched"])
        scaler.load_state_dict(ckpt["scaler"])
        step = ckpt["step"]
        print(f"reanudado desde el paso {step}")

    def save(path):
        torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "scaler": scaler.state_dict(), "step": step,
                    "model_cfg": model_cfg, "args": vars(args)}, path)

    log_path = out / "log.csv"
    new_log = not log_path.exists()
    log_file = open(log_path, "a", newline="", encoding="utf-8")
    log = csv.writer(log_file)
    if new_log:
        log.writerow(["step", "loss", "lr", "grad_norm", "val_loss", "seconds", "gpu_mem_gb"])

    model.train()
    batches = iter(train_loader)
    t0, running = time.time(), []
    while step < args.iters:
        try:
            x1 = next(batches)
        except StopIteration:
            batches = iter(train_loader)
            x1 = next(batches)
        x1 = x1.to(device, non_blocking=True)

        with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
            loss, _ = cfm_loss(model, x1, t_kind=args.t_kind)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad).item()
        scaler.step(opt)
        scaler.update()
        sched.step()
        ema.update_parameters(model)
        step += 1
        running.append(loss.item())

        val_loss = ""
        if step % args.val_every == 0 and len(val_set):
            val_loss = validate(ema.module, val_loader, device, amp_dtype, args.t_kind)
        if step % args.log_every == 0 or val_loss != "":
            elapsed = time.time() - t0
            mem = torch.cuda.max_memory_allocated() / 2**30   # pico de memoria de la GPU desde el inicio
            log.writerow([step, np.mean(running), sched.get_last_lr()[0], grad_norm, val_loss, round(elapsed, 1),
                          round(mem, 2)])
            log_file.flush()
            print(f"paso {step:>7}  perdida {np.mean(running):.4f}  lr {sched.get_last_lr()[0]:.2e}  "
                  f"|g| {grad_norm:.2f}" + (f"  val {val_loss:.4f}" if val_loss != "" else "") +
                  f"  {elapsed / 60:.1f} min  {mem:.2f} GB", flush=True)
            running = []
        if step % args.sample_every == 0:
            ema.module.eval()
            x, _ = euler_sample(ema.module, 2, args.n_points, steps=args.sample_steps, device=device)
            for i, cloud in enumerate(x.cpu().numpy()):
                save_ply(cloud, out / "samples" / f"{step:07d}" / f"sample_{i}.ply")
            ema.module.train()
        if step % args.save_every == 0:
            save(out / f"ckpt_{step:07d}.pt")
            save(out / LAST_CKPT)

    save(out / LAST_CKPT)
    log_file.close()
    print("entrenamiento terminado:", out / LAST_CKPT)


if __name__ == "__main__":
    main()
