"""Dataset de vasijas completas: <carpeta de nubes>/<Categoria>/<modelo>.ply.

Por defecto datasets/generated/points_complete (16384 puntos por nube); también sirve
datasets/points_complete_2048 (2048 puntos, las 1012 vasijas). Todas las nubes son
uniformes y están en el marco canónico (eje +Y, esfera unitaria).
En cada lectura se elige un subconjunto al azar de `n_points` y, en entrenamiento,
se aplica una rotación alrededor de Y (las vasijas son de revolución) y una escala.

Split por VASIJA (nunca la misma vasija en dos splits): si la vasija ya está en
manifest.csv se respeta su split; si no, se asigna de forma determinista por hash.
"""

import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import trimesh
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[1]
GENERATED = ROOT / "datasets" / "generated"


def list_clouds(clouds_dir=GENERATED / "points_complete"):
    """Nubes de una carpeta organizada como <clouds_dir>/<Categoria>/<modelo>.ply."""
    return sorted(Path(clouds_dir).glob("*/*.ply"))


def split_clouds(paths, manifest_csv=GENERATED / "manifest.csv", val=0.1, test=0.1, seed=0):
    """{'train': [...], 'val': [...], 'test': [...]} con rutas de .ply."""
    known = {}
    if Path(manifest_csv).exists():
        m = pd.read_csv(manifest_csv, usecols=["model_file", "split"]).drop_duplicates("model_file")
        known = {Path(f).stem: s for f, s in zip(m["model_file"], m["split"])}
    out = {"train": [], "val": [], "test": []}
    for p in paths:
        split = known.get(p.stem)
        if split is None:
            u = zlib.crc32(f"{seed}:{p.parent.name}/{p.stem}".encode("utf-8")) % 10_000 / 10_000
            split = "test" if u < test else "val" if u < test + val else "train"
        out[split].append(p)
    return out


def load_cloud(path):
    return np.asarray(trimesh.load(path).vertices, dtype=np.float32)


class CompleteClouds(Dataset):
    """Nubes completas, todas cargadas en memoria (16384 × 3 float32 ≈ 0.2 MB cada una)."""

    def __init__(self, paths, n_points=8192, augment=True, scale_range=(0.9, 1.1)):
        self.paths = list(paths)
        self.clouds = [load_cloud(p) for p in self.paths]
        self.n_points, self.augment, self.scale_range = n_points, augment, scale_range

    def __len__(self):
        return len(self.clouds)

    def __getitem__(self, i):
        cloud = torch.from_numpy(self.clouds[i])
        if self.augment:
            idx = torch.randperm(len(cloud))[: self.n_points]
        else:  # validación: siempre el mismo subconjunto de cada nube
            g = torch.Generator().manual_seed(zlib.crc32(str(self.paths[i]).encode("utf-8")))
            idx = torch.randperm(len(cloud), generator=g)[: self.n_points]
        x = cloud[idx]
        if self.augment:
            theta = torch.rand(()) * 2 * torch.pi
            c, s = torch.cos(theta), torch.sin(theta)
            rot = torch.stack([torch.stack([c, torch.zeros(()), s]),
                               torch.tensor([0.0, 1.0, 0.0]),
                               torch.stack([-s, torch.zeros(()), c])])
            lo, hi = self.scale_range
            x = (x @ rot.T) * (lo + (hi - lo) * torch.rand(()))
        return x
