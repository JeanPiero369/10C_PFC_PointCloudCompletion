"""Point Transformer V3 tal cual, cargado desde la carpeta "Point Transformer v3".

La carpeta tiene espacios en el nombre, así que no se puede importar como paquete
normal; se carga con importlib bajo el nombre "ptv3" para que su import relativo
(`from .serialization import encode`) funcione.

Único cambio en model.py: PDNorm toma el contexto por nube, `point.context[point.batch]`,
para poder modular cada normalización con el tiempo (ver flow_model.py).
"""

import importlib.util
import sys
from pathlib import Path

PTV3_DIR = Path(__file__).resolve().parents[1] / "Point Transformer v3"
PDNORM_CONTEXT = 256  # tamaño del contexto de PDNorm: PTv3 no lo expone en su constructor y queda en 256

# configuración original de PTv3, con la ruta de atención sin flash-attn (no hay flash-attn en Windows)
PTV3_DEFAULTS = dict(
    order=("z", "z-trans", "hilbert", "hilbert-trans"),
    stride=(2, 2, 2, 2),
    enc_depths=(2, 2, 2, 6, 2),
    enc_channels=(32, 64, 128, 256, 512),
    enc_num_head=(2, 4, 8, 16, 32),
    enc_patch_size=(1024, 1024, 1024, 1024, 1024),
    dec_depths=(2, 2, 2, 2),
    dec_channels=(64, 64, 128, 256),
    dec_num_head=(4, 4, 8, 16),
    dec_patch_size=(1024, 1024, 1024, 1024),
    mlp_ratio=4,
    drop_path=0.3,
    enable_flash=False,
    upcast_attention=True,
    upcast_softmax=True,
)


def load_ptv3_module():
    """El módulo model.py de PTv3 (se carga una sola vez por proceso)."""
    if "ptv3" in sys.modules:
        return sys.modules["ptv3"]
    spec = importlib.util.spec_from_file_location(
        "ptv3", PTV3_DIR / "model.py", submodule_search_locations=[str(PTV3_DIR)])
    module = importlib.util.module_from_spec(spec)
    sys.modules["ptv3"] = module
    spec.loader.exec_module(module)
    return module


def build_ptv3(in_channels, **overrides):
    """PointTransformerV3 con la configuración original y los cambios pedidos en `overrides`."""
    kwargs = dict(PTV3_DEFAULTS)
    kwargs.update(overrides)
    return load_ptv3_module().PointTransformerV3(in_channels=in_channels, **kwargs)
