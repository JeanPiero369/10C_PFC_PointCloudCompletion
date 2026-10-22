"""Flow matching sobre nubes de puntos de vasijas con Point Transformer V3 como backbone.

Formulación (generativa, incondicional):
  x0 ~ N(0, I)            ruido, N × 3
  x1 ~ p_data             vasija completa (datasets/generated/points_complete), N × 3
  x_t = (1 - t) x0 + t x1 camino recto (rectified flow)
  u_t = x1 - x0           velocidad objetivo
  pérdida = || v_θ(x_t, t) - u_t ||²

La red v_θ (`FlowPTv3`) ve solo los N puntos ruidosos y el tiempo t. El tiempo modula
todas las normalizaciones de PTv3 (como AdaLN en DiT) y la cabeza (FiLM). En inferencia
se integra dx/dt = v_θ(x, t) de t = 0 a t = 1 con Euler.

Módulos:
  backbone.py    carga PTv3 desde "Point Transformer v3/" sin modificarlo
  embeddings.py  embedding de tiempo y modulación FiLM
  flow_model.py  FlowPTv3: entrada, vóxeles, PTv3, cabeza
  cfm.py         camino, muestreo de t, pérdida
  sampler.py     integrador de Euler (con trayectoria opcional)
  data.py        dataset de nubes completas y split por vasija
  metrics.py     Chamfer, EMD, MMD / COV / 1-NNA
  train.py       entrenamiento
  sample.py      muestreo y evaluación

Requiere el entorno "Python (PFC)": torch 2.11 + cu128, spconv-cu126, torch_scatter, addict, timm.
"""
