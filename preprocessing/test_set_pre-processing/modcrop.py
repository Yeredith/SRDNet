"""
FUNCIONAMIENTO GENERAL (equivalente MATLAB modcrop):

- Recorta una imagen para que sus dimensiones espaciales (alto y ancho)
  sean múltiplos exactos de un valor dado (modulo).
- Si la imagen es 2D (una sola banda), recorta solo H y W.
- Si la imagen es 3D (H, W, C), recorta H y W y conserva todas las bandas.
- Se usa antes de downsampling para evitar inconsistencias de tamaño.
"""

import numpy as np


def modcrop(imgs: np.ndarray, modulo: int) -> np.ndarray:
    if imgs.ndim == 2 or (imgs.ndim == 3 and imgs.shape[2] == 1):
        h, w = imgs.shape[:2]
        h = h - (h % modulo)
        w = w - (w % modulo)
        return imgs[:h, :w]
    else:
        h, w = imgs.shape[:2]
        h = h - (h % modulo)
        w = w - (w % modulo)
        return imgs[:h, :w, :]
