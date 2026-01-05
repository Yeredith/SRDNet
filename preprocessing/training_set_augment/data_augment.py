"""
FUNCIONAMIENTO GENERAL (equivalente MATLAB data_augment):

- Recibe una imagen HR (label) en formato (H, W, C).
- Genera la versión LR aplicando imresize(label, 1/upscale_factor, 'bicubic').
- Incrementa un contador global y genera un nombre con 5 dígitos (00001, 00002, …).
- Reordena dimensiones de (H, W, C) a (C, H, W) usando permute.
- Convierte HR y LR a float32 (single).
- Guarda cada par (lr, hr) en un archivo .mat dentro de savePath.

El resize se realiza en GPU (PyTorch) si hay CUDA disponible.
El guardado .mat se realiza en CPU (SciPy), igual que en MATLAB.
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
from scipy.io import savemat

# contador global (equivalente a "global count" en MATLAB)
count = 0


def _bicubic_resize_gpu(img_hwc: np.ndarray, scale_factor: float, device: str) -> np.ndarray:
    x = torch.from_numpy(img_hwc.astype(np.float32)).permute(2, 0, 1).unsqueeze(0).to(device)
    kwargs = dict(scale_factor=scale_factor, mode="bicubic", align_corners=False)
    try:
        y = F.interpolate(x, **kwargs, antialias=(scale_factor < 1))
    except TypeError:
        y = F.interpolate(x, **kwargs)
    return y.squeeze(0).permute(1, 2, 0).detach().cpu().numpy().astype(np.float32)


def data_augment(label, upscale_factor, savePath, device=None):
    global count

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(savePath, exist_ok=True)

    input_lr = _bicubic_resize_gpu(label, 1.0 / float(upscale_factor), device)

    count += 1
    count_name = f"{count:05d}"

    lr = np.transpose(input_lr, (2, 0, 1)).astype(np.float32)
    hr = np.transpose(label, (2, 0, 1)).astype(np.float32)

    savemat(os.path.join(savePath, f"{count_name}.mat"), {"lr": lr, "hr": hr}, do_compression=False)

    return lr, hr
