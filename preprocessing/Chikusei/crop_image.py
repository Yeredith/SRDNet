"""
Equivalente GPU del MATLAB crop_image(img, patch_size, stride, factor, file_name):

1) img (H,W,C) -> tensor (1,C,H,W) en GPU
2) Extrae todos los parches HR con unfold (mismo barrido: up lento, left rápido)
3) ms = imresize(gt, factor) con bicúbico (y antialias cuando factor<1 si está disponible)
4) lr/hr quedan como (C,H,W) float32 y se guardan en .mat individuales
5) out = número total de parches
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
from scipy.io import savemat


def crop_image_gpu(img, patch_size, stride, factor, file_name, out_dir="./2", device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    img = img.astype(np.float32)
    H, W, C = img.shape

    x = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)  # (1,C,H,W)

    patches = F.unfold(x, kernel_size=patch_size, stride=stride)         # (1, C*p*p, L)
    L = patches.shape[-1]

    hr = patches.transpose(1, 2).contiguous().view(L, C, patch_size, patch_size)  # (L,C,p,p)

    interp_kwargs = dict(scale_factor=factor, mode="bicubic", align_corners=False)
    try:
        lr = F.interpolate(hr, **interp_kwargs, antialias=(factor < 1))
    except TypeError:
        lr = F.interpolate(hr, **interp_kwargs)

    os.makedirs(out_dir, exist_ok=True)

    for k in range(L):
        lr_np = lr[k].detach().cpu().numpy().astype(np.float32)  # (C, h_lr, w_lr)
        hr_np = hr[k].detach().cpu().numpy().astype(np.float32)  # (C, p, p)
        file_path = os.path.join(out_dir, f"block_{file_name}_{k + 1}.mat")
        savemat(file_path, {"lr": lr_np, "hr": hr_np}, do_compression=False)

    return int(L)
