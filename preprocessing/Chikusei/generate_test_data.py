"""
FUNCIONAMIENTO GENERAL (equivalente al script MATLAB):

- Lee todos los archivos .mat de la carpeta de prueba (cada uno contiene la variable 'test' con forma HxWxC).
- Para cada archivo:
  1) Genera una versión LR con imresize(test, factor) usando bicúbico (en GPU con PyTorch).
  2) Convierte HR y LR a formato (C, H, W) como permute([3 1 2]).
  3) Guarda cada par (hr, lr) en un .mat individual: Chikusei_test_i.mat (formato v6 compatible).
- Nota: La carga/guardado .mat es CPU (SciPy), el resize es GPU (PyTorch).
"""

import os
import glob
import numpy as np
import torch
import torch.nn.functional as F
from scipy.io import loadmat, savemat


def _interp_bicubic(x: torch.Tensor, scale_factor: float) -> torch.Tensor:
    kwargs = dict(scale_factor=scale_factor, mode="bicubic", align_corners=False)
    try:
        return F.interpolate(x, **kwargs, antialias=(scale_factor < 1))
    except TypeError:
        return F.interpolate(x, **kwargs)


def generate_chikusei_tests_gpu(
    in_folder: str,
    out_folder: str,
    factor: float = 0.125,
    device: str | None = None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(out_folder, exist_ok=True)

    file_list = sorted(glob.glob(os.path.join(in_folder, "*.mat")))
    if len(file_list) == 0:
        raise FileNotFoundError(f"No se encontraron .mat en: {in_folder}")

    for i, fpath in enumerate(file_list, start=1):
        d = loadmat(fpath)
        if "test" not in d:
            raise KeyError(f"El archivo no contiene la variable 'test': {fpath}")

        test = d["test"].astype(np.float32)  # (H,W,C)

        hr_t = torch.from_numpy(test).permute(2, 0, 1).unsqueeze(0).to(device)  # (1,C,H,W)
        lr_t = _interp_bicubic(hr_t, factor)                                    # (1,C,h,w)

        hr = hr_t.squeeze(0).detach().cpu().numpy().astype(np.float32)          # (C,H,W)
        lr = lr_t.squeeze(0).detach().cpu().numpy().astype(np.float32)          # (C,h,w)

        out_path = os.path.join(out_folder, f"Chikusei_test_{i}.mat")
        savemat(out_path, {"hr": hr, "lr": lr}, do_compression=False)

    return len(file_list)


if __name__ == "__main__":
    in_folder = r"E:\�㷨\mcodes\test"
    out_folder = r"E:\�㷨\mcodes\dataset\tests"
    generate_chikusei_tests_gpu(in_folder, out_folder, factor=0.125)
