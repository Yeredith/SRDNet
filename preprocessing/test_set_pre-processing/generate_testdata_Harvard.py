"""
FUNCIONAMIENTO GENERAL (equivalente MATLAB, Harvard):

- Recorre todos los archivos .mat en la carpeta Harvard (cada uno contiene 'ref' y 'lbl').
- Para cada archivo:
  1) Carga 'ref' como cubo hiperespectral (H,W,31) y descarta 'lbl'.
  2) Normaliza por el máximo global (data / max(data)).
  3) Recorta a 512x512.
  4) Aplica modcrop para que (H,W) sean múltiplos de 'upscale'.
  5) Genera LR = imresize(HR, 1/upscale, 'bicubic') (resize en GPU con PyTorch si hay CUDA).
  6) Guarda HR y LR en savePath con el mismo nombre base.
"""

import os
import glob
import numpy as np
import torch
import torch.nn.functional as F
from scipy.io import loadmat, savemat


def modcrop(img_hwc: np.ndarray, scale: int) -> np.ndarray:
    h, w, c = img_hwc.shape
    h2 = h - (h % scale)
    w2 = w - (w % scale)
    return img_hwc[:h2, :w2, :]


def _bicubic_resize_gpu(img_hwc: np.ndarray, scale_factor: float, device: str) -> np.ndarray:
    x = torch.from_numpy(img_hwc.astype(np.float32)).permute(2, 0, 1).unsqueeze(0).to(device)
    kwargs = dict(scale_factor=scale_factor, mode="bicubic", align_corners=False)
    try:
        y = F.interpolate(x, **kwargs, antialias=(scale_factor < 1))
    except TypeError:
        y = F.interpolate(x, **kwargs)
    return y.squeeze(0).permute(1, 2, 0).detach().cpu().numpy().astype(np.float32)


def generate_harvard_tests(
    dataset: str = "Hararvd",
    upscale: int = 4,
    save_root: str = r"D:\Users\LTT\Desktop\11",
    sr_path: str = r"D:\�߹������ݼ�\Harvard\CZ_hsdbi",
    device: str | None = None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    save_path = os.path.join(f"{save_root}{dataset}", str(upscale))
    os.makedirs(save_path, exist_ok=True)

    mat_files = sorted(glob.glob(os.path.join(sr_path, "*.mat")))
    print("number =", len(mat_files))

    for idx, fpath in enumerate(mat_files, start=1):
        name = os.path.basename(fpath)
        print(f"-----deal with: {idx} ---- name: {name}")

        d = loadmat(fpath)
        if "ref" not in d:
            raise KeyError(f"No se encontró 'ref' en: {fpath}")

        data = d["ref"].astype(np.float32)

        mx = float(np.max(data))
        if mx == 0:
            raise ValueError(f"Max=0 en {fpath}, no se puede normalizar.")
        data = data / mx

        data = data[:512, :512, :]

        hr = modcrop(data, upscale)
        lr = _bicubic_resize_gpu(hr, 1.0 / float(upscale), device)

        out_path = os.path.join(save_path, name)
        savemat(out_path, {"HR": hr, "LR": lr}, do_compression=False)


if __name__ == "__main__":
    generate_harvard_tests(
        dataset="Hararvd",
        upscale=4,
        save_root=r"D:\Users\LTT\Desktop\11",
        sr_path=r"D:\�߹������ݼ�\Harvard\CZ_hsdbi",
    )
