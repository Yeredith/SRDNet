"""
FUNCIONAMIENTO GENERAL (equivalente MATLAB):

- Recorre todas las carpetas de escenas del dataset CAVE (cada escena tiene 31 PNGs, uno por banda).
- Para cada escena:
  1) Lee todas las bandas .png y arma un cubo hiperespectral HR (H,W,B).
  2) Normaliza dividiendo entre 65535.
  3) Aplica modcrop para que (H,W) sean múltiplos de 'upscale'.
  4) Genera LR = imresize(HR, 1/upscale, 'bicubic') (resize en GPU con PyTorch si hay CUDA).
  5) Guarda HR y LR en .mat dentro de savePath.
"""

import os
import glob
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.io import savemat


def modcrop(img_hwc: np.ndarray, scale: int) -> np.ndarray:
    h, w, c = img_hwc.shape
    h2 = h - (h % scale)
    w2 = w - (w % scale)
    return img_hwc[:h2, :w2, :]


def _bicubic_resize_gpu(img_hwc: np.ndarray, scale_factor: float, device: str) -> np.ndarray:
    x = torch.from_numpy(img_hwc.astype(np.float32)).permute(2, 0, 1).unsqueeze(0).to(device)  # (1,C,H,W)
    kwargs = dict(scale_factor=scale_factor, mode="bicubic", align_corners=False)
    try:
        y = F.interpolate(x, **kwargs, antialias=(scale_factor < 1))
    except TypeError:
        y = F.interpolate(x, **kwargs)
    out = y.squeeze(0).permute(1, 2, 0).detach().cpu().numpy().astype(np.float32)  # (H,W,C)
    return out


def build_hsi_from_pngs(scene_dir: str) -> np.ndarray:
    pngs = sorted(glob.glob(os.path.join(scene_dir, "*.png")))
    if len(pngs) == 0:
        raise FileNotFoundError(f"No se encontraron PNGs en: {scene_dir}")

    bands = []
    for p in pngs:
        im = np.array(Image.open(p))
        if im.ndim == 3:
            im = im[:, :, 0]
        bands.append(im.astype(np.float32))

    h, w = bands[0].shape
    B = len(bands)
    cube = np.stack(bands, axis=-1)  # (H,W,B)
    if cube.shape[0] != h or cube.shape[1] != w:
        raise ValueError("Inconsistencia de tamaños entre bandas PNG.")
    return cube


def generate_cave_tests(
    dataset: str = "CAVE",
    upscale: int = 3,
    save_root: str = r"C:\Users\yered\OneDrive\Documentos\REDES SR\SRDNet\CAVE\datasets\tests",
    sr_path: str = r"C:\Users\yered\OneDrive\Documentos\REDES SR\SRDNet\CAVE\complete_ms_data",
    device: str | None = None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    save_path = os.path.join(save_root, dataset, str(upscale))
    os.makedirs(save_path, exist_ok=True)

    scene_names = sorted([d for d in os.listdir(sr_path) if os.path.isdir(os.path.join(sr_path, d))])
    print("number =", len(scene_names))

    for idx, name in enumerate(scene_names, start=1):
        print(f"-----deal with: {idx} ---- name: {name}")

        single_path = os.path.join(sr_path, name, name)
        hr_raw = build_hsi_from_pngs(single_path)  # (H,W,B)

        hr_raw = hr_raw / 65535.0
        hr_raw = hr_raw.astype(np.float32)

        hr = modcrop(hr_raw, upscale)
        lr = _bicubic_resize_gpu(hr, 1.0 / float(upscale), device)

        out_path = os.path.join(save_path, f"{name}.mat")
        savemat(out_path, {"HR": hr, "LR": lr}, do_compression=False)


if __name__ == "__main__":
    generate_cave_tests(
        dataset="CAVE",
        upscale=2,
        save_root=r"C:\Users\yered\OneDrive\Documentos\REDES SR\SRDNet\CAVE\datasets\tests",
        sr_path=r"C:\Users\yered\OneDrive\Documentos\REDES SR\SRDNet\CAVE\complete_ms_data",
    )
