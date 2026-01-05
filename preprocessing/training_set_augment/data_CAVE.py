"""
FUNCIONAMIENTO GENERAL (equivalente MATLAB):

- Hiperparámetros: patchSize, randomNumber, upscale_factor, scales.
- Recorre escenas del dataset CAVE (cada escena: carpeta con PNGs por banda).
- Construye el cubo HSI (H,W,B), normaliza dividiendo entre 65535.
- Para cada escala en scales:
  1) Reescala la imagen (imresize) para generar variación de tamaño.
  2) Selecciona coordenadas aleatorias (randperm) para extraer randomNumber parches HR
     de tamaño imagePatch = patchSize * upscale_factor.
  3) Por cada parche HR, crea aumentaciones:
     - original
     - rotación 180
     - rotación 90
     - rotación 270
     - flip vertical (eje 0 / filas)
  4) Cada label se pasa a data_augment: genera LR por bicúbico (1/upscale) y guarda .mat (lr, hr).

GPU:
- imresize y downsample bicúbico se hacen en GPU con PyTorch si hay CUDA.
- Lectura PNG y guardado .mat se hacen en CPU.
"""

import os
import glob
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.io import savemat

count = 0


def _interp_bicubic_gpu(x_chw: torch.Tensor, scale_factor: float) -> torch.Tensor:
    kwargs = dict(scale_factor=scale_factor, mode="bicubic", align_corners=False)
    try:
        return F.interpolate(x_chw, **kwargs, antialias=(scale_factor < 1))
    except TypeError:
        return F.interpolate(x_chw, **kwargs)


def _imresize_hwc_gpu(img_hwc: np.ndarray, scale: float, device: str) -> np.ndarray:
    x = torch.from_numpy(img_hwc.astype(np.float32)).permute(2, 0, 1).unsqueeze(0).to(device)
    y = _interp_bicubic_gpu(x, scale)
    return y.squeeze(0).permute(1, 2, 0).detach().cpu().numpy().astype(np.float32)


def _read_hsi_from_png_folder(folder: str) -> np.ndarray:
    pngs = sorted(glob.glob(os.path.join(folder, "*.png")))
    if len(pngs) == 0:
        raise FileNotFoundError(f"No se encontraron PNGs en: {folder}")

    bands = []
    for p in pngs:
        im = np.array(Image.open(p))
        if im.ndim == 3:
            im = im[:, :, 0]
        bands.append(im.astype(np.float32))

    cube = np.stack(bands, axis=-1)
    return cube


def data_augment(label_hwc: np.ndarray, upscale_factor: int, savePath: str, device: str):
    global count
    os.makedirs(savePath, exist_ok=True)

    lr_hwc = _imresize_hwc_gpu(label_hwc, 1.0 / float(upscale_factor), device)

    count += 1
    count_name = f"{count:05d}"

    lr = np.transpose(lr_hwc, (2, 0, 1)).astype(np.float32)
    hr = np.transpose(label_hwc, (2, 0, 1)).astype(np.float32)

    savemat(os.path.join(savePath, f"{count_name}.mat"), {"lr": lr, "hr": hr}, do_compression=False)


def generate_cave_train_patches(
    data_type: str = "CAVE",
    upscale_factor: int = 3,
    patchSize: int = 32,
    randomNumber: int = 24,
    scales=(1.0, 0.75, 0.5),
    save_root: str = r"E:\NANLIGONG\OurMethod\dataset\trains",
    srPath: str = r"D:\�߹������ݼ�\CAVE\complete_ms_data",
    device: str | None = None,
    seed: int | None = None,
):
    global count
    count = 0

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if seed is not None:
        np.random.seed(seed)

    imagePatch = patchSize * upscale_factor
    savePath = os.path.join(save_root, data_type, str(upscale_factor))
    os.makedirs(savePath, exist_ok=True)

    scene_names = sorted([d for d in os.listdir(srPath) if os.path.isdir(os.path.join(srPath, d))])

    for idx, name in enumerate(scene_names, start=1):
        print(f"----:{data_type}----upscale_factor:{upscale_factor}----deal with:{idx}----name:{name}")

        singleFile = os.path.join(srPath, name, name)
        t = _read_hsi_from_png_folder(singleFile) / 65535.0
        t = t.astype(np.float32)

        for sc in scales:
            newt = _imresize_hwc_gpu(t, float(sc), device)

            max_x = newt.shape[0] - imagePatch
            max_y = newt.shape[1] - imagePatch
            if max_x <= 0 or max_y <= 0:
                continue

            x_random = np.random.permutation(max_x)[:randomNumber]
            y_random = np.random.permutation(max_y)[:randomNumber]

            for j in range(randomNumber):
                x0 = int(x_random[j])
                y0 = int(y_random[j])

                hrImage = newt[x0:x0 + imagePatch, y0:y0 + imagePatch, :]

                label = hrImage
                data_augment(label, upscale_factor, savePath, device)

                label = np.rot90(hrImage, 2)
                data_augment(label, upscale_factor, savePath, device)

                label = np.rot90(hrImage, 1)
                data_augment(label, upscale_factor, savePath, device)

                label = np.rot90(hrImage, 3)
                data_augment(label, upscale_factor, savePath, device)

                label = np.flip(hrImage, axis=0)
                data_augment(label, upscale_factor, savePath, device)


if __name__ == "__main__":
    generate_cave_train_patches(
        data_type="CAVE",
        upscale_factor=4,
        patchSize=32,
        randomNumber=24,
        scales=(1.0, 0.75, 0.5),
        save_root=r"C:\Users\yered\OneDrive\Documentos\REDES SR\SRDNet\CAVE\datasets\trains",
        srPath=r"C:\Users\yered\OneDrive\Documentos\REDES SR\SRDNet\CAVE\complete_ms_data",
        device=None,
        seed=None,
    )
