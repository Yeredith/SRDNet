"""
FUNCIONAMIENTO GENERAL (equivalente MATLAB, Harvard train patches):

- Hiperparámetros: patchSize, randomNumber, upscale_factor, scales y probabilidad P.
- Recorre todos los .mat del dataset Harvard (cada archivo contiene 'ref').
- Para cada imagen:
  1) Normaliza: t = ref / max(ref).
  2) Para j=1..randomNumber y para cada escala en scales:
     - Reescala la imagen (imresize) en GPU.
     - Genera x_random e y_random (randperm) y toma el parche #j de tamaño imagePatch.
     - Aumenta datos guardando siempre el parche original.
     - Con probabilidad (rand > P) aplica rotaciones y flips, guardando cada aumento.
  3) Cada aumento se guarda con data_augment: crea LR con bicúbico (1/upscale) en GPU
     y guarda .mat (lr, hr) con contador global en savePath.

GPU:
- imresize (escalado) y downsample bicúbico se hacen en GPU con PyTorch si hay CUDA.
- Carga .mat y guardado .mat se hace en CPU.
"""

import os
import glob
import numpy as np
import torch
import torch.nn.functional as F
from scipy.io import loadmat, savemat

count = 0


def _interp_bicubic_gpu(x_chw: torch.Tensor, scale_factor: float) -> torch.Tensor:
    kwargs = dict(scale_factor=scale_factor, mode="bicubic", align_corners=False)
    try:
        return F.interpolate(x_chw, **kwargs, antialias=(scale_factor < 1))
    except TypeError:
        return F.interpolate(x_chw, **kwargs)


def _imresize_hwc_gpu(img_hwc: np.ndarray, scale: float, device: str) -> np.ndarray:
    x = torch.from_numpy(img_hwc.astype(np.float32)).permute(2, 0, 1).unsqueeze(0).to(device)
    y = _interp_bicubic_gpu(x, float(scale))
    return y.squeeze(0).permute(1, 2, 0).detach().cpu().numpy().astype(np.float32)


def data_augment(label_hwc: np.ndarray, upscale_factor: int, savePath: str, device: str):
    global count
    os.makedirs(savePath, exist_ok=True)

    lr_hwc = _imresize_hwc_gpu(label_hwc, 1.0 / float(upscale_factor), device)

    count += 1
    count_name = f"{count:05d}"

    lr = np.transpose(lr_hwc, (2, 0, 1)).astype(np.float32)
    hr = np.transpose(label_hwc, (2, 0, 1)).astype(np.float32)

    savemat(os.path.join(savePath, f"{count_name}.mat"), {"lr": lr, "hr": hr}, do_compression=False)


def generate_harvard_train_patches(
    data_type: str = "Hararvd",
    upscale_factor: int = 4,
    patchSize: int = 32,
    randomNumber: int = 32,
    P: float = 0.5,
    scales=(1.0, 0.75, 0.5),
    save_root: str = r"D:\Users\LTT\Desktop\11",
    srPath: str = r"D:\�߹������ݼ�\Harvard\CZ_hsdbi",
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

    file_list = sorted(glob.glob(os.path.join(srPath, "*.mat")))
    print("length(fileNames) =", len(file_list))

    for i, fpath in enumerate(file_list, start=1):
        name = os.path.basename(fpath)
        print(f"----:{data_type}----upscale_factor:{upscale_factor}----deal with:{i}----name:{name}")

        d = loadmat(fpath)
        if "ref" not in d:
            raise KeyError(f"No se encontró 'ref' en: {fpath}")

        ref = d["ref"].astype(np.float32)

        mx = float(np.max(ref))
        if mx == 0:
            continue
        t = ref / mx

        for j in range(1, randomNumber + 1):
            for sc in scales:
                newt = _imresize_hwc_gpu(t, float(sc), device)

                max_x = newt.shape[0] - imagePatch
                max_y = newt.shape[1] - imagePatch
                if max_x <= 0 or max_y <= 0:
                    continue

                x_random = np.random.permutation(max_x)[:randomNumber]
                y_random = np.random.permutation(max_y)[:randomNumber]

                x0 = int(x_random[j - 1])
                y0 = int(y_random[j - 1])

                hrImage = newt[x0:x0 + imagePatch, y0:y0 + imagePatch, :]

                label = hrImage
                data_augment(label, upscale_factor, savePath, device)

                if np.random.rand() > P:
                    label = np.rot90(hrImage, 2)
                    data_augment(label, upscale_factor, savePath, device)

                if np.random.rand() > P:
                    label = np.rot90(hrImage, 1)
                    data_augment(label, upscale_factor, savePath, device)

                if np.random.rand() > P:
                    label = np.rot90(hrImage, 3)
                    data_augment(label, upscale_factor, savePath, device)

                if np.random.rand() > P:
                    label = np.flip(hrImage, axis=0)
                    data_augment(label, upscale_factor, savePath, device)

                if np.random.rand() > P:
                    label = np.flip(hrImage, axis=1)
                    data_augment(label, upscale_factor, savePath, device)


if __name__ == "__main__":
    generate_harvard_train_patches(
        data_type="Hararvd",
        upscale_factor=4,
        patchSize=32,
        randomNumber=32,
        P=0.5,
        scales=(1.0, 0.75, 0.5),
        save_root=r"D:\Users\LTT\Desktop\11",
        srPath=r"D:\�߹������ݼ�\Harvard\CZ_hsdbi",
        device=None,
        seed=None,
    )
