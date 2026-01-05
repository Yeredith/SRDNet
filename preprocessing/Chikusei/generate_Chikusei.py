"""
DEMO (Python) para generar muestras de entrenamiento y prueba desde un HSI (Chikusei),
equivalente al script MATLAB que compartiste.

RESUMEN DE LO QUE HACE:
1) Carga el .mat de Chikusei, recorta al centro (2304x2048), normaliza por el máximo global y guarda:
   - ./test/Chikusei_test_i.mat (parches 512x512 de la primera franja/“fila” superior)
   - ./train/Chikusei_train.mat (el resto para entrenamiento)
2) Genera datos de prueba para mains.py (pares LR/HR) en carpetas ./test_x{scale}
3) Genera muestras de entrenamiento (patches) desde ./train/Chikusei_train.mat en ./train_samples_x{scale}
4) (Manual) mover ~10% de las muestras a ./evals

NOTAS:
- El procesamiento pesado (recorte y resize) se hace en GPU (si hay CUDA).
- El guardado .mat se hace en CPU (SciPy).
- Si tu .mat es v7.3 (HDF5), se carga con h5py automáticamente.
"""

import os
import math
import glob
import numpy as np
import torch
import torch.nn.functional as F
from scipy.io import loadmat, savemat

try:
    import h5py
except ImportError:
    h5py = None


def _load_mat_var(mat_path: str, var_name: str):
    try:
        d = loadmat(mat_path)
        if var_name in d:
            return d[var_name]
    except NotImplementedError:
        pass

    if h5py is None:
        raise RuntimeError("El archivo parece ser .mat v7.3 (HDF5) pero h5py no está instalado.")

    with h5py.File(mat_path, "r") as f:
        if var_name not in f:
            raise KeyError(f"No se encontró la variable '{var_name}' en {mat_path}.")
        x = f[var_name][()]
        x = np.array(x)
        if x.ndim == 3 and x.shape[0] < 10 and x.shape[1] > 1000:
            x = np.transpose(x, (2, 1, 0))
        return x


def _to_tensor_chw(img_hwc: np.ndarray, device: str):
    return torch.from_numpy(img_hwc.astype(np.float32)).permute(2, 0, 1).unsqueeze(0).to(device)


def _interp_bicubic(x: torch.Tensor, scale_factor: float):
    kwargs = dict(scale_factor=scale_factor, mode="bicubic", align_corners=False)
    try:
        return F.interpolate(x, **kwargs, antialias=(scale_factor < 1))
    except TypeError:
        return F.interpolate(x, **kwargs)


def step1_generate_train_test_from_hsi(
    chikusei_mat_path: str,
    var_name: str = "chikusei",
    test_img_size: int = 512,
    out_test_dir: str = "./test",
    out_train_dir: str = "./train",
):
    os.makedirs(out_test_dir, exist_ok=True)
    os.makedirs(out_train_dir, exist_ok=True)

    chikusei = _load_mat_var(chikusei_mat_path, var_name)

    img = chikusei[106:2410, 143:2191, :]
    del chikusei

    mx = np.max(img)
    if mx == 0:
        raise ValueError("El máximo global es 0, no se puede normalizar.")
    img = (img / mx).astype(np.float32)

    H, W, C = img.shape
    test_pic_num = W // test_img_size

    for i in range(1, test_pic_num + 1):
        left = (i - 1) * test_img_size
        right = left + test_img_size
        test = img[:test_img_size, left:right, :]
        savemat(os.path.join(out_test_dir, f"Chikusei_test_{i}.mat"), {"test": test}, do_compression=False)

    train_img = img[test_img_size:, :, :]
    savemat(os.path.join(out_train_dir, "Chikusei_train.mat"), {"img": train_img}, do_compression=False)


def generate_test_data(
    scales=(2, 4),
    test_dir: str = "./test",
    out_root: str = ".",
    device: str = None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    test_files = sorted(glob.glob(os.path.join(test_dir, "*.mat")))
    if len(test_files) == 0:
        raise FileNotFoundError(f"No hay .mat en {test_dir}")

    for s in scales:
        out_dir = os.path.join(out_root, f"test_x{s}")
        os.makedirs(out_dir, exist_ok=True)

        factor = 1.0 / float(s)

        for fpath in test_files:
            name = os.path.splitext(os.path.basename(fpath))[0]
            d = loadmat(fpath)
            if "test" not in d:
                raise KeyError(f"'{fpath}' no contiene la variable 'test'")
            hr_hwc = d["test"].astype(np.float32)

            x = _to_tensor_chw(hr_hwc, device)          # (1,C,H,W)
            lr = _interp_bicubic(x, factor)             # (1,C,h,w)

            lr_np = lr.squeeze(0).detach().cpu().numpy().astype(np.float32)
            hr_np = x.squeeze(0).detach().cpu().numpy().astype(np.float32)

            savemat(
                os.path.join(out_dir, f"{name}_x{s}.mat"),
                {"lr": lr_np, "hr": hr_np},
                do_compression=False
            )


def generate_train_data(
    scales=(2, 4),
    train_mat_path: str = "./train/Chikusei_train.mat",
    patch_size: int = 64,
    stride: int = 32,
    out_root: str = ".",
    device: str = None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    d = loadmat(train_mat_path)
    if "img" not in d:
        raise KeyError(f"'{train_mat_path}' no contiene la variable 'img'")

    img_hwc = d["img"].astype(np.float32)
    H, W, C = img_hwc.shape

    x = _to_tensor_chw(img_hwc, device)  # (1,C,H,W)

    for s in scales:
        out_dir = os.path.join(out_root, f"train_samples_x{s}")
        os.makedirs(out_dir, exist_ok=True)

        factor = 1.0 / float(s)

        patches = F.unfold(x, kernel_size=patch_size, stride=stride)  # (1, C*p*p, L)
        L = patches.shape[-1]
        hr = patches.transpose(1, 2).contiguous().view(L, C, patch_size, patch_size)

        lr = _interp_bicubic(hr, factor)  # (L,C,h,w)

        for k in range(L):
            lr_np = lr[k].detach().cpu().numpy().astype(np.float32)
            hr_np = hr[k].detach().cpu().numpy().astype(np.float32)
            savemat(
                os.path.join(out_dir, f"block_Chikusei_train_{k+1}.mat"),
                {"lr": lr_np, "hr": hr_np},
                do_compression=False
            )


def main():
    chikusei_mat_path = r"F:\高光谱数据集\Hyperspec_Chikusei_MATLAB\Chikusei_MATLAB\HyperspecVNIR_Chikusei_20140729.mat"
    step1_generate_train_test_from_hsi(chikusei_mat_path, var_name="chikusei")

    generate_test_data(scales=(2, 4), test_dir="./test", out_root=".")
    generate_train_data(scales=(2, 4), train_mat_path="./train/Chikusei_train.mat", patch_size=64, stride=32, out_root=".")


if __name__ == "__main__":
    main()
