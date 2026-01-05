"""
FUNCIONAMIENTO GENERAL:

- Busca todos los archivos .mat dentro de ./train
- Para cada archivo:
  1) Carga la variable 'img' (HSI) desde el .mat
  2) Obtiene el nombre base del archivo (sin .mat)
  3) Genera parches usando crop_image_gpu con:
     patch_size=64, stride=32, factor=0.25
  4) Guarda cada parche como .mat en ./2 con nombre block_<name>_<idx>.mat
"""

import os
import glob
from crop_image import crop_image_gpu
from scipy.io import loadmat

# Importa tu función GPU (la que ya te pasé)
# from crop_image_gpu import crop_image_gpu


def convert_hs_dataset_to_patches_gpu(train_dir="./train", patch_size=64, stride=32, factor=0.25, out_dir="./2"):
    file_list = sorted(glob.glob(os.path.join(train_dir, "*.mat")))
    if len(file_list) == 0:
        raise FileNotFoundError(f"No se encontraron .mat en: {train_dir}")

    for fpath in file_list:
        base = os.path.splitext(os.path.basename(fpath))[0]
        d = loadmat(fpath)

        if "img" not in d:
            raise KeyError(f"El archivo {fpath} no contiene la variable 'img'")

        img = d["img"]
        crop_image_gpu(img, patch_size, stride, factor, base, out_dir=out_dir)


if __name__ == "__main__":
    convert_hs_dataset_to_patches_gpu(train_dir="./train", patch_size=64, stride=32, factor=0.25, out_dir="./2")
