import os
import numpy as np
from scipy.io import loadmat

# -----------------------------
# Configuración
# -----------------------------
dataset = 'CAVE'

src_path = r'E:\算法\MCNet-master\dataset\trains\{}\4'.format(dataset)

# Obtener archivos .mat
file_names = [f for f in os.listdir(src_path) if f.endswith('.mat')]
print(len(file_names))

band_mean_list = []

# -----------------------------
# Procesamiento
# -----------------------------
for i, name in enumerate(file_names, start=1):
    print(f'----- deal with: {i} ---- name: {name}')

    data_path = os.path.join(src_path, name)
    data = loadmat(data_path)

    # Se asume que el archivo contiene la variable 'hsi'
    hsi = data['hsi']     # shape: (H, W, B)

    H, W, B = hsi.shape

    # Equivalente a reshape + mean en MATLAB
    band_mean = np.mean(hsi.reshape(H * W, B), axis=0)
    band_mean_list.append(band_mean)

# -----------------------------
# Promedio final
# -----------------------------
band_mean = np.mean(np.stack(band_mean_list, axis=0), axis=0)

print("Band mean shape:", band_mean.shape)
print("Band mean:", band_mean)
