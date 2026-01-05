import os
import torch
import numpy as np
import torch.utils.data as data

from os import listdir
from os.path import join
import scipy.io as scio
import numpy as np
from torchvision import transforms


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.mat'])
       
class TrainsetFromFolder(data.Dataset):
    def __init__(self, dataset_dir):
        super(TrainsetFromFolder, self).__init__()
        if not os.path.exists(dataset_dir):
            raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}.\nPlease create the folder and put your .mat files there, e.g. dataset/trains/CAVE/3/")
        self.image_filenames = [join(dataset_dir , x) for x in listdir(dataset_dir ) if is_image_file(x)]

    def __getitem__(self, index):
        mat = scio.loadmat(self.image_filenames[index], verify_compressed_data_integrity=False)

        # Possible variable names in .mat files: 'lr'/'hr' (lowercase from crop_image)
        # or 'LR'/'HR' (uppercase from other scripts). Try both.
        def _find_key(m, keys):
            for k in keys:
                if k in m:
                    return k
            return None

        lr_key = _find_key(mat, ['lr', 'LR', 'LR', 'LRs', 'LR_image'])
        hr_key = _find_key(mat, ['hr', 'HR', 'HRs', 'HR_image', 'hsi', 'ref'])

        if lr_key is None or hr_key is None:
            # Try common alternatives
            possible = list(mat.keys())
            # fallback: pick first two 3D arrays
            arrays = [k for k in possible if isinstance(mat[k], np.ndarray) and mat[k].ndim == 3]
            if len(arrays) >= 2:
                lr_key, hr_key = arrays[0], arrays[1]
            else:
                raise KeyError(f"Could not find LR/HR arrays in {self.image_filenames[index]}")

        input = mat[lr_key]
        label = mat[hr_key]

        # Ensure float32
        input = input.astype(np.float32)
        label = label.astype(np.float32)

        # If keys were uppercase or typical full-image variables, they are likely (H, W, C).
        # If keys are lowercase (from our crop_image), they are likely already (C, H, W).
        def to_chw(arr, key_name):
            if arr.ndim != 3:
                raise ValueError(f"Array {key_name} is not 3D")
            # Heuristic: if first dim is much larger than third dim, assume (H,W,C)
            H, W, C = arr.shape
            if key_name.isupper():
                # uppercase names from older scripts -> (H,W,C)
                return np.transpose(arr, (2, 0, 1)).astype(np.float32)
            else:
                # lowercase likely already (C,H,W) or (H,W,C). If first dim equals typical band count (<100)
                # assume it's (C,H,W); otherwise if first dim > third dim assume (H,W,C)
                if H > C and W > C:
                    return np.transpose(arr, (2, 0, 1)).astype(np.float32)
                else:
                    return arr.astype(np.float32)

        input = to_chw(input, lr_key)
        label = to_chw(label, hr_key)

        return torch.from_numpy(input).float(), torch.from_numpy(label).float()

        # return input, label

        
    def __len__(self):
        return len(self.image_filenames)
     
class ValsetFromFolder(data.Dataset):
    def __init__(self, dataset_dir):
        super(ValsetFromFolder, self).__init__()
        if not os.path.exists(dataset_dir):
            raise FileNotFoundError(f"Validation dataset directory not found: {dataset_dir}.\nPlease create the folder and put your .mat files there, e.g. dataset/evals/CAVE/3/")
        self.image_filenames = [join(dataset_dir , x) for x in listdir(dataset_dir ) if is_image_file(x)]

    def __getitem__(self, index):
        mat = scio.loadmat(self.image_filenames[index], verify_compressed_data_integrity=False)

        # Reuse the same logic as TrainsetFromFolder to pick keys and normalize to (C,H,W)
        def _find_key(m, keys):
            for k in keys:
                if k in m:
                    return k
            return None

        lr_key = _find_key(mat, ['lr', 'LR', 'LRs', 'LR_image'])
        hr_key = _find_key(mat, ['hr', 'HR', 'HRs', 'HR_image', 'hsi', 'ref', 'test'])

        if lr_key is None or hr_key is None:
            possible = list(mat.keys())
            arrays = [k for k in possible if isinstance(mat[k], np.ndarray) and mat[k].ndim == 3]
            if len(arrays) >= 2:
                lr_key, hr_key = arrays[0], arrays[1]
            else:
                raise KeyError(f"Could not find LR/HR arrays in {self.image_filenames[index]}")

        input = mat[lr_key].astype(np.float32)
        label = mat[hr_key].astype(np.float32)

        def to_chw(arr, key_name):
            H, W, C = arr.shape
            if key_name.isupper():
                return np.transpose(arr, (2, 0, 1)).astype(np.float32)
            if H > C and W > C:
                return np.transpose(arr, (2, 0, 1)).astype(np.float32)
            return arr.astype(np.float32)

        input = to_chw(input, lr_key)
        label = to_chw(label, hr_key)

        return torch.from_numpy(input).float(), torch.from_numpy(label).float()



    def __len__(self):
        return len(self.image_filenames)



