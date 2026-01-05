# Entorno de prueba para SRDNet

Estos son los pasos recomendados para crear un entorno aislado y probar el entrenamiento de la red.

1) Crear el entorno desde `environment.yml` (conda):

```powershell
conda env create -f environment.yml
conda activate srdnet-test
```

2) Instalar PyTorch (elige la línea según tu hardware). Visita https://pytorch.org/get-started/locally/ para la línea exacta.

Ejemplo CUDA 11.8:
```powershell
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
```

Ejemplo CPU-only:
```powershell
conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
```

3) (Alternativa con pip) Si prefieres usar `requirements.txt` en lugar de `environment.yml`:

```powershell
conda create -n srdnet-test python=3.10 -y
conda activate srdnet-test
pip install -r requirements.txt
# luego instalar PyTorch como paso anterior
```

4) Verificar instalaciones:

```powershell
python -c "import torch, tensorboardX, torchnet, skimage, scipy; print('OK')"
```

5) Ejecutar un test rápido (1 epoch, batch pequeño):

```powershell
python train.py --datasetName CAVE --upscale_factor 3 --batchSize 4 --nEpochs 1 --cuda
```

Notas:
- `torchnet` puede requerir `pip install git+https://github.com/pytorch/tnt` si la instalación por pip falla.
- Si tus `.mat` usan nombres de variables distintos a `lr`/`hr`, ajusta `data_utils.py` o renombra las variables en los `.mat`.
