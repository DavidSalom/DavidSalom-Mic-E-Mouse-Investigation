# Mic-E-Mouse: your computer mouse has big ears
## Description

Side-Channel Audio Attack through computer mice.

## Requirements
- [Conda](https://docs.conda.io/en/latest/miniconda.html) or [Mamba](https://mamba.readthedocs.io/en/latest/index.html)
- A computer mouse with a high polling rate, and a high CPI count.
## Installation
The conda/mamba environment can be installed with the following command: `conda env create -f environment.yml`

## Fixes

### 1. Robustez Numérica y Prevención de `NaN` / `Inf`

#### 🎯 Motivo del cambio
Durante el procesamiento de señales no uniformes o de baja energía (tramos en silencio), las operaciones matemáticas de interpolación y normalización generaban valores `NaN` o `Inf`:
- **División por cero en el cálculo de derivadas temporales**: Cuando dos muestras consecutivas compartían la misma marca de tiempo o tenían un intervalo despreciable ($dt = 0$), el factor de ponderación $W$ divergía.
- **Señales con desviación estándar nula o casi nula**: Al normalizar tramos silenciosos, `torch.std()` tiende a cero, colapsando el entrenamiento de la red.
- **Corrupción en la conversión a escala Mel dB**: Valores inválidos residuales provocaban divergencia en el logaritmo (`log10`).

#### 📝 Modificaciones implementadas

##### A) Protección en interpolación temporal (`core2/signal/preprocess.py`)
Se añadió un factor de regularización $\epsilon = 10^{-8}$ en el diferencial de tiempo y se sanitizó la salida mediante `torch.nan_to_num`:

```diff
-   W = ((Tperiodic - Tcumul[idxx]) / (Tcumul[idxx + 1] - Tcumul[idxx]))
+   # Evitar divisiones por cero sumando un epsilon
+   dt = Tcumul[idxx + 1] - Tcumul[idxx]
+   dt = torch.where(dt == 0, torch.tensor(1e-8, device=device), dt)
+   W = ((Tperiodic - Tcumul[idxx]) / dt)
...
+   # Limpieza preventiva de valores anómalos residuales
+   X = torch.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
+   Y = torch.nan_to_num(Y, nan=0.0, posinf=0.0, neginf=0.0)
```

##### B) Normalización y conversión Mel dB seguras (`core2/nn/utils.py`)
Se estabilizó el denominador durante el cálculo de potencia y se aseguró la entrada previa al logaritmo:

```diff
-   W /= torch.std(W)
-   M /= torch.std(M)
+   W /= (torch.std(W) + 1e-8)
+   M /= (torch.std(M) + 1e-8)
...
    # Conversión Amplitud a dB
+   torch.nan_to_num(W)
    W = (torch.maximum(torch.clamp(W, min=1e-10).log10(), torch.clamp(W, min=1e-10).log10().max() - 8.0) + 4.0) / 4.0
```

---

### 2. Limpieza y Reorganización de Archivos

#### 🎯 Motivo del cambio
- **Desacoplamiento de la configuración**: Se retiró `config.py` para favorecer una inyección de hiperparámetros dinámica mediante argumentos CLI / diccionarios en lugar de valores fijados en disco.
- **Modularización de la arquitectura**: Se movió la definición de `ReconstructionModel` de `core/nn/reconstruction.py` directamente hacia los pipelines y notebooks experimentales correspondientes, evitando dependencias rígidas en el paquete central.

#### 🗂️ Comparativa de estructura

| Archivo | Estado en `core` (default) | Estado en `core2` (v2) | Justificación |
| :--- | :---: | :---: | :--- |
| `core/config.py` | ✅ Presente | ❌ Retirado | Parametrización externa / flexible. |
| `core/nn/reconstruction.py` | ✅ Presente | ❌ Retirado | Modularización trasladada a los scripts de entrenamiento. |
| `waveformReconstruction_AUDIOMNIST.ipynb` | ✅ ~49 MB | — | Notebook principal con outputs y trazas completas. |
| `waveformReconstruction_AUDIOMNIST2.ipynb` | — | 📄 Plantilla | Cuaderno limpio para nuevas pruebas. |

> [!NOTE]
> Con estas modificaciones se asegura que el entrenamiento de las redes neuronales no sufra cortes abruptos por gradientes corruptos (`Loss: NaN`) provocados por artefactos en la captura de señales acústicas.

## New features

## Investigation
https://mic-e-mouse-investigation.com

## Data
Basedir = https://github.com/soerenab/AudioMNIST.git
https://drive.google.com/drive/folders/1DcTldouupfp7BMteE1Br0lq7RCdQQ0Hc?usp=drive_link

## Authors
- [David Salom](https://github.com/DavidSalom) - Student of vocational training in Sant Josep Obrer Palma de Mallorca.
## Citation
- [Mohamad Fakih]() - PhD Student, EECS, University of California Irvine
- [Rahul Dharmaji](https://iika.re) - MS Student, EECS, University of California Irvine
- [Mohammad Abdullah Al Faruque](https://aicps.eng.uci.edu/) - Professor, EECS, University of California Irvine

## Contact
salomflorit@gmail.com
