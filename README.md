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
