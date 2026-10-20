import glob
import pandas as pd

# Busca todos los archivos CSV en la ruta deseada
files = glob.glob("/home/david/Música/AudioMNIST/8000_20000_50/01/0_01_*.csv", recursive=True)

# Lee y concatena todos los archivos en un único DataFrame
df_combined = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

# Guarda el resultado en un nuevo archivo CSV unificado
df_combined.to_csv("dataset_completo.csv", index=False)

print("Completado")