import numpy as np
import pandas as pd

# Generar datos aleatorios
np.random.seed(0)  # Para reproducibilidad
n_samples = 50

funcionalidad = np.random.randint(0, 11, n_samples)
calidad_codigo = np.random.randint(0, 11, n_samples)
facilidad_uso = np.random.randint(0, 11, n_samples)
compatibilidad = np.random.randint(0, 2, n_samples)

# Crear un DataFrame con los datos
data = pd.DataFrame({
    'Funcionalidad': funcionalidad,
    'Calidad del código': calidad_codigo,
    'Facilidad de uso': facilidad_uso,
    'Compatibilidad': compatibilidad
})

# Coeficientes del modelo
beta_0 = 10
beta_1 = 2.5
beta_2 = 1.8
beta_3 = 1.2
beta_4 = 3.7

# Función para predecir el número de descargas
def predecir_descargas(funcionalidad, calidad_codigo, facilidad_uso, compatibilidad, epsilon=0):
    return beta_0 + beta_1 * \
           funcionalidad + beta_2 * \
           calidad_codigo + beta_3 * \
           facilidad_uso + beta_4 * \
           compatibilidad + epsilon

# Generar datos de prueba
datos_prueba = pd.DataFrame({
    'Funcionalidad':        [8, 6, 9, 5, 7],
    'Calidad del código':   [7, 4, 6, 5, 8],
    'Facilidad de uso':     [9, 7, 8, 6, 7],
    'Compatibilidad':       [1, 0, 1, 0, 1]
})

# Predecir descargas para los datos de prueba
predicciones = datos_prueba.apply(
    lambda x: predecir_descargas(
        x['Funcionalidad'], x['Calidad del código'],
        x['Facilidad de uso'], x['Compatibilidad'], epsilon=2.5),
    axis=1)

# Mostrar las predicciones
#print(predicciones)

# Mostrar dataframe
#print(data)
