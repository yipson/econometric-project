import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Generar datos aleatorios
np.random.seed(0)  # Para reproducibilidad
n_samples = 50

funcionalidad = np.random.randint(0, 11, n_samples)
calidad_codigo = np.random.randint(0, 11, n_samples)
facilidad_uso = np.random.randint(0, 11, n_samples)
compatibilidad = np.random.randint(0, 2, n_samples)

# Generar descargas (valores de destino) usando un modelo ficticio con ruido
true_beta_0 = 10
true_beta_1 = 2.5
true_beta_2 = 1.8
true_beta_3 = 1.2
true_beta_4 = 3.7
epsilon = np.random.randn(n_samples)  # Ruido aleatorio

descargas = (true_beta_0 + true_beta_1 * funcionalidad + true_beta_2 * calidad_codigo +
             true_beta_3 * facilidad_uso + true_beta_4 * compatibilidad + epsilon)

# Crear un DataFrame con los datos
data = pd.DataFrame({
    'Funcionalidad': funcionalidad,
    'Calidad del código': calidad_codigo,
    'Facilidad de uso': facilidad_uso,
    'Compatibilidad': compatibilidad,
    'Descargas': descargas
})

# Dividir los datos en conjuntos de entrenamiento y prueba
X = data[['Funcionalidad', 'Calidad del código', 'Facilidad de uso', 'Compatibilidad']]
y = data['Descargas']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Ajustar el modelo de regresión lineal
model = LinearRegression()
model.fit(X_train, y_train)

# Obtener los coeficientes de regresión
beta_0 = model.intercept_
beta_1, beta_2, beta_3, beta_4 = model.coef_

print(f"Coeficientes del modelo:")
print(f"β0: {beta_0}")
print(f"β1: {beta_1}")
print(f"β2: {beta_2}")
print(f"β3: {beta_3}")
print(f"β4: {beta_4}")

# Hacer predicciones con los datos de prueba
y_pred = model.predict(X_test)

# Calcular el término de error (epsilon)
mse = mean_squared_error(y_test, y_pred)
epsilon = np.sqrt(mse)
print(f"Término de error (epsilon): {epsilon}")

# Función para predecir el número de descargas
def predecir_descargas(funcionalidad, calidad_codigo, facilidad_uso, compatibilidad, epsilon=0):
    return beta_0 + beta_1 * funcionalidad + beta_2 * calidad_codigo + beta_3 * facilidad_uso + beta_4 * compatibilidad + epsilon

# Generar nuevos datos de prueba
datos_prueba = pd.DataFrame({
    'Funcionalidad': [8, 6, 9, 5, 7],
    'Calidad del código': [7, 4, 6, 5, 8],
    'Facilidad de uso': [9, 7, 8, 6, 7],
    'Compatibilidad': [1, 0, 1, 0, 1]
})

# Predecir descargas para los datos de prueba
predicciones = datos_prueba.apply(lambda x: predecir_descargas(x['Funcionalidad'], x['Calidad del código'], x['Facilidad de uso'], x['Compatibilidad'], epsilon=epsilon), axis=1)

# Mostrar las predicciones
print(predicciones)
