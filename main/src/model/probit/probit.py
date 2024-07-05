import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from statsmodels.discrete.discrete_model import Probit
from statsmodels.api import add_constant
from scipy.stats import norm
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Generar datos aleatorios
np.random.seed(0)  # Para reproducibilidad
n_samples = 50

# Generar datos aleatorios para la portabilidad y el éxito
portabilidad = np.random.randint(0, 2, n_samples)  # Valores binarios 0 o 1
true_beta_0 = 2.5
true_beta_1 = 1.2
epsilon = np.random.randn(n_samples)  # Ruido aleatorio

# Definir éxito como una variable binaria dependiente de la portabilidad y el ruido
z = true_beta_0 + true_beta_1 * portabilidad + epsilon
exito = (z > 0).astype(int)  # Convertir a variable binaria

# Crear un DataFrame con los datos
data = pd.DataFrame({
    'Portabilidad': portabilidad,
    'Éxito': exito
})

# Dividir los datos en conjuntos de entrenamiento y prueba
X = data[['Portabilidad']]
y = data['Éxito']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Agregar una columna de unos para el intercepto
X_train = add_constant(X_train)
X_test = add_constant(X_test)

# Ajustar el modelo Probit
probit_model = Probit(y_train, X_train).fit()

# Obtener los coeficientes de regresión
beta_0 = probit_model.params[0]
beta_1 = probit_model.params[1]

print(f"Coeficientes del modelo:")
print(f"β0: {beta_0}")
print(f"β1: {beta_1}")

# Hacer predicciones con los datos de prueba
z_test = beta_0 + beta_1 * X_test['Portabilidad']
probabilidades = norm.cdf(z_test)

# Establecer un punto de corte (umbral) para clasificar el éxito
punto_de_corte = 0.5
predicciones = (probabilidades > punto_de_corte).astype(int)

# Evaluar el desempeño del modelo
# Calcular la tasa de acierto
accuracy = accuracy_score(y_test, predicciones)
print(f"Tasa de acierto: {accuracy}")

# Matriz de confusión
conf_matrix = confusion_matrix(y_test, predicciones)
print(f"Matriz de confusión:\n{conf_matrix}")

# Informe de clasificación
class_report = classification_report(y_test, predicciones)
print(f"Informe de clasificación:\n{class_report}")

# Generar nuevos datos de prueba
datos_prueba = pd.DataFrame({
    'Portabilidad': [1, 0, 1, 0, 1]
})

# Agregar una columna de unos para el intercepto
datos_prueba = add_constant(datos_prueba)

# Predecir el éxito para los nuevos datos de prueba
z_nuevos = beta_0 + beta_1 * datos_prueba['Portabilidad']
probabilidades_nuevas = norm.cdf(z_nuevos)
predicciones_nuevas = (probabilidades_nuevas > punto_de_corte).astype(int)

# Mostrar las predicciones para los nuevos datos
print(f"Predicciones para los nuevos datos:\n{predicciones_nuevas}")
