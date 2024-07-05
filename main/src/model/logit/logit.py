import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

# Generar datos aleatorios
#np.random.seed(123)
portabilidad = np.random.choice([0, 1], size=100)
exito = np.random.binomial(n=1, p=0.5, size=100)
datos = pd.DataFrame({'Portabilidad': portabilidad, 'Exito': exito})

# Ajustar el modelo Logit
modelo = LogisticRegression()
modelo.fit(datos[['Portabilidad']], datos['Exito'])

# Predecir el éxito del software
datos['Prob_Exito'] = modelo.predict_proba(datos[['Portabilidad']])[:, 1]
datos['Prediccion'] = np.where(datos['Prob_Exito'] >= 0.5, 1, 0)

# Evaluar el modelo
exactitud = np.mean(datos['Prediccion'] == datos['Exito'])
print('Exactitud:', exactitud)


