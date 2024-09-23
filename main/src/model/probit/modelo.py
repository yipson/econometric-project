import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import Probit
from scipy.stats import norm  # Cambiamos esta línea para usar scipy.stats
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score
from tkinter import messagebox



class ModelProbit:
    def __init__(self):
        self.probit_model  = None
        self.probit_fit = None
        self.beta_0 = None
        self.beta_1 = None
        self.sample_len = None


    def train_model(self, string_portability, string_success):

        self.portability = self._map_string_to_array(string_portability)
        self.success = self._map_string_to_array(string_success)

        self.sample_len = len(self.portability)

        data = pd.DataFrame({
            'Portabilidad': self.portability,
            'Exito': self.success
        })
        x = sm.add_constant(data['Portabilidad'])
        y = data['Exito']

        # Definimos y ajustamos el modelo Probit
        self.probit_model = sm.Probit(y, x)
        try:
            self.probit_fit = self.probit_model.fit()
        except np.linalg.LinAlgError as e:
            raise Exception(f"Se requiere mas cantidad o mas variabilidad en los datos") from e

        # Extraemos B0 y B1
        self.beta_0 = self.probit_fit.params[0]
        self.beta_1 = self.probit_fit.params[1]

        # Datos aleatorios para probar modelo
        data_test = np.random.randint(0, 2, size=(self.sample_len))
        X_new = sm.add_constant(pd.DataFrame({'X1': data_test}), has_constant='add')

        # Predecimos la probabilidad para los nuevos datos
        y_pred_prob  = self.probit_fit.predict(X_new)
        y_pred = (y_pred_prob >= 0.5).astype(int)
        print(f'Probabilidades predichas:\n{y_pred_prob}')
        print(f'Exito 1 - Fracaso 0:\n{y_pred}') # se conoce el exito o fracaso

        ## Metricas
        exactitud = accuracy_score(y, y_pred)
        sensibilidad = recall_score(y, y_pred)
        print(f"Tasa de acierto : {exactitud}")
        print(f"Sensibilidad    : {sensibilidad}")


    def predict_success(self, portability):
        if self.probit_fit is None:
            raise ValueError("El modelo no está entrenado. Por favor, entrene el modelo antes de hacer predicciones.")

        data = np.full(1, portability).astype(int)
        X_new = sm.add_constant(pd.DataFrame({'X1': data}), has_constant='add')
        y_pred_prob  = self.probit_fit.predict(X_new)
        y_pred = (y_pred_prob >= 0.5).astype(int)

        if y_pred[0] == 1:
            return "Exitoso"
        return "Fracaso"



    #definir como metodo en package utils para que pueda ser usado de forma global
    def _map_string_to_array(self, string_numeros):
        return list(map(int, string_numeros.split()))
