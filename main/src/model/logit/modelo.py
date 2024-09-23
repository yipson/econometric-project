import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


class ModelLogit:
    def __init__(self):
            self.logit_model  = None
            self.logit_fit = None
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
        x = data[['Portabilidad']]
        y = data['Exito']

        # Definimos y ajustamos el modelo Probit
        self.logit_model = LogisticRegression()
        try:
            self.logit_fit = self.logit_model.fit(x, y)
        except np.linalg.LinAlgError as e:
            raise Exception(f"Se requiere mas cantidad o mas variabilidad en los datos") from e

        # Extraemos B0 y B1
        self.beta_0 = self.logit_fit.intercept_[0]
        self.beta_1 = self.logit_fit.coef_[0][0]


        # Datos aleatorios para probar modelo
        x_test = np.random.randint(0, 2, self.sample_len)
        data_test = pd.DataFrame({'Portabilidad': x_test})
        data_test['prob_success'] = self.logit_fit.predict_proba(data_test[['Portabilidad']])[:, 1]
        data_test['prediction'] = np.where(data_test['prob_success'] >= 0.5, 1, 0)

        print(f"Probabilidades exito:\n{data_test['prob_success']}")
        print(f"Prediccion:\n{data_test['prediction']}")

        # Evaluar el modelo
        exactitud = np.mean(data_test['prediction'] == data['Exito'])
        print(f'Exactitud:\n{exactitud}')



    def predict_success(self, portability):
        if self.logit_fit is None:
            raise ValueError("El modelo no está entrenado. Por favor, entrene el modelo antes de hacer predicciones.")

        x = np.full(1, portability).astype(int)
        data = pd.DataFrame({'Portabilidad': x})
        data['prob_success'] = self.logit_fit.predict_proba(data[['Portabilidad']])[:, 1]
        data['prediction'] = np.where(data['prob_success'] >= 0.5, 1, 0)

        if data['prediction'][0] == 1:
            return "Exitoso"
        return "Fracaso"



    def _map_string_to_array(self, string_numeros):
        return list(map(int, string_numeros.split()))
