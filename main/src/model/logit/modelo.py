import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from statsmodels.tools.sm_exceptions import ValueWarning

# Clase encargada de la logica del modelo logit
class ModelLogit:

    # Constructor
    def __init__(self):
        self.logit_model = None
        self.logit_fit = None
        self.beta_0 = None
        self.beta_1 = None
        self.sample_len = None


    # funcion encargada de entrenar el modelo
    def train_model(self, string_portability, string_success):

        # se mapean los datos de entrada de string a arrays numericos
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


    # Metodo encargado de predecir el exito
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


    # Metodo encargado de mapear las entradas de string a array numericos
    def _map_string_to_array(self, string_numeros):
        numeros = list(map(int, string_numeros.split()))
        if len(numeros) < 5:
            raise ValueWarning("La lista debe contener al menos 5 elementos.")
        return numeros


    # Grafica media, mediana, desviacion estandar
    def plot_statistics_one(self):
        if self.logit_model is None:
            raise ValueError("El modelo no está entrenado. Por favor, entrene el modelo antes de generar la gráfica.")

        data = pd.DataFrame({
            'Portabilidad': self.portability,
            'Exito': self.success
        })

        # Calcular estadísticas
        mean_values = data.mean()
        median_values = data.median()
        std_dev_values = data.std()

        # Crear subplots
        fig, axs = plt.subplots(2, 2, figsize=(15, 20))

        axs[0, 0].bar(data.columns, mean_values)
        axs[0, 0].set_title('Media', fontsize=14)
        axs[0, 0].set_xticklabels(data.columns, rotation=45, ha='right')

        axs[0, 1].bar(data.columns, median_values)
        axs[0, 1].set_title('Mediana', fontsize=14)
        axs[0, 1].set_xticklabels(data.columns, rotation=45, ha='right')

        axs[1, 0].bar(data.columns, std_dev_values)
        axs[1, 0].set_title('Desviación Estándar', fontsize=14)
        axs[1, 0].set_xticklabels(data.columns, rotation=45, ha='right')

        # Ajustar el espacio entre las gráficas
        plt.subplots_adjust(left=0.1, bottom=0.2, right=0.9, top=0.9, wspace=0.3, hspace=0.5)

        # Mostrar la gráfica
        plt.show()


    # Grafica varianza moda y coeficiente de variacion
    def plot_statistics_two(self):
        if self.logit_model is None:
            raise ValueError("El modelo no está entrenado. Por favor, entrene el modelo antes de generar la gráfica.")

        data = pd.DataFrame({
            'Portabilidad': self.portability,
            'Exito': self.success
        })

        # Calcular estadísticas
        mean_values = data.mean()
        std_dev_values = data.std()
        mode_values = data.mode().iloc[0]
        variance_values = data.var()
        coef_variation_values = std_dev_values / mean_values

        # Crear subplots
        fig, axs = plt.subplots(2, 2, figsize=(15, 20))

        axs[0, 0].bar(data.columns, variance_values)
        axs[0, 0].set_title('Varianza')
        axs[0, 0].set_xticklabels(data.columns, rotation=45, ha='right')

        axs[0, 1].bar(data.columns, mode_values)
        axs[0, 1].set_title('Moda')
        axs[0, 1].set_xticklabels(data.columns, rotation=45, ha='right')

        axs[1, 0].bar(data.columns, coef_variation_values)
        axs[1, 0].set_title('Coeficiente de Variación')
        axs[1, 0].set_xticklabels(data.columns, rotation=45, ha='right')

        # Ajustar el espacio entre las gráficas
        plt.subplots_adjust(left=0.1, bottom=0.2, right=0.9, top=0.9, wspace=0.3, hspace=0.5)

        # mostrar gráfica
        plt.show()
