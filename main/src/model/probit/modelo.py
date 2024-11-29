import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import accuracy_score, recall_score
from statsmodels.tools.sm_exceptions import ValueWarning

# Clase encargada de la logica del modelo logit
class ModelProbit:

    # Constructor
    def __init__(self):
        self.probit_model  = None
        self.probit_fit = None
        self.beta_0 = None
        self.beta_1 = None
        self.sample_len = None

    # Metodo encargado de entrenar el modelo
    def train_model(self, string_portability, string_success):

        # se mapean los datos de entrada de string a arrays numericos
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


        # se obtienen coeficientes
        try:
            self.beta_0 = self.probit_fit.params[0]
            self.beta_1 = self.probit_fit.params[1]
        except IndexError as e:
            raise Exception(f"Se mas variabilidad en los datos") from e

        # Datos aleatorios para probar modelo
        data_test = np.random.randint(0, 2, size=(self.sample_len))
        X_new = sm.add_constant(pd.DataFrame({'X1': data_test}), has_constant='add')

        # Predecimos la probabilidad para los nuevos datos
        y_pred_prob  = self.probit_fit.predict(X_new)
        y_pred = (y_pred_prob >= 0.5).astype(int)
        print(f'Probabilidades predichas:\n{y_pred_prob}')
        print(f'Exito 1 - Fracaso 0:\n{y_pred}') # se conoce el exito o fracaso

        ## Imprimimos en consola tasa de acierto y sensibilidad
        exactitud = accuracy_score(y, y_pred)
        sensibilidad = recall_score(y, y_pred)
        print(f"Tasa de acierto : {exactitud}")
        print(f"Sensibilidad    : {sensibilidad}")

    # metodo que predice el exito
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



    # Metodo que mapea datos recibidos como strings, en array numericos
    def _map_string_to_array(self, string_numeros):
        numeros = list(map(int, string_numeros.split()))
        if len(numeros) < 5:
            raise ValueWarning("La lista debe contener al menos 5 elementos.")
        return numeros


    # Metodo que grafica media, mediana y desviacion estandar
    def plot_statistics_one(self):
        if self.probit_model is None:
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


    # Metodo que grafica varianza moda y coeficiente de variacion
    def plot_statistics_two(self):
        if self.probit_model is None:
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
