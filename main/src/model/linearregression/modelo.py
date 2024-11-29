import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.tools.sm_exceptions import ValueWarning

# Clase encargada de la logica del modelo de regresion lineal
class ModelLinearRegression:

    # Constructor
    def __init__(self):
        self.model  = None
        self.beta_0 = None
        self.beta_1 = None
        self.beta_2 = None
        self.beta_3 = None
        self.beta_4 = None
        self.epsilon = None


    # Funcion encargada de entrenar el modelo
    def train_model(self, string_functionality, string_quality_code, string_easy_usage, string_compatibility, string_downloads):

        # se convierten los daton recibidos como strings en arrays
        self.functionality  = self._map_string_to_array(string_functionality)
        self.quality_code   = self._map_string_to_array(string_quality_code)
        self.easy_usage     = self._map_string_to_array(string_easy_usage)
        self.compatibility = self._map_string_to_array(string_compatibility)
        self.downloads      = self._map_string_to_array(string_downloads)

        # Se degine dataframe con cada array de datos
        data = pd.DataFrame({
            'Funcionalidad': self.functionality,
            'Calidad del código': self.quality_code,
            'Facilidad de uso': self.easy_usage,
            'Compatibilidad': self.compatibility,
            'Descargas': self.downloads
        })

        # Dividir los datos en conjuntos de entrenamiento y prueba
        X = data[['Funcionalidad', 'Calidad del código', 'Facilidad de uso', 'Compatibilidad']]
        y = data['Descargas']

        X_train, X_test, y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=0)

        # Ajustar el modelo de regresión lineal
        self.model = LinearRegression()
        self.model.fit(X_train, y_train)


        # Obtener los coeficientes de regresión
        self.beta_0 = self.model.intercept_
        self.beta_1, self.beta_2, self.beta_3, self.beta_4 = self.model.coef_

        # Hacer predicciones con los datos de prueba
        self.y_pred = self.model.predict(X_test)


        # Calcular el término de error (epsilon)
        self.mse = mean_squared_error(self.y_test, self.y_pred)
        self.epsilon = np.sqrt(self.mse)


    # Metodo que mapea datos recibidos como strings, en array numericos
    def _map_string_to_array(self, string_numeros):
        numeros = list(map(int, string_numeros.split()))
        if len(numeros) < 5:
            raise ValueWarning("La lista debe contener al menos 5 elementos.")
        return numeros


    # Metodo encargado de realizar predicciones
    def predict_downloads(self, functionality, quality_code, easy_usage, compatibility):
        if self.model is None:
            raise ValueError("El modelo no está entrenado. Por favor, entrene el modelo antes de hacer predicciones.")

        self.y_pred = (self.beta_0 +
                self.beta_1 * functionality +
                self.beta_2 * quality_code +
                self.beta_3 * easy_usage +
                self.beta_4 * compatibility +
                self.epsilon)

        return self.y_pred


    # Metodo encargado de graficar valores reales vs predicciones
    def plot_predictions(self):
        if self.model is None:
            raise ValueError("El modelo no está entrenado. Por favor, entrene el modelo antes de generar la gráfica.")

        plt.figure(figsize=(10, 6))

        # Graficar los valores reales vs. predicciones
        plt.scatter(self.y_test, self.y_pred, color='blue', label='Predicciones')
        plt.plot([self.y_test.min(), self.y_test.max()], [self.y_test.min(), self.y_test.max()], 'k--', lw=2, label='Línea perfecta')

        plt.xlabel('Descargas Reales')
        plt.ylabel('Descargas Predichas')
        plt.title('Descargas Reales vs. Descargas Predichas')
        plt.legend()
        plt.grid(True)
        plt.show()

    # Metodo encargado de graficar media, mediana, desviacion estandar
    def plot_statistics_one(self):
        if self.model is None:
            raise ValueError("El modelo no está entrenado. Por favor, entrene el modelo antes de generar la gráfica.")

        data = pd.DataFrame({
            'Funcionalidad': self.functionality,
            'Calidad del código': self.quality_code,
            'Facilidad de uso': self.easy_usage,
            'Compatibilidad': self.compatibility,
            'Descargas': self.downloads
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

    # Metodo encargado de graficar varianza, moda y coeficiente de varianza
    def plot_statistics_two(self):
        if self.model is None:
            raise ValueError("El modelo no está entrenado. Por favor, entrene el modelo antes de generar la gráfica.")

        data = pd.DataFrame({
            'Funcionalidad': self.functionality,
            'Calidad del código': self.quality_code,
            'Facilidad de uso': self.easy_usage,
            'Compatibilidad': self.compatibility,
            'Descargas': self.downloads
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

        # mostrar gráfica
        plt.show()
