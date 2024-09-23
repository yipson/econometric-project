import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


class ModelLinearRegression:
    def __init__(self):
        self.model  = None
        self.beta_0 = None
        self.beta_1 = None
        self.beta_2 = None
        self.beta_3 = None
        self.beta_4 = None
        self.epsilon = None

    def train_model(self, string_functionality, string_quality_code, string_easy_usage, string_compatibility, string_downloads):

        self.functionality  = self._map_string_to_array(string_functionality)
        self.quality_code   = self._map_string_to_array(string_quality_code)
        self.easy_usage     = self._map_string_to_array(string_easy_usage)
        self.compatibility = self._map_string_to_array(string_compatibility)
        self.downloads      = self._map_string_to_array(string_downloads)

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



    def _map_string_to_array(self, string_numeros):
        return list(map(int, string_numeros.split()))


    def predict_downloads(self, functionality, quality_code, easy_usage, compatibility):
        if self.model is None:
            raise ValueError("El modelo no está entrenado. Por favor, entrene el modelo antes de hacer predicciones.")

        return (self.beta_0 +
                self.beta_1 * functionality +
                self.beta_2 * quality_code +
                self.beta_3 * easy_usage +
                self.beta_4 * compatibility +
                self.epsilon)

    # por revisar
    def predict_downloads_random_data(self, functionality, quality_code, easy_usage, compatibility, epsilon):
        if self.model is None:
            raise ValueError("El modelo no está entrenado. Por favor, entrene el modelo antes de hacer predicciones.")

        return (self.beta_0 +
                self.beta_1 * functionality +
                self.beta_2 * quality_code +
                self.beta_3 * easy_usage +
                self.beta_4 * compatibility +
                self.epsilon)


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


    def plot_statistics(self):
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
        mode_values = data.mode().iloc[0]
        range_values = data.max() - data.min()
        variance_values = data.var()
        coef_variation_values = std_dev_values / mean_values

        # Crear subplots
        fig, axs = plt.subplots(4, 2, figsize=(15, 20))

        axs[0, 0].bar(data.columns, mean_values)
        axs[0, 0].set_title('Media')

        axs[0, 1].bar(data.columns, median_values)
        axs[0, 1].set_title('Mediana')

        axs[1, 0].bar(data.columns, std_dev_values)
        axs[1, 0].set_title('Desviación Estándar')

        axs[1, 1].bar(data.columns, mode_values)
        axs[1, 1].set_title('Moda')

        axs[2, 0].bar(data.columns, range_values)
        axs[2, 0].set_title('Rango')

        axs[2, 1].bar(data.columns, variance_values)
        axs[2, 1].set_title('Varianza')

        axs[3, 0].bar(data.columns, coef_variation_values)
        axs[3, 0].set_title('Coeficiente de Variación')

        # Ajustar diseño y mostrar gráfica
        plt.tight_layout()
        plt.show()
