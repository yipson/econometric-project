# interfaz grafica 
import tkinter as tk
from tkinter import messagebox
import random
import math
from scipy.stats import norm

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Desarrollo Econométrico")

        # Coeficientes del modelo de regresión lineal
        self.beta0_regresion = 10  # Constante
        self.beta1_regresion = 2.5  # Coeficiente para Funcionalidad
        self.beta2_regresion = 1.8  # Coeficiente para Calidad del código
        self.beta3_regresion = 1.2  # Coeficiente para Facilidad de uso
        self.beta4_regresion = 3.7  # Coeficiente para Compatibilidad

        # Coeficientes del modelo Probit
        self.beta0_probit = 2.5  # Constante
        self.beta1_probit = 1.2  # Coeficiente para Portabilidad

        # Coeficientes del modelo Logit
        self.beta0_logit = -1.3  # Constante
        self.beta1_logit = 0.9   # Coeficiente para Portabilidad

        # Crear la interfaz del menú principal
        self.create_main_menu()

    def create_main_menu(self):
        # Limpiar la ventana actual
        for widget in self.root.winfo_children():
            widget.destroy()

        # Título centrado
        self.lbl_titulo = tk.Label(self.root, text="Desarrollo Econométrico", font=("Arial", 24))
        self.lbl_titulo.pack(pady=20)

        # Botones del menú
        self.btn_regresion = tk.Button(self.root, text="Predicción de Descargas (Regresión Lineal)", command=self.create_regression_interface)
        self.btn_regresion.pack(pady=10)

        self.btn_probit = tk.Button(self.root, text="Predicción de Éxito (Modelo Probit)", command=self.create_probit_interface)
        self.btn_probit.pack(pady=10)

        self.btn_logit = tk.Button(self.root, text="Predicción de Éxito (Modelo Logit)", command=self.create_logit_interface)
        self.btn_logit.pack(pady=10)

        self.btn_salir = tk.Button(self.root, text="Salir", command=self.root.quit)
        self.btn_salir.pack(pady=10)

    def create_regression_interface(self):
        self.clear_window()

        # Funcionalidad
        self.lbl_funcionalidad = tk.Label(self.root, text="Funcionalidad:")
        self.lbl_funcionalidad.pack(pady=5)
        self.entry_funcionalidad = tk.Entry(self.root)
        self.entry_funcionalidad.pack(pady=5)

        # Calidad del código
        self.lbl_calidad = tk.Label(self.root, text="Calidad del código:")
        self.lbl_calidad.pack(pady=5)
        self.entry_calidad = tk.Entry(self.root)
        self.entry_calidad.pack(pady=5)

        # Facilidad de uso
        self.lbl_facilidad = tk.Label(self.root, text="Facilidad de uso:")
        self.lbl_facilidad.pack(pady=5)
        self.entry_facilidad = tk.Entry(self.root)
        self.entry_facilidad.pack(pady=5)

        # Compatibilidad
        self.lbl_compatibilidad = tk.Label(self.root, text="Compatibilidad (0 o 1):")
        self.lbl_compatibilidad.pack(pady=5)
        self.entry_compatibilidad = tk.Entry(self.root)
        self.entry_compatibilidad.pack(pady=5)

        # Botón para predecir descargas
        self.btn_predecir = tk.Button(self.root, text="Predecir Descargas", command=self.predecir_descargas)
        self.btn_predecir.pack(pady=10)

        # Botón para generar datos aleatorios y predecir descargas
        self.btn_aleatorio = tk.Button(self.root, text="Generar Datos Aleatorios y Predecir", command=self.generar_y_predecir)
        self.btn_aleatorio.pack(pady=10)

        # Botón para limpiar entradas y resultados
        self.btn_limpiar = tk.Button(self.root, text="Limpiar", command=self.limpiar_campos)
        self.btn_limpiar.pack(pady=10)

        # Resultados
        self.lbl_resultados = tk.Label(self.root, text="Resultados:")
        self.lbl_resultados.pack(pady=5)
        self.txt_resultados = tk.Text(self.root, height=20, width=60)
        self.txt_resultados.pack(pady=5)

    def create_probit_interface(self):
        self.clear_window()

        # Portabilidad
        self.lbl_portabilidad = tk.Label(self.root, text="Portabilidad (0 o 1):")
        self.lbl_portabilidad.pack(pady=5)
        self.entry_portabilidad = tk.Entry(self.root)
        self.entry_portabilidad.pack(pady=5)

        # Botón para predecir éxito
        self.btn_predecir_exito = tk.Button(self.root, text="Predecir Éxito", command=self.predecir_exito_probit)
        self.btn_predecir_exito.pack(pady=10)

        # Resultados
        self.lbl_resultados = tk.Label(self.root, text="Resultados:")
        self.lbl_resultados.pack(pady=5)
        self.txt_resultados = tk.Text(self.root, height=20, width=60)
        self.txt_resultados.pack(pady=5)

    def create_logit_interface(self):
        self.clear_window()

        # Portabilidad
        self.lbl_portabilidad = tk.Label(self.root, text="Portabilidad (0 o 1):")
        self.lbl_portabilidad.pack(pady=5)
        self.entry_portabilidad = tk.Entry(self.root)
        self.entry_portabilidad.pack(pady=5)

        # Botón para predecir éxito
        self.btn_predecir_exito = tk.Button(self.root, text="Predecir Éxito", command=self.predecir_exito_logit)
        self.btn_predecir_exito.pack(pady=10)

        # Resultados
        self.lbl_resultados = tk.Label(self.root, text="Resultados:")
        self.lbl_resultados.pack(pady=5)
        self.txt_resultados = tk.Text(self.root, height=20, width=60)
        self.txt_resultados.pack(pady=5)

    def clear_window(self):
        for widget in self.root.winfo_children():
            widget.destroy()

    def predecir_descargas(self):
        try:
            funcionalidad = float(self.entry_funcionalidad.get())
            calidad = float(self.entry_calidad.get())
            facilidad = float(self.entry_facilidad.get())
            compatibilidad = float(self.entry_compatibilidad.get())
        except ValueError:
            messagebox.showwarning("Advertencia", "Por favor, ingrese valores numéricos en todos los campos.")
            return

        # Término de error aleatorio
        epsilon = random.uniform(-5, 5)

        # Predicción del número de descargas
        prediccion = (self.beta0_regresion +
                      self.beta1_regresion * funcionalidad +
                      self.beta2_regresion * calidad +
                      self.beta3_regresion * facilidad +
                      self.beta4_regresion * compatibilidad +
                      epsilon)

        resultado = f"Predicción del número de descargas: {prediccion:.2f}\n"
        self.txt_resultados.insert(tk.END, resultado)

    def generar_y_predecir(self):
        self.txt_resultados.delete('1.0', tk.END)
        for i in range(50):
            funcionalidad = random.uniform(0, 10)
            calidad = random.uniform(0, 10)
            facilidad = random.uniform(0, 10)
            compatibilidad = random.choice([0, 1])
            epsilon = random.uniform(-5, 5)

            # Predicción del número de descargas
            prediccion = (self.beta0_regresion +
                          self.beta1_regresion * funcionalidad +
                          self.beta2_regresion * calidad +
                          self.beta3_regresion * facilidad +
                          self.beta4_regresion * compatibilidad +
                          epsilon)

            resultado = (f"Software {i+1}:\n"
                         f"Funcionalidad: {funcionalidad:.2f}\n"
                         f"Calidad del código: {calidad:.2f}\n"
                         f"Facilidad de uso: {facilidad:.2f}\n"
                         f"Compatibilidad: {compatibilidad}\n"
                         f"Predicción del número de descargas: {prediccion:.2f}\n\n")
            self.txt_resultados.insert(tk.END, resultado)

    def predecir_exito_probit(self):
        try:
            portabilidad = float(self.entry_portabilidad.get())
        except ValueError:
            messagebox.showwarning("Advertencia", "Por favor, ingrese un valor numérico en el campo de portabilidad.")
            return

        # Cálculo de la probabilidad usando el modelo Probit
        z = self.beta0_probit + self.beta1_probit * portabilidad
        probabilidad_exito = norm.cdf(z)

        # Determinar éxito o fracaso basado en un umbral de 0.5
        exito = "Éxito" if probabilidad_exito >= 0.5 else "Fracaso"

        resultado = f"Probabilidad de éxito: {probabilidad_exito:.2f} ({exito})\n"
        self.txt_resultados.insert(tk.END, resultado)

    def predecir_exito_logit(self):
        try:
            portabilidad = float(self.entry_portabilidad.get())
        except ValueError:
            messagebox.showwarning("Advertencia", "Por favor, ingrese un valor numérico en el campo de portabilidad.")
            return

        # Cálculo de la probabilidad usando el modelo Logit
        logit_value = self.beta0_logit + self.beta1_logit * portabilidad
        probabilidad_exito = math.exp(logit_value) / (1 + math.exp(logit_value))

        # Determinar éxito o fracaso basado en un umbral de 0.5
        exito = "Éxito" if probabilidad_exito >= 0.5 else "Fracaso"

        resultado = f"Probabilidad de éxito: {probabilidad_exito:.2f} ({exito})\n"
        self.txt_resultados.insert(tk.END, resultado)

    def limpiar_campos(self):
        if hasattr(self, 'entry_funcionalidad'):
            self.entry_funcionalidad.delete(0, tk.END)
        if hasattr(self, 'entry_calidad'):
            self.entry_calidad.delete(0, tk.END)
        if hasattr(self, 'entry_facilidad'):
            self.entry_facilidad.delete(0, tk.END)
        if hasattr(self, 'entry_compatibilidad'):
            self.entry_compatibilidad.delete(0, tk.END)
        if hasattr(self, 'entry_portabilidad'):
            self.entry_portabilidad.delete(0, tk.END)
        self.txt_resultados.delete('1.0', tk.END)

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
