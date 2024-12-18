# interfaz grafica
import tkinter as tk
from tkinter import messagebox
import random
from model.linearregression.modelo import ModelLinearRegression
from model.probit.modelo import ModelProbit
from model.logit.modelo import ModelLogit

# Clase principal que recibe datos de entrada
# y define la interfaz grafica
class App:

    # Metodo que inicia la aplicacion
    # inicializa variables globales de interfaz grafica
    def __init__(self, root):
        self.root = root
        self.modelo_regresion_lineal = ModelLinearRegression()
        self.modelo_probit = ModelProbit()
        self.modelo_logit = ModelLogit()

        self.root.title("Success Analyzer")

        self.root.geometry("500x700")
        self.root.resizable(False, False)

        self.bg_color = "#01283b"  # Fondo general
        self.btn_bg_color = "#00acca"  # Fondo de botones
        self.btn_bg_color_amarillo = "#cad300"
        self.label_bg_color_verde = "#5bbb01"
        self.text_blanco = "#ffffff"
        self.gris = "#504f5e"
        self.amarillo = "#f0c818"

        self.root.configure(bg=self.bg_color)

        self.create_main_menu_view()


    # Pantalla de menu principal
    def create_main_menu_view(self):
        # Limpiar la ventana actual
        for widget in self.root.winfo_children():
            widget.destroy()

        # Título centrado
        self.lbl_titulo = tk.Label(self.root, text="Success Analyzer", font=("Arial", 30),
                                   bg=self.label_bg_color_verde, fg=self.text_blanco)
        self.lbl_titulo.pack(pady=100)

        # Botones del menú
        self.btn_regresion = tk.Button(self.root, text="Regresión Lineal", font=("Arial", 18),
                                       bg=self.btn_bg_color, fg=self.text_blanco, command=self.linear_regression_view)
        self.btn_regresion.pack(pady=20)

        self.btn_probit = tk.Button(self.root, text="Modelo Probit", font=("Arial", 18),
                                    bg=self.btn_bg_color, fg=self.text_blanco, command=self.probit_view)
        self.btn_probit.pack(pady=20)

        self.btn_logit = tk.Button(self.root, text="Modelo Logit", font=("Arial", 18),
                                   bg=self.btn_bg_color, fg=self.text_blanco, command=self.logit_view)
        self.btn_logit.pack(pady=20)

        self.btn_salir = tk.Button(self.root, text="Salir", font=("Arial", 10), command=self.root.quit,
                                   bg=self.btn_bg_color_amarillo)
        self.btn_salir.pack(pady=10)



    ### VISTAS DE LOS MODELOS #############################################################

    # Pantalla del menu del modelo regresion lineal
    def linear_regression_view(self):
        # Limpiar la ventana actual
        for widget in self.root.winfo_children():
            widget.destroy()

        # Título centrado
        self.lbl_titulo = tk.Label(self.root, text="Regresion Lineal", font=("Arial", 30),
                                   bg=self.label_bg_color_verde, fg=self.text_blanco)
        self.lbl_titulo.pack(pady=100)

        # Botones del menú
        self.btn_regresion = tk.Button(self.root, text="Entrenar modelo", font=("Arial", 18),
                                       bg=self.btn_bg_color, fg=self.text_blanco, command=self.train_linear_regression_model_view)
        self.btn_regresion.pack(pady=20)

        self.btn_probit = tk.Button(self.root, text="Predecir datos", font=("Arial", 18),
                                    bg=self.btn_bg_color, fg=self.text_blanco, command=self.predict_downloads_linear_regression_view)
        self.btn_probit.pack(pady=20)

        self.btn_regresar = tk.Button(self.root, text="Regresar", font=("Arial", 10),
                                      bg=self.btn_bg_color_amarillo, command=self.create_main_menu_view)
        self.btn_regresar.pack(pady=10)

    # Pantalla del menu de modelo probit
    def probit_view(self):
        # Limpiar la ventana actual
        for widget in self.root.winfo_children():
            widget.destroy()

        # Título centrado
        self.lbl_titulo = tk.Label(self.root, text="Probit", font=("Arial", 30),
                                   bg=self.label_bg_color_verde, fg=self.text_blanco)
        self.lbl_titulo.pack(pady=100)

        # Botones del menú
        self.btn_entrenar = tk.Button(self.root, text="Entrenar modelo", font=("Arial", 18),
                                       bg=self.btn_bg_color, fg=self.text_blanco, command=self.train_probit_model_view)
        self.btn_entrenar.pack(pady=20)

        self.btn_predecir = tk.Button(self.root, text="Predecir datos", font=("Arial", 18),
                                    bg=self.btn_bg_color, fg=self.text_blanco, command=self.predict_probit_view)
        self.btn_predecir.pack(pady=20)

        self.btn_regresar = tk.Button(self.root, text="Regresar", font=("Arial", 10),
                                      bg=self.btn_bg_color_amarillo, command=self.create_main_menu_view)
        self.btn_regresar.pack(pady=10)

    # Pantalla del menu de modelo logit
    def logit_view(self):
        # Limpiar la ventana actual
        for widget in self.root.winfo_children():
            widget.destroy()

        # Título centrado
        self.lbl_titulo = tk.Label(self.root, text="Logit", font=("Arial", 30),
                                   bg=self.label_bg_color_verde, fg=self.text_blanco)
        self.lbl_titulo.pack(pady=100)

        # Botones del menú
        self.btn_entrenar = tk.Button(self.root, text="Entrenar modelo", font=("Arial", 18),
                                       bg=self.btn_bg_color, fg=self.text_blanco, command=self.train_logit_model_view)
        self.btn_entrenar.pack(pady=20)

        self.btn_predecir = tk.Button(self.root, text="Predecir datos", font=("Arial", 18),
                                        bg=self.btn_bg_color, fg=self.text_blanco, command=self.predict_logit_view)
        self.btn_predecir.pack(pady=20)

        self.btn_regresar = tk.Button(self.root, text="Regresar", font=("Arial", 10),
                                      bg=self.btn_bg_color_amarillo, command=self.create_main_menu_view)
        self.btn_regresar.pack(pady=10)



    ### VISTAS DE ENTRENAMIENTO  ############################################################

    # Pantalla de entrenamiento de modelo de regresion lineal
    def train_linear_regression_model_view(self):
        try:
            self.clear_window()

            # Título centrado
            self.lbl_titulo1 = tk.Label(self.root, text="Entrenamiento de datos", font=("Arial", 20),
                                        bg=self.label_bg_color_verde, fg=self.text_blanco)
            self.lbl_titulo1.pack(pady=0.4)
            self.lbl_titulo2 = tk.Label(self.root, text="Recuerde ingresar los valores separados por un espacio",
                                        font=("Arial", 12), bg=self.bg_color, fg=self.text_blanco)
            self.lbl_titulo2.pack(pady=20)

            # Funcionalidad
            self.lbl_funcionalidad = tk.Label(self.root, text="Funcionalidad:", bg=self.bg_color, fg=self.text_blanco)
            self.lbl_funcionalidad.pack(pady=5)
            validate_func = (self.root.register(self._validate_numeric_entry), '%P')
            self.entry_funcionalidad = tk.Entry(self.root, validate='key', validatecommand=validate_func,
                                                bg=self.gris, fg=self.text_blanco)
            self.entry_funcionalidad.pack(pady=5)

            # Calidad del código
            self.lbl_calidad = tk.Label(self.root, text="Calidad del código:", bg=self.bg_color, fg=self.text_blanco)
            self.lbl_calidad.pack(pady=5)
            validate_cali = (self.root.register(self._validate_numeric_entry), '%P')
            self.entry_calidad = tk.Entry(self.root, validate='key', validatecommand=validate_cali,
                                          bg=self.gris, fg=self.text_blanco)
            self.entry_calidad.pack(pady=5)

            # Facilidad de uso
            self.lbl_facilidad = tk.Label(self.root, text="Facilidad de uso:", bg=self.bg_color, fg=self.text_blanco)
            self.lbl_facilidad.pack(pady=5)
            validate_facil = (self.root.register(self._validate_numeric_entry), '%P')
            self.entry_facilidad = tk.Entry(self.root, validate='key', validatecommand=validate_facil,
                                            bg=self.gris, fg=self.text_blanco)
            self.entry_facilidad.pack(pady=5)

            # Compatibilidad
            self.lbl_compatibilidad = tk.Label(self.root, text="Compatibilidad (0 o 1):", bg=self.bg_color, fg=self.text_blanco)
            self.lbl_compatibilidad.pack(pady=5)
            vcmd = (self.root.register(self._validate_bit_entry), '%P')
            self.entry_compatibilidad = tk.Entry(self.root, validate='key', validatecommand=vcmd,
                                                 bg=self.gris, fg=self.text_blanco)
            self.entry_compatibilidad.pack(pady=5)

            self.lbl_descargas = tk.Label(self.root, text="Descargas:", bg=self.bg_color, fg=self.text_blanco)
            self.lbl_descargas.pack(pady=5)
            validate_descargas = (self.root.register(self._validate_numeric_entry), '%P')
            self.entry_descargas = tk.Entry(self.root, validate='key', validatecommand=validate_descargas,
                                            bg=self.gris, fg=self.text_blanco)
            self.entry_descargas.pack(pady=5)


            # Botones
            self.frame_botones = tk.Frame(self.root, bg=self.bg_color)
            self.frame_botones.pack(pady=10)

            self.btn_entrenar = tk.Button(self.frame_botones, text="Entrenar", command=self.train_linear_regresion_model_handler,
                                          bg=self.btn_bg_color, fg=self.text_blanco)
            self.btn_entrenar.pack(side=tk.LEFT, padx=5)

            self.btn_limpiar = tk.Button(self.frame_botones, text="Limpiar", command=self._clean_fields,
                                         bg=self.btn_bg_color, fg=self.text_blanco)
            self.btn_limpiar.pack(side=tk.LEFT, padx=5)

            self.btn_predecir = tk.Button(self.frame_botones, text="Predecir datos",
                                          command=self.predict_downloads_linear_regression_view,
                                          bg=self.btn_bg_color, fg=self.text_blanco)
            self.btn_predecir.pack(side=tk.LEFT, padx=5)


            # Resultados
            self.lbl_resultados = tk.Label(self.root, text="Resultados:", bg=self.bg_color, fg=self.text_blanco)
            self.lbl_resultados.pack(pady=5)
            self.txt_resultados = tk.Text(self.root, height=8, width=40, bg=self.gris, fg=self.text_blanco)
            self.txt_resultados.pack(pady=10)

            self.btn_regresar = tk.Button(self.root, text="Regresar", command=self.linear_regression_view,
                                          bg=self.btn_bg_color_amarillo)
            self.btn_regresar.pack(pady=15)
        except Exception as e:
            messagebox.showerror("Error", f"Error al entrenar el modelo: {e}")

    # Pantalla de entrenamiento de modelo probit
    def train_probit_model_view(self):
        try:
            self.clear_window()

            # Título centrado
            self.lbl_titulo1 = tk.Label(self.root, text="Entrenamiento de datos", font=("Arial", 24),
                                        bg=self.label_bg_color_verde, fg=self.text_blanco)
            self.lbl_titulo1.pack(pady=1)
            self.lbl_titulo2 = tk.Label(self.root, text="Recuerde ingresar los valores separados por un espacio",
                                        font=("Arial", 15), bg=self.bg_color, fg=self.text_blanco)
            self.lbl_titulo2.pack(pady=20)

            # Portabilidad
            self.lbl_portabilidad = tk.Label(self.root, text="Portabilidad (0 o 1):",
                                             bg=self.bg_color, fg=self.text_blanco)
            self.lbl_portabilidad.pack(pady=5)
            vcmd = (self.root.register(self._validate_bit_entry), '%P')
            self.entry_portabilidad = tk.Entry(self.root, validate='key', validatecommand=vcmd,
                                               bg=self.gris, fg=self.text_blanco)
            self.entry_portabilidad.pack(pady=5)

            self.lbl_exito = tk.Label(self.root, text="Exito:", bg=self.bg_color, fg=self.text_blanco)
            self.lbl_exito.pack(pady=5)
            self.entry_exito = tk.Entry(self.root, validate='key', validatecommand=vcmd,
                                        bg=self.gris, fg=self.text_blanco)
            self.entry_exito.pack(pady=5)

            # Botones
            self.frame_botones = tk.Frame(self.root, bg=self.bg_color)
            self.frame_botones.pack(pady=70)

            self.btn_entrenar_probit = tk.Button(self.frame_botones, text="Entrenar", command=self.train_probit_model_handler,
                                                 bg=self.btn_bg_color, fg=self.text_blanco)
            self.btn_entrenar_probit.pack(side=tk.LEFT, padx=5)

            self.btn_limpiar = tk.Button(self.frame_botones, text="Limpiar", command=self._clean_fields,
                                         bg=self.btn_bg_color, fg=self.text_blanco)
            self.btn_limpiar.pack(side=tk.LEFT, padx=5)

            self.btn_predecir = tk.Button(self.frame_botones, text="Predecir datos",
                                          command=self.predict_probit_view, bg=self.btn_bg_color, fg=self.text_blanco)
            self.btn_predecir.pack(side=tk.LEFT, padx=5)

            # Resultados
            self.lbl_resultados = tk.Label(self.root, text="Resultados:", bg=self.bg_color, fg=self.text_blanco)
            self.lbl_resultados.pack(pady=5)
            self.txt_resultados = tk.Text(self.root, height=8, width=40, bg=self.gris, fg=self.text_blanco)
            self.txt_resultados.pack(pady=10)

            self.btn_regresar = tk.Button(self.root, text="Regresar", command=self.probit_view,
                                          bg=self.btn_bg_color_amarillo)
            self.btn_regresar.pack(pady=15)
        except Exception as e:
            messagebox.showerror("Error", f"Error al entrenar el modelo: {e}")

    # Pantalla de entrenamiento de modelo logit
    def train_logit_model_view(self):
        try:
            self.clear_window()

            # Título centrado
            self.lbl_titulo1 = tk.Label(self.root, text="Entrenamiento de datos", font=("Arial", 24),
                                        bg=self.label_bg_color_verde, fg=self.text_blanco)
            self.lbl_titulo1.pack(pady=1)
            self.lbl_titulo2 = tk.Label(self.root, text="Recuerde ingresar los valores separados por un espacio",
                                        font=("Arial", 15), bg=self.bg_color, fg=self.text_blanco)
            self.lbl_titulo2.pack(pady=20)

            # Portabilidad
            self.lbl_portabilidad = tk.Label(self.root, text="Portabilidad (0 o 1):", bg=self.bg_color, fg=self.text_blanco)
            self.lbl_portabilidad.pack(pady=5)
            vcmd = (self.root.register(self._validate_bit_entry), '%P')
            self.entry_portabilidad = tk.Entry(self.root, validate='key', validatecommand=vcmd,
                                               bg=self.gris, fg=self.text_blanco)
            self.entry_portabilidad.pack(pady=5)

            self.lbl_exito = tk.Label(self.root, text="Exito:", bg=self.bg_color, fg=self.text_blanco)
            self.lbl_exito.pack(pady=5)
            self.entry_exito = tk.Entry(self.root, validate='key', validatecommand=vcmd,
                                        bg=self.gris, fg=self.text_blanco)
            self.entry_exito.pack(pady=5)


            # Botones
            self.frame_botones = tk.Frame(self.root, bg=self.bg_color)
            self.frame_botones.pack(pady=70)

            self.btn_entrenar_probit = tk.Button(self.frame_botones, text="Entrenar", command=self.train_logit_model_handler,
                                                 bg=self.btn_bg_color, fg=self.text_blanco)
            self.btn_entrenar_probit.pack(side=tk.LEFT, padx=5)

            self.btn_limpiar = tk.Button(self.frame_botones, text="Limpiar", command=self._clean_fields,
                                         bg=self.btn_bg_color, fg=self.text_blanco)
            self.btn_limpiar.pack(side=tk.LEFT, padx=5)

            self.btn_predecir = tk.Button(self.frame_botones, text="Predecir datos",command=self.predict_logit_view,
                                          bg=self.btn_bg_color, fg=self.text_blanco)
            self.btn_predecir.pack(side=tk.LEFT, padx=5)

            # Resultados
            self.lbl_resultados = tk.Label(self.root, text="Resultados:", bg=self.bg_color, fg=self.text_blanco)
            self.lbl_resultados.pack(pady=5)
            self.txt_resultados = tk.Text(self.root, height=8, width=40, bg=self.gris, fg=self.text_blanco)
            self.txt_resultados.pack(pady=10)

            self.btn_regresar = tk.Button(self.root, text="Regresar", command=self.logit_view, bg=self.btn_bg_color_amarillo)
            self.btn_regresar.pack(pady=15)
        except Exception as e:
            messagebox.showerror("Error", f"Error al entrenar el modelo: {e}")



    ### VISTAS DE PREDICCION  ###############################################################

    # Pantalla de prediccion de modelo de regresion lineal
    def predict_downloads_linear_regression_view(self):
        try:
            self.clear_window()

            # Título centrado
            self.lbl_titulo = tk.Label(self.root, text="Prediccion de datos.", font=("Arial", 24),
                                       bg=self.label_bg_color_verde, fg=self.text_blanco)
            self.lbl_titulo.pack(pady=10)

            # Funcionalidad
            self.lbl_funcionalidad = tk.Label(self.root, text="Funcionalidad:", bg=self.bg_color, fg=self.text_blanco)
            self.lbl_funcionalidad.pack(pady=5)
            validate_func = (self.root.register(self._validate_numeric_entry), '%P')
            self.entry_funcionalidad = tk.Entry(self.root, validate='key', validatecommand=validate_func,
                                                bg=self.gris, fg=self.text_blanco)
            self.entry_funcionalidad.pack(pady=5)

            # Calidad del código
            self.lbl_calidad = tk.Label(self.root, text="Calidad del código:", bg=self.bg_color, fg=self.text_blanco)
            self.lbl_calidad.pack(pady=5)
            validate_cali = (self.root.register(self._validate_numeric_entry), '%P')
            self.entry_calidad = tk.Entry(self.root, validate='key', validatecommand=validate_cali,
                                          bg=self.gris, fg=self.text_blanco)
            self.entry_calidad.pack(pady=5)

            # Facilidad de uso
            self.lbl_facilidad = tk.Label(self.root, text="Facilidad de uso:", bg=self.bg_color, fg=self.text_blanco)
            self.lbl_facilidad.pack(pady=5)
            validate_facil = (self.root.register(self._validate_numeric_entry), '%P')
            self.entry_facilidad = tk.Entry(self.root, validate='key', validatecommand=validate_facil,
                                            bg=self.gris, fg=self.text_blanco)
            self.entry_facilidad.pack(pady=5)

            # Compatibilidad
            self.lbl_compatibilidad = tk.Label(self.root, text="Compatibilidad (0 o 1):", bg=self.bg_color, fg=self.text_blanco)
            self.lbl_compatibilidad.pack(pady=5)
            vcmd = (self.root.register(self._validate_bit_entry), '%P')
            self.entry_compatibilidad = tk.Entry(self.root, validate='key', validatecommand=vcmd,
                                                 bg=self.gris, fg=self.text_blanco)
            self.entry_compatibilidad.pack(pady=5)


            # Botones
            self.frame_botones = tk.Frame(self.root, bg=self.bg_color)
            self.frame_botones.pack(pady=50)

            self.btn_predecir = tk.Button(self.frame_botones, text="Predecir Descargas", command=self.predict_downloads_handler,
                                          bg=self.btn_bg_color, fg=self.text_blanco)
            self.btn_predecir.pack(side=tk.LEFT, padx=5)

            self.btn_aleatorio = tk.Button(self.frame_botones, text="Generar Datos Aleatorios y Predecir",
                                           command=self.random_linear_regresion_predictions,
                                           bg=self.btn_bg_color, fg=self.text_blanco)
            self.btn_aleatorio.pack(side=tk.LEFT, padx=5)

            self.btn_limpiar = tk.Button(self.frame_botones, text="Limpiar", command=self._clean_fields,
                                         bg=self.btn_bg_color, fg=self.text_blanco)
            self.btn_limpiar.pack(side=tk.LEFT, padx=5)

            # Resultados
            self.lbl_resultados = tk.Label(self.root, text="Resultados:", bg=self.bg_color, fg=self.text_blanco)
            self.lbl_resultados.pack(pady=5)
            self.txt_resultados = tk.Text(self.root, height=8, width=40, bg=self.gris, fg=self.text_blanco)
            self.txt_resultados.pack(pady=5)

            self.btn_regresar = tk.Button(self.root, text="Regresar", command=self.linear_regression_view,
                                          bg=self.btn_bg_color_amarillo)
            self.btn_regresar.pack(pady=15)
        except Exception as e:
            messagebox.showerror("Error", f"Error al predecir datos: {e}")

    # Pantalla de prediccion de modelo probit
    def predict_probit_view(self):
        try:
            self.clear_window()

            # Título centrado
            self.lbl_titulo = tk.Label(self.root, text="Prediccion de exito.", font=("Arial", 24),
                                       bg=self.label_bg_color_verde, fg=self.text_blanco)
            self.lbl_titulo.pack(pady=20)

            # Compatibilidad
            self.lbl_portabilidad = tk.Label(self.root, text="Portabilidad (0 o 1):",
                                             bg=self.bg_color, fg=self.text_blanco)
            self.lbl_portabilidad.pack(pady=5)
            vcmd = (self.root.register(self._validate_bit_entry), '%P')
            self.entry_portabilidad = tk.Entry(self.root, validate='key', validatecommand=vcmd,
                                               bg=self.gris, fg=self.text_blanco)
            self.entry_portabilidad.pack(pady=5)


            # Botones
            self.frame_botones = tk.Frame(self.root, bg=self.bg_color)
            self.frame_botones.pack(pady=70)

            self.btn_predecir = tk.Button(self.frame_botones, text="Predecir exito", command=self.predict_probit_handler,
                                          bg=self.btn_bg_color, fg=self.text_blanco)
            self.btn_predecir.pack(side=tk.LEFT, padx=5)

            self.btn_aleatorio = tk.Button(self.frame_botones, text="Generar Datos Aleatorios y Predecir",
                                           command=self.random_probit_predictions,
                                           bg=self.btn_bg_color, fg=self.text_blanco)
            self.btn_aleatorio.pack(side=tk.LEFT, padx=5)

            self.btn_limpiar = tk.Button(self.frame_botones, text="Limpiar", command=self._clean_fields,
                                         bg=self.btn_bg_color, fg=self.text_blanco)
            self.btn_limpiar.pack(side=tk.LEFT, padx=5)

            # Resultados
            self.lbl_resultados = tk.Label(self.root, text="Resultados:", bg=self.bg_color, fg=self.text_blanco)
            self.lbl_resultados.pack(pady=5)
            self.txt_resultados = tk.Text(self.root, height=8, width=40, bg=self.gris, fg=self.text_blanco)
            self.txt_resultados.pack(pady=5)

            self.btn_regresar = tk.Button(self.root, text="Regresar", command=self.probit_view, bg=self.btn_bg_color_amarillo)
            self.btn_regresar.pack(pady=15)
        except Exception as e:
            messagebox.showerror("Error", f"Error al predecir datos: {e}")

    # Pantalla de prediccion de modelo logit
    def predict_logit_view(self):
        try:
            self.clear_window()

            # Título centrado
            self.lbl_titulo = tk.Label(self.root, text="Prediccion de exito.", font=("Arial", 24),
                                       bg=self.label_bg_color_verde, fg=self.text_blanco)
            self.lbl_titulo.pack(pady=20)

            # Compatibilidad
            self.lbl_portabilidad = tk.Label(self.root, text="Portabilidad (0 o 1):", bg=self.bg_color, fg=self.text_blanco)
            self.lbl_portabilidad.pack(pady=5)
            vcmd = (self.root.register(self._validate_bit_entry), '%P')
            self.entry_portabilidad = tk.Entry(self.root, validate='key', validatecommand=vcmd,
                                               bg=self.gris, fg=self.text_blanco)
            self.entry_portabilidad.pack(pady=5)

            # Botones
            self.frame_botones = tk.Frame(self.root, bg=self.bg_color)
            self.frame_botones.pack(pady=70)

            self.btn_predecir = tk.Button(self.frame_botones, text="Predecir exito", command=self.predict_logit_handler,
                                          bg=self.btn_bg_color, fg=self.text_blanco)
            self.btn_predecir.pack(side=tk.LEFT, padx=5)

            self.btn_aleatorio = tk.Button(self.frame_botones, text="Generar Datos Aleatorios y Predecir",
                                           command=self.random_logit_predictions,
                                           bg=self.btn_bg_color, fg=self.text_blanco)
            self.btn_aleatorio.pack(side=tk.LEFT, padx=5)

            self.btn_limpiar = tk.Button(self.frame_botones, text="Limpiar", command=self._clean_fields,
                                         bg=self.btn_bg_color, fg=self.text_blanco)
            self.btn_limpiar.pack(side=tk.LEFT, padx=5)

            # Resultados
            self.lbl_resultados = tk.Label(self.root, text="Resultados:", bg=self.bg_color, fg=self.text_blanco)
            self.lbl_resultados.pack(pady=5)
            self.txt_resultados = tk.Text(self.root, height=8, width=40, bg=self.gris, fg=self.text_blanco)
            self.txt_resultados.pack(pady=5)

            self.btn_regresar = tk.Button(self.root, text="Regresar", command=self.logit_view, bg=self.btn_bg_color_amarillo)
            self.btn_regresar.pack(pady=15)
        except Exception as e:
            messagebox.showerror("Error", f"Error al predecir datos: {e}")



    ### HANDLER ENTRENAMIENTOS ###################################################################
    # los metodos handler nos sirven para estandarizar los datos que vamos a usar
    # posteriormente para el entrenamiento y la prediccion. son metodos intermedios
    # entre las vistas y la logica de los modelos.
    # estos metodos reciben los datos entregados por la interfaz y tambien entrega
    # las respuestas procesadas por la logica de cada modelo

    # Handler para el entrenamiento de regresion lineal
    def train_linear_regresion_model_handler(self):
        try:
            # self.entrenar_modelo_regresion_lineal()

            funcionalidad = self.entry_funcionalidad.get()
            calidad_codigo = self.entry_calidad.get()
            facilidad_uso = self.entry_facilidad.get()
            compatibilidad = self.entry_compatibilidad.get()
            descargas = self.entry_descargas.get()

            self.modelo_regresion_lineal.train_model(
                funcionalidad, calidad_codigo, facilidad_uso, compatibilidad, descargas)

            self._clean_fields()
            self.txt_resultados.insert(tk.END, "Entrenamiento exitoso")
        except Exception as e:
            messagebox.showwarning("Error", f"Error en la predicción: {e}")

    # Handler para el entrenamiento probit
    def train_probit_model_handler(self):
        try:
            portabilidad = self.entry_portabilidad.get()
            exito = self.entry_exito.get()

            self.modelo_probit.train_model(portabilidad, exito)

            self._clean_fields()
            self.txt_resultados.insert(tk.END, "Entrenamiento exitoso")
        except Exception as e:
            messagebox.showwarning("Advertencia", f"Problema en la predicción: {str(e)}")

    # Handler para el entrenamiento logit
    def train_logit_model_handler(self):
        try:
            portabilidad = self.entry_portabilidad.get()
            exito = self.entry_exito.get()

            self.modelo_logit.train_model(portabilidad, exito)

            self._clean_fields()
            self.txt_resultados.insert(tk.END, "Entrenamiento exitoso")
        except Exception as e:
            messagebox.showwarning("Advertencia", f"Problema en la predicción: {str(e)}")



    ### HANDLER PREDICCIONES #######################################################################
    # los metodos handler nos sirven para estandarizar los datos que vamos a usar
    # posteriormente para el entrenamiento y la prediccion. son metodos intermedios
    # entre las vistas y la logica de los modelos.
    # estos metodos reciben los datos entregados por la interfaz y tambien entrega
    # las respuestas procesadas por la logica de cada modelo

    # Handler para la prediccion de regresion lineal
    def predict_downloads_handler(self):
        try:
            funcionalidad = float(self.entry_funcionalidad.get())
            calidad_codigo = float(self.entry_calidad.get())
            facilidad_uso = float(self.entry_facilidad.get())
            compatibilidad = float(self.entry_compatibilidad.get())

            prediccion = self.modelo_regresion_lineal.predict_downloads(funcionalidad, calidad_codigo, facilidad_uso,
                                                                        compatibilidad)

            resultado = f"Predicción del número de descargas: {prediccion:.2f}\n"
            self.txt_resultados.insert(tk.END, resultado)
            self.modelo_regresion_lineal.plot_predictions()
            self.modelo_regresion_lineal.plot_statistics_one()
            self.modelo_regresion_lineal.plot_statistics_two()
        except Exception as e:
            messagebox.showerror("Error", f"Error en la predicción: {e}")

    # Handler para la prediccion probit
    def predict_probit_handler(self):
        try:
            portabilidad = float(self.entry_portabilidad.get())

            prediccion = self.modelo_probit.predict_success(portabilidad)

            resultado = f"El software sera: " + prediccion + "\n"
            self.txt_resultados.insert(tk.END, resultado)
            self.modelo_probit.plot_statistics_one()
            self.modelo_probit.plot_statistics_two()
        except Exception as e:
            messagebox.showerror("Error", f"Error en la predicción: {e}")

    # Handler para la prediccion logit
    def predict_logit_handler(self):
        try:
            portabilidad = float(self.entry_portabilidad.get())

            prediccion = self.modelo_logit.predict_success(portabilidad)

            resultado = f"El software sera: " + prediccion + "\n"
            self.txt_resultados.insert(tk.END, resultado)
            self.modelo_logit.plot_statistics_one()
            self.modelo_logit.plot_statistics_two()
        except Exception as e:
            messagebox.showerror("Error", f"Error en la predicción: {e}")



    ### RANDOM DATA ###############################################################################
    # Estos metodos se encargan de generar 50 datos aleatorios de cada input
    # y asi poder probar la prediccion de los datos con multiples entradas
    # generadas aleatoriamente

    # Metodo que genera entradas aleatorias para la prediccion
    # de regresion lineal
    def random_linear_regresion_predictions(self):
        self.txt_resultados.delete('1.0', tk.END)
        for i in range(50):
            funcionalidad = random.uniform(0, 10)
            calidad = random.uniform(0, 10)
            facilidad = random.uniform(0, 10)
            compatibilidad = random.choice([0, 1])

            # Predicción del número de descargas
            prediccion = self.modelo_regresion_lineal.predict_downloads(funcionalidad, calidad,
                                                                        facilidad,compatibilidad,)

            resultado = (f"Software {i + 1}:\n"
                         f"Funcionalidad: {funcionalidad:.2f}\n"
                         f"Calidad del código: {calidad:.2f}\n"
                         f"Facilidad de uso: {facilidad:.2f}\n"
                         f"Compatibilidad: {compatibilidad}\n"
                         f"Predicción del número de descargas: {prediccion:.2f}\n\n")
            self.txt_resultados.insert(tk.END, resultado)

    # Metodo que genera entradas aleatorias para la prediccion probit
    def random_probit_predictions(self):
        self.txt_resultados.delete('1.0', tk.END)
        for i in range(50):
            portability = random.choice([0, 1])

            # Predicción del número de descargas
            prediccion = self.modelo_probit.predict_success(portability)

            resultado = (f"Software {i + 1}:\n"
                         f"Portabilidad: {portability:.2f}\n"
                         f"Predicción: {prediccion}\n\n")
            self.txt_resultados.insert(tk.END, resultado)

    # Metodo que genera entradas aleatorias para la prediccion logit
    def random_logit_predictions(self):
        self.txt_resultados.delete('1.0', tk.END)
        for i in range(50):
            portability = random.choice([0, 1])

            # Predicción del número de descargas
            prediccion = self.modelo_logit.predict_success(portability)

            resultado = (f"Software {i + 1}:\n"
                         f"Portabilidad: {portability:.2f}\n"
                         f"Predicción: {prediccion}\n\n")
            self.txt_resultados.insert(tk.END, resultado)



    ### VALIDAR ENTRADAS ############################################################################
    # Metodos que se encargan de hacer las validaciones en el mismo
    # momento que el usuario esta digitando las entradas en los inputs

    # Metodo que valida una entrada binaria
    def _validate_bit_entry(self, new_value):
        # Verificar que cada carácter sea '0', '1' o un espacio
        if all(char in '01 ' for char in new_value):
            # Dividir el texto en partes usando espacios
            parts = new_value.split(' ')
            # Verificar que cada parte sea '0' o '1' (permitir partes vacías por múltiples espacios)
            return all(part in ('', '0', '1') for part in parts)

        return False

    # Metodo que valida que una entrada sea numerica
    def _validate_numeric_entry(self, new_value):
        # Verificar si el nuevo valor es válido (dígitos, puntos, o espacios)
        if all(char.isdigit() or char == '.' or char == ' ' for char in new_value):
            # Verificar si hay más de un punto
            if new_value.count('.') <= 1:
                return True
        return False


    # Metodo que nos ayuda a limpiar todos los campos
    def _clean_fields(self):
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
        if hasattr(self, 'entry_descargas'):
            self.entry_descargas.delete(0, tk.END)
        if hasattr(self, 'entry_exito'):
            self.entry_exito.delete(0, tk.END)
        self.txt_resultados.delete('1.0', tk.END)

    # Metodo que nos ayuda a eliminar las ventanas que no estamos usando
    def clear_window(self):
        for widget in self.root.winfo_children():
            widget.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
