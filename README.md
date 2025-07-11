# 🍎 Valoración Nutricional Antropométrica con IA Multi-Modelo

Esta aplicación innovadora aprovecha el poder de la Inteligencia Artificial para ofrecer una valoración nutricional antropométrica precisa y exhaustiva. Utiliza una combinación de tres modelos de Machine Learning de última generación para predecir el estado nutricional, proporcionando un análisis comparativo robusto y recomendaciones basadas en datos.

## 🧠 Modelos de Machine Learning Utilizados

Hemos integrado y optimizado tres algoritmos de Machine Learning de alto rendimiento para garantizar la máxima precisión y fiabilidad:

### 1. XGBoost (Extreme Gradient Boosting)
* **Modelo ensemble de alto rendimiento:** Combina múltiples árboles de decisión para lograr una precisión superior.
* **Excelente para datos estructurados:** Idóneo para el tipo de datos médicos y antropométricos.
* **Manejo automático de valores faltantes:** Preprocesamiento de datos simplificado.

### 2. Regresión Lineal (Linear Regression)
* **Modelo interpretable y rápido:** Ofrece una comprensión clara de las relaciones entre variables.
* **Relaciones lineales entre variables:** Ideal para identificar tendencias directas en los datos.
* **Base sólida para comparación:** Proporciona un punto de referencia para evaluar el rendimiento de modelos más complejos.

### 3. Random Forest (Bosques Aleatorios)
* **Ensemble de árboles de decisión:** Agrupa múltiples árboles para mejorar la robustez y la precisión.
* **Robusto contra overfitting:** Minimiza el riesgo de que el modelo se ajuste demasiado a los datos de entrenamiento.
* **Buena generalización:** Capaz de predecir con precisión en nuevos datos no vistos.

## 📊 Características Principales

Nuestra aplicación está diseñada pensando en la facilidad de uso y la profundidad del análisis:

* **Análisis Comparativo:** Evalúa simultáneamente el rendimiento de los tres modelos para cada valoración.
* **Métricas Avanzadas:** Proporciona un conjunto completo de métricas de evaluación: precisión, exactitud, recall, F1-score, AUC-ROC y nivel de confianza.
* **Reporte Profesional:** Genera un PDF detallado con un análisis comparativo de los resultados de cada modelo.
* **Identificación del Mejor Modelo:** Recomienda automáticamente el modelo con el mejor rendimiento basado en las métricas evaluadas.
* **Visualizaciones Interactivas:** Incluye gráficos de probabilidades por modelo para una comprensión visual de las predicciones.

## 🏥 Aplicación Clínica

Esta herramienta es invaluable para profesionales de la salud y nutricionistas, permitiendo:

* Evaluación precisa de Talla para la Edad.
* Evaluación de IMC para la Talla.
* Análisis con datos completos del paciente para una valoración integral.

## 🛠️ Tecnologías Utilizadas

Hemos empleado un stack de tecnologías robusto y moderno para desarrollar esta aplicación:

* **Python:** Lenguaje de programación principal.
* **Streamlit:** Utilizado para crear una interfaz web interactiva y amigable.
* **XGBoost, Scikit-learn:** Bibliotecas fundamentales para la implementación de los modelos de Machine Learning.
* **Matplotlib, Seaborn:** Herramientas esenciales para la generación de visualizaciones de datos impactantes.
* **FPDF:** Biblioteca para la generación de reportes en formato PDF.
* **Joblib:** Utilizado para la serialización y deserialización de los modelos, permitiendo su carga y uso eficiente.

## 📁 Estructura del Proyecto

Dentro de la carpeta `DATA`, encontrarás el archivo CSV con los datos utilizados para el entrenamiento y las predicciones, así como los archivos `.py` de cada modelo (XGBoost, Regresión Lineal, Random Forest) listos para ser utilizados.

## 🚀 ¡Prueba la aplicación aquí!

Puedes interactuar con nuestra aplicación directamente en la plataforma de Streamlit:

[https://valoracionnutri-s5inv99opte7tvkdqzn8jx.streamlit.app/](https://valoracionnutri-s5inv99opte7tvkdqzn8jx.streamlit.app/)
