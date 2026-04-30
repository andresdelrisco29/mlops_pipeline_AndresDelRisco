# Proyecto Integrador Módulo 5 - Modelo de Riesgo Crediticio

## Descripción general

Este proyecto tiene como objetivo desarrollar un flujo de trabajo tipo MLOps para un modelo de riesgo crediticio. El caso de negocio consiste en anticipar el comportamiento de pago de clientes a partir de información histórica de créditos, características financieras, laborales y de comportamiento crediticio.

La variable objetivo del proyecto es:

- `Pago_atiempo`
  - `1`: el cliente paga a tiempo
  - `0`: el cliente no paga a tiempo

El proyecto no solo se enfoca en entrenar modelos predictivos, sino también en estructurar el código de forma modular, versionar el desarrollo con Git/GitHub, construir pipelines reproducibles, desarrollar un sistema básico de monitoreo de data drift mediante Streamlit, y disponibilizar el modelo final a través de una API REST construida con FastAPI, contenerizada mediante Docker para facilitar su despliegue y portabilidad.

---

## Objetivos del proyecto

- Cargar y analizar una base histórica de créditos.
- Realizar análisis exploratorio de datos.
- Identificar problemas de calidad de datos.
- Detectar posible data leakage.
- Aplicar ingeniería de características.
- Construir pipelines de preprocesamiento con `ColumnTransformer`.
- Entrenar y evaluar modelos supervisados.
- Comparar modelos usando métricas adecuadas para clasificación desbalanceada.
- Guardar el mejor modelo entrenado.
- Crear un dashboard de monitoreo en Streamlit.
- Detectar data drift entre datos históricos y datos nuevos.
- Generar alertas y recomendaciones automáticas de monitoreo.
- Exponer el modelo final mediante una API REST con `FastAPI`.
- Permitir predicciones individuales o por lotes mediante endpoint `/predict`.
- Empaquetar la solución en una imagen `Docker` para facilitar su despliegue.
- Garantizar portabilidad y ejecución reproducible del servicio de inferencia.

---

## Estructura del proyecto

mlops_pipeline_AndresDelRisco/
│
├── src/
│ ├── cargar_datos.py
│ ├── cargar_datos.ipynb
│ ├── comprension_eda.ipynb
│ ├── ft_engineering.py
│ ├── model_train_evaluation.py
│ ├── model_monitoring.py
│ └── model_deploy.py
│
├── models/
│ └── best_model.pkl
│
├── Base_de_datos.xlsx
├── Base_de_datos_historica.xlsx
├── Base_de_datos_nueva.xlsx
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── .gitignore
└── README.md
```

DESCRIPCIÓN DE ARCHIVOS PRINCIPALES
src/cargar_datos.py
Este archivo contiene la función cargarDatos(), encargada de cargar archivos Excel ubicados en la raíz del proyecto.
La función permite cargar distintos archivos mediante el parámetro nombre_archivo, lo que facilita reutilizarla para:
•	base original, 
•	base histórica, 
•	base nueva. 
Ejemplo de uso:
from cargar_datos import cargarDatos

df = cargarDatos("Base_de_datos.xlsx")
Este archivo permite centralizar la ingesta de datos y evitar rutas absolutas locales, favoreciendo la reproducibilidad del proyecto.

src/comprension_eda.ipynb
Notebook utilizado para el análisis exploratorio de datos.
En este notebook se revisaron:
•	dimensiones del dataset, 
•	tipos de datos, 
•	valores nulos, 
•	distribución de la variable objetivo, 
•	desbalance de clases, 
•	distribución de variables categóricas, 
•	correlaciones entre variables numéricas, 
•	posible data leakage. 
Uno de los hallazgos más importantes fue que la variable puntaje presentaba una correlación extremadamente alta con Pago_atiempo, lo cual sugería posible fuga de información. Por esta razón, se decidió excluirla del modelado.

src/ft_engineering.py
Este archivo contiene la función prepararDatos(), encargada de preparar los datos para entrenamiento.
Las principales tareas realizadas son:
•	carga de datos mediante cargarDatos(), 
•	limpieza de valores inválidos en tendencia_ingresos, 
•	eliminación de la variable puntaje por posible data leakage, 
•	eliminación de un valor atípico en tipo_credito, 
•	conversión de tipo_credito a variable categórica, 
•	creación de la variable ratio_deuda_ingreso, 
•	separación de variables predictoras y variable objetivo, 
•	identificación de variables numéricas, categóricas y ordinales, 
•	construcción del ColumnTransformer, 
•	división train/test con estratificación. 
La función retorna:
X_train, X_test, y_train, y_test, preprocessor

src/model_train_evaluation.py
Este archivo entrena y evalúa distintos modelos supervisados.
Modelos evaluados:
•	Logistic Regression 
•	Random Forest 
•	XGBoost 
•	CatBoost 
También contiene dos funciones principales:
-build_model()
Construye un pipeline completo con:
•	preprocesamiento, 
•	modelo supervisado. 
-summarize_classification()
Entrena el modelo y devuelve métricas relevantes para el problema.
Las métricas principales evaluadas fueron:
•	precision_0 
•	recall_0 
•	f1_0 
•	accuracy 
•	roc_auc 
Se prestó especial atención al recall_0, ya que la clase 0 representa a los clientes que no pagan a tiempo, que son los más importantes desde el punto de vista del riesgo crediticio.
El script también genera una tabla comparativa de modelos y guarda el mejor modelo según ROC-AUC en:
models/best_model.pkl

src/model_monitoring.py
Este archivo contiene la aplicación en Streamlit para monitoreo del modelo y detección de data drift.
La app permite:
•	cargar datos históricos, 
•	cargar datos nuevos, 
•	cargar el modelo entrenado, 
•	generar pronósticos sobre datos nuevos, 
•	calcular métricas de drift, 
•	visualizar distribuciones históricas vs nuevas, 
•	generar semáforos de riesgo, 
•	emitir recomendaciones automáticas. 
Para ejecutar el dashboard:
python -m streamlit run src/model_monitoring.py

model_deploy.py 
Aplicación desarrollada en `FastAPI` para cargar el modelo entrenado y disponibilizarlo mediante una API REST. Incluye endpoint raíz de validación y endpoint `/predict` para generar predicciones a partir de datos en formato JSON.

Dockerfile
 Archivo de configuración utilizado para construir la imagen Docker del proyecto. Define el entorno de ejecución, instalación de dependencias y comando de arranque de la API.

.dockerignore 
Archivo que excluye carpetas y archivos innecesarios del proceso de construcción de la imagen Docker, optimizando tamaño y tiempo de build.
-----------------------------------------------------------------------------
Bases de datos utilizadas
Base_de_datos.xlsx
Base original del proyecto.
Base_de_datos_historica.xlsx
Corresponde al 90% inicial de los datos ordenados por fecha. Se utiliza como referencia histórica para entrenamiento y monitoreo.
Base_de_datos_nueva.xlsx
Corresponde al 10% final de los datos ordenados por fecha. Se utiliza como muestra de datos nuevos para simular información reciente en producción.
Esta división permite hacer una comparación más realista para data drift, ya que respeta el orden temporal de los registros.
--------------------------------------------------------------
Análisis exploratorio de datos
Durante el EDA se identificaron varios aspectos relevantes:
Desbalance de clases
La variable objetivo presenta un desbalance importante. La mayoría de los clientes pagan a tiempo, mientras que la clase de clientes que no pagan es minoritaria.
Esto implica que métricas como accuracy no son suficientes para evaluar el modelo, ya que un modelo podría predecir mayoritariamente la clase dominante y aun así obtener una accuracy alta.
Por esta razón, se priorizaron métricas como:
•	recall de la clase 0, 
•	F1-score de la clase 0, 
•	ROC-AUC. 

Valores faltantes
Se encontraron valores faltantes en variables relacionadas con saldos, mora, ingresos y comportamiento financiero.
En lugar de imputar manualmente antes del entrenamiento, se decidió manejar los valores faltantes dentro del pipeline mediante SimpleImputer, para evitar data leakage y asegurar reproducibilidad.

Variable tendencia_ingresos
La variable tendencia_ingresos contiene categorías esperadas como:
•	Decreciente 
•	Estable 
•	Creciente 
También se detectaron valores inválidos o inconsistentes. Por ello, se filtraron los registros que no pertenecían a estas categorías válidas.
Esta variable fue tratada como categórica ordinal y transformada mediante OrdinalEncoder.

Variable tipo_credito
Aunque tipo_credito aparece como numérica, se interpretó como una variable categórica nominal, ya que representa códigos de tipos de crédito y no una magnitud continua.
Por esta razón, fue convertida a texto y tratada mediante OneHotEncoder.
También se identificó un valor atípico 68, presente en un único registro, el cual fue eliminado por considerarse inconsistente.

Data leakage en puntaje
Se detectó una correlación extremadamente alta entre la variable puntaje y la variable objetivo Pago_atiempo.
Además, al probar un modelo utilizando solo esta variable, el desempeño fue casi perfecto, lo que confirmó que esta variable probablemente contenía información posterior al evento de pago o una transformación muy cercana al target.
Por esta razón, puntaje fue eliminada del modelado.
La variable puntaje_datacredito, en cambio, no mostró una correlación tan fuerte con el target, por lo que se mantuvo.

Ingeniería de características
Se creó la variable:
ratio_deuda_ingreso
Calculada como:
saldo_total / salario_cliente
Esta variable busca capturar la proporción entre la deuda total del cliente y su ingreso, lo cual es relevante para analizar capacidad de pago.
Cuando el salario era nulo o igual a cero, se asignó NaN, permitiendo que el pipeline se encargara posteriormente de la imputación.
------------------------------------------------------------------
Preprocesamiento
El preprocesamiento se construyó mediante ColumnTransformer, separando las variables en tres grupos:
-Variables numéricas
Procesadas con:
SimpleImputer(strategy="median")
-Variables categóricas nominales
Procesadas con:
SimpleImputer(strategy="most_frequent")
OneHotEncoder(handle_unknown="ignore")
-Variables categóricas ordinales
Procesadas con:
SimpleImputer(strategy="most_frequent")
OrdinalEncoder()
Este enfoque permite mantener un flujo reproducible y modular para entrenamiento y predicción.
---------------------------------------------------
Modelos entrenados
Se entrenaron cuatro modelos supervisados:
-Logistic Regression
Modelo baseline interpretable. Se utilizó:
class_weight="balanced"
para compensar el desbalance de clases.
-Random Forest
Modelo basado en árboles. Se utilizó:
class_weight="balanced"
Aunque obtuvo el mejor ROC-AUC, no logró detectar clientes de la clase 0 con el umbral por defecto.
-XGBoost
Modelo boosting basado en árboles. Se mantuvo la codificación original del target:
•	0 = no paga 
•	1 = paga 
No se utilizó scale_pos_weight, ya que este parámetro pondera la clase positiva (1) y en este problema la clase de interés es la clase 0.
-CatBoost
Modelo boosting que suele funcionar bien en datos tabulares. Se utilizaron pesos de clase para dar mayor relevancia a la clase minoritaria.
-------------------------------------------
Resultados de modelado
Los modelos fueron evaluados principalmente con:
•	recall_0 
•	roc_auc 
La comparación mostró que:
•	Random Forest obtuvo el mayor ROC-AUC. 
•	Logistic Regression obtuvo el mayor recall para la clase 0. 
•	CatBoost mostró un comportamiento intermedio. 
•	XGBoost no mejoró significativamente la detección de la clase minoritaria. 
El mejor modelo se guardó según ROC-AUC en:
models/best_model.pkl
________________________________________
Monitoreo y Data Drift
Para el avance de monitoreo, se construyó una aplicación en Streamlit que compara:
•	datos históricos, 
•	datos nuevos. 
El objetivo es detectar cambios en la distribución de las variables que puedan afectar el desempeño del modelo.
-------------------------------------------------------
Métricas de drift implementadas
-Kolmogorov-Smirnov Test
Utilizado para variables numéricas.
Permite comparar si dos muestras provienen de distribuciones similares.
Si el p_value es menor a 0.05, se considera que existe evidencia de drift.
-Chi-cuadrado
Utilizado para variables categóricas.
Permite evaluar si la distribución de categorías cambia significativamente entre los datos históricos y los datos nuevos.
----------------------------------------------------------
Dashboard en Streamlit
La aplicación de monitoreo incluye:
•	resumen de datasets, 
•	número de registros históricos, 
•	número de registros nuevos, 
•	tabla de datos nuevos con pronósticos del modelo, 
•	tabla de métricas de drift, 
•	semáforo de riesgo, 
•	visualización de distribuciones numéricas, 
•	visualización de distribuciones categóricas, 
•	análisis temporal, 
•	recomendaciones automáticas. 
---------------------------------------------------------
Semáforo de riesgo
El dashboard asigna niveles de riesgo según los resultados de drift:
•	🟢 Bajo: no se detecta drift significativo. 
•	🟡 Medio: se detecta drift con p_value < 0.05. 
•	🔴 Alto: se detecta drift fuerte con p_value < 0.01. 
---------------------------------------------------------------
Recomendaciones automáticas
Cuando se detecta drift, el sistema genera recomendaciones como:
•	monitorear la variable en próximas ejecuciones, 
•	revisar variables con cambios significativos, 
•	considerar retraining del modelo si el drift es alto. 
---------------------------------------------------------------
Despliegue del modelo

En esta fase del proyecto se implementó la capa de serving del modelo predictivo, permitiendo disponibilizar el artefacto entrenado mediante una API REST desarrollada con `FastAPI`. Adicionalmente, se contenerizó la solución usando `Docker`, facilitando su portabilidad, ejecución reproducible y despliegue en distintos entornos.

## API desarrollada con FastAPI

La aplicación carga automáticamente el modelo almacenado en:
models/best_model.pkl
Este artefacto corresponde a un pipeline completo de scikit-learn, incluyendo preprocesamiento y modelo predictivo.
-Endpoints disponibles
GET /
Permite validar que la API se encuentra operativa.
Respuesta esperada:
{
  "mensaje": "API funcionando correctamente",
  "modelo": "best_model.pkl",
  "endpoint_prediccion": "/predict"
}
-POST /predict
Recibe registros en formato JSON y retorna las predicciones del modelo.
Ejemplo de entrada:
{
  "data": [
    {
      "tipo_credito": "Libre inversion",
      "fecha_prestamo": "2023-01-15",
      "capital_prestado": 5000000,
      "plazo_meses": 24,
      "edad_cliente": 35,
      "tipo_laboral": "Empleado",
      "salario_cliente": 3000000,
      "total_otros_prestamos": 1,
      "cuota_pactada": 350000,
      "puntaje_datacredito": 720,
      "cant_creditosvigentes": 2,
      "huella_consulta": 1,
      "saldo_mora": 0,
      "saldo_total": 2500000,
      "saldo_principal": 2300000,
      "saldo_mora_codeudor": 0,
      "creditos_sectorFinanciero": 2,
      "creditos_sectorCooperativo": 0,
      "creditos_sectorReal": 1,
      "promedio_ingresos_datacredito": 3200000,
      "tendencia_ingresos": "Estable",
      "ratio_deuda_ingreso": 0.35
    }
  ]
}
Ejemplo de salida:
{
  "mensaje": "Predicción realizada correctamente",
  "predicciones": [1]
}
Ejecución local de la API
Desde la raíz del proyecto:
python -m uvicorn src.model_deploy:app --reload
Acceso a documentación interactiva:
http://127.0.0.1:8000/docs

Contenerización con Docker
-Construcción de imagen
docker build -t mlops-pipeline-andres .
-Ejecución del contenedor
docker run -p 8000:8000 mlops-pipeline-andres
-Acceso al servicio desplegado
http://localhost:8000/docs

Cómo ejecutar el proyecto
1. Crear entorno virtual
python -m venv venv
2. Activar entorno virtual
En Windows:
venv\Scripts\activate
3. Instalar dependencias
pip install -r requirements.txt
4. Probar carga de datos
python src/cargar_datos.py
5. Ejecutar feature engineering
python src/ft_engineering.py
6. Entrenar y evaluar modelos
python src/model_train_evaluation.py
7. Ejecutar dashboard de monitoreo
python -m streamlit run src/model_monitoring.py
8. Ejecutar API de predicción
python -m uvicorn src.model_deploy:app –reload
9. Abrir documentación interactiva de la API
http://127.0.0.1:8000/docs
10. Construir imagen Docker
docker build -t mlops-pipeline-andres .
11. Ejecutar contenedor Docker
docker run -p 8000:8000 mlops-pipeline-andres
12. Acceder al servicio desplegado en Docker
http://localhost:8000/docs

________________________________________
Dependencias principales
El proyecto utiliza principalmente:
•	pandas 
•	numpy 
•	scikit-learn 
•	matplotlib 
•	seaborn 
•	xgboost 
•	catboost 
•	scipy 
•	streamlit 
•	joblib 
•	openpyxl 
•	fastapi
•	uvicorn
•	pydantic
•	docker
---------------------------------
Conclusiones
El proyecto permitió construir un flujo completo de ciencia de datos aplicado a riesgo crediticio, integrando no solo entrenamiento de modelos, sino también buenas prácticas de MLOps.
Entre los principales hallazgos se destacan:
•	la clase de clientes que no pagan está fuertemente desbalanceada; 
•	la variable puntaje presentó data leakage y fue eliminada; 
•	el desempeño real de los modelos bajó al eliminar leakage, lo cual permitió una evaluación más honesta; 
•	el monitoreo de data drift es necesario para validar si los datos nuevos mantienen distribuciones similares a los datos históricos; 
•	Streamlit permitió construir una herramienta interactiva para visualizar alertas, métricas y recomendaciones. 
•	FastAPI` permitió disponibilizar el modelo entrenado mediante una API REST lista para consumo por aplicaciones externas.
•	La contenerización con `Docker` facilitó una ejecución estandarizada, portable y reproducible en distintos entornos.

