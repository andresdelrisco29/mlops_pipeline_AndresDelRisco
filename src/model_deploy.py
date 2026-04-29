# Importamos FastAPI para crear la API web
from fastapi import FastAPI

# Importamos BaseModel para validar la estructura de los datos de entrada
from pydantic import BaseModel

# Tipados para indicar listas, diccionarios y cualquier tipo de dato
from typing import List, Dict, Any

# Pandas para convertir los datos recibidos en DataFrame
import pandas as pd

# Joblib para cargar el modelo entrenado guardado en archivo .pkl
import joblib

# OS para trabajar con rutas del sistema operativo
import os

## CREACIÓN DE LA APLICACIÓN FASTAPI

# Inicializar FastAPI
app = FastAPI(

    # Nombre visible en la documentación Swagger
    title="API de Predicción - Proyecto Integrador Módulo 5",

    # Descripción visible en /docs
    description="API para disponibilizar el modelo entrenado mediante FastAPI.",

    # Versión de la API
    version="1.0.0"
)

## UBICAR Y CARGAR EL MODELO

# Obtenemos la ruta absoluta del archivo actual:
# src/model_deploy.py
CURRENT_FILE = os.path.abspath(__file__)

# Subimos un nivel para llegar a /src
SRC_DIR = os.path.dirname(CURRENT_FILE)

# Subimos otro nivel para llegar a la raíz del proyecto:
# mlops_pipeline_AndresDelRisco/
BASE_DIR = os.path.dirname(SRC_DIR)

# Construimos la ruta completa al modelo:
# mlops_pipeline_AndresDelRisco/models/best_model.pkl
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.pkl")

# Cargamos el modelo entrenado en memoria
modelo = joblib.load(MODEL_PATH)



## DEFINIR ESTRUCTURA DE LOS DATOS DE ENTRADA


# Esta clase define cómo debe venir el JSON enviado al endpoint /predict
class PredictionRequest(BaseModel):

    # Esperamos una clave llamada "data" que contenga una lista de registros. Cada registro será un diccionario:
    #{"edad": 30, "ingresos": 5000}
    data: List[Dict[str, Any]]

## ENDPOINT INICIAL
# Endpoint GET para probar que la API está viva
@app.get("/")

# Función que responde cuando alguien entra a /
def home():

    return {
        "mensaje": "API funcionando correctamente",
        "modelo": "best_model.pkl",
        "endpoint_prediccion": "/predict"
    }


## ENDPOINT DE PREDICCIÓN

# Endpoint POST para enviar datos y obtener predicciones
@app.post("/predict")

# Recibe datos validados con PredictionRequest
def predict(request: PredictionRequest):

    # Convertimos la lista JSON en DataFrame
    # Esto es necesario porque sklearn trabaja con tablas
    input_df = pd.DataFrame(request.data)

    # Ejecutamos predicción con el modelo cargado
    predicciones = modelo.predict(input_df)

    # Retornamos las predicciones como lista JSON
    return {
        "mensaje": "Predicción realizada correctamente",
        "predicciones": predicciones.tolist()
    }