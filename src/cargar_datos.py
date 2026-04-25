import os
import pandas as pd


def cargarDatos(nombre_archivo="Base_de_datos.xlsx"):
    """
    Carga un archivo Excel ubicado en la raíz del proyecto.

    Parámetros:
    nombre_archivo : str
        Nombre del archivo Excel a cargar.

    Retorna:
    DataFrame de pandas.
    """

    # Ruta del archivo actual: src/cargar_datos.py
    ruta_actual = os.path.dirname(os.path.abspath(__file__))

    # Subir un nivel para llegar a la raíz del proyecto
    ruta_proyecto = os.path.dirname(ruta_actual)

    # Construir ruta completa al archivo
    ruta_archivo = os.path.join(ruta_proyecto, nombre_archivo)

    # Cargar archivo Excel
    df = pd.read_excel(ruta_archivo)

    return df


if __name__ == "__main__":
    df = cargarDatos()
    print("Datos cargados correctamente")
    print("Dimensiones:", df.shape)
    print(df.head())