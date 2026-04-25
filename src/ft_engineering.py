import numpy as np

from cargar_datos import cargarDatos

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder


def prepararDatos(nombre_archivo="Base_de_datos.xlsx"):
    """
    Carga, limpia y transforma la base de datos para entrenamiento.

    Retorna:
    X_train, X_test, y_train, y_test, preprocessor
    """

    # Cargar datos
    df = cargarDatos(nombre_archivo)

    # Limpiar tendencia_ingresos
    valores_validos = ["Decreciente", "Estable", "Creciente"]
    df = df[df["tendencia_ingresos"].isin(valores_validos)].copy()

    # Eliminar posible leakage
    if "puntaje" in df.columns:
        df = df.drop(columns=["puntaje"])

    # Eliminar outlier en tipo_credito
    df = df[df["tipo_credito"] != 68].copy()

    # Tratar tipo_credito como categórica nominal
    df["tipo_credito"] = df["tipo_credito"].astype(str)

    # Feature engineering: ratio deuda / ingreso
    df["ratio_deuda_ingreso"] = np.where(
        (df["salario_cliente"].notnull()) & (df["salario_cliente"] != 0),
        df["saldo_total"] / df["salario_cliente"],
        np.nan
    )

    # Separar variables predictoras y target
    X = df.drop(columns=["Pago_atiempo"])
    y = df["Pago_atiempo"]

    # Detectar tipos de variables
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

    # Definir ordinales
    ord_cols = ["tendencia_ingresos"]

    # Quitar ordinales de numéricas/categóricas para evitar duplicidad
    num_cols = [col for col in num_cols if col not in ord_cols]
    cat_cols = [col for col in cat_cols if col not in ord_cols]

    # Excluir fecha del modelo
    if "fecha_prestamo" in num_cols:
        num_cols.remove("fecha_prestamo")

    if "fecha_prestamo" in cat_cols:
        cat_cols.remove("fecha_prestamo")

    # Pipelines de preprocesamiento
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    ordinal_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ordinal", OrdinalEncoder(categories=[["Decreciente", "Estable", "Creciente"]]))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_pipeline, num_cols),
        ("cat", categorical_pipeline, cat_cols),
        ("ord", ordinal_pipeline, ord_cols)
    ])

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    return X_train, X_test, y_train, y_test, preprocessor


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, preprocessor = prepararDatos()

    print("Datos preparados correctamente")
    print("X_train:", X_train.shape)
    print("X_test:", X_test.shape)
    print("y_train:", y_train.shape)
    print("y_test:", y_test.shape)