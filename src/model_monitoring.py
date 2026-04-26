import os
import joblib
import numpy as np
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import ks_2samp, chi2_contingency

from cargar_datos import cargarDatos


def obtener_ruta_modelo():
    ruta_actual = os.path.dirname(os.path.abspath(__file__))
    ruta_proyecto = os.path.dirname(ruta_actual)
    return os.path.join(ruta_proyecto, "models", "best_model.pkl")


def cargar_modelo():
    ruta_modelo = obtener_ruta_modelo()

    if os.path.exists(ruta_modelo):
        return joblib.load(ruta_modelo)

    return None


def preparar_datos_para_prediccion(df):
    df_pred = df.copy()

    valores_validos = ["Decreciente", "Estable", "Creciente"]
    df_pred = df_pred[df_pred["tendencia_ingresos"].isin(valores_validos)].copy()

    if "puntaje" in df_pred.columns:
        df_pred = df_pred.drop(columns=["puntaje"])

    df_pred = df_pred[df_pred["tipo_credito"] != 68].copy()
    df_pred["tipo_credito"] = df_pred["tipo_credito"].astype(str)

    df_pred["ratio_deuda_ingreso"] = np.where(
        (df_pred["salario_cliente"].notnull()) &
        (df_pred["salario_cliente"] != 0),
        df_pred["saldo_total"] / df_pred["salario_cliente"],
        np.nan
    )

    df_pred["ratio_deuda_ingreso"] = df_pred["ratio_deuda_ingreso"].replace(
        [np.inf, -np.inf],
        np.nan
    )

    if "Pago_atiempo" in df_pred.columns:
        X_pred = df_pred.drop(columns=["Pago_atiempo"])
    else:
        X_pred = df_pred.copy()

    # Convertir cualquier pd.NA restante a np.nan para que sklearn pueda imputar
    X_pred = X_pred.replace({pd.NA: np.nan})

    return df_pred, X_pred


def calcular_drift_numerico(df_base, df_nuevo, columnas_numericas):
    resultados = []

    for col in columnas_numericas:
        serie_base = df_base[col].dropna()
        serie_nueva = df_nuevo[col].dropna()

        if len(serie_base) > 0 and len(serie_nueva) > 0:
            stat, p_value = ks_2samp(serie_base, serie_nueva)

            resultados.append({
                "variable": col,
                "tipo_variable": "numérica",
                "metrica": "KS Test",
                "estadistico": stat,
                "p_value": p_value,
                "drift_detectado": p_value < 0.05
            })

    return pd.DataFrame(resultados)


def calcular_drift_categorico(df_base, df_nuevo, columnas_categoricas):
    resultados = []

    for col in columnas_categoricas:
        base_counts = df_base[col].astype(str).value_counts()
        nuevo_counts = df_nuevo[col].astype(str).value_counts()

        categorias = sorted(set(base_counts.index).union(set(nuevo_counts.index)))

        tabla = pd.DataFrame({
            "historico": [base_counts.get(cat, 0) for cat in categorias],
            "nuevo": [nuevo_counts.get(cat, 0) for cat in categorias]
        }).T

        if tabla.shape[1] > 1:
            try:
                stat, p_value, _, _ = chi2_contingency(tabla)

                resultados.append({
                    "variable": col,
                    "tipo_variable": "categórica",
                    "metrica": "Chi-cuadrado",
                    "estadistico": stat,
                    "p_value": p_value,
                    "drift_detectado": p_value < 0.05
                })
            except ValueError:
                pass

    return pd.DataFrame(resultados)


def asignar_riesgo(row):
    if row["drift_detectado"] and row["p_value"] < 0.01:
        return "🔴 Alto"
    elif row["drift_detectado"]:
        return "🟡 Medio"
    else:
        return "🟢 Bajo"


def generar_recomendacion(row):
    if row["riesgo"] == "🔴 Alto":
        return "Revisar variable y considerar retraining del modelo."
    elif row["riesgo"] == "🟡 Medio":
        return "Monitorear la variable en próximas ejecuciones."
    else:
        return "Sin acción inmediata requerida."


def generar_tabla_drift(df_base, df_nuevo):
    columnas_numericas = df_base.select_dtypes(include=["int64", "float64"]).columns.tolist()

    if "Pago_atiempo" in columnas_numericas:
        columnas_numericas.remove("Pago_atiempo")

    columnas_categoricas = df_base.select_dtypes(include=["object"]).columns.tolist()

    drift_num = calcular_drift_numerico(df_base, df_nuevo, columnas_numericas)
    drift_cat = calcular_drift_categorico(df_base, df_nuevo, columnas_categoricas)

    drift_df = pd.concat([drift_num, drift_cat], ignore_index=True)

    if not drift_df.empty:
        drift_df["riesgo"] = drift_df.apply(asignar_riesgo, axis=1)
        drift_df["recomendacion"] = drift_df.apply(generar_recomendacion, axis=1)

    return drift_df, columnas_numericas, columnas_categoricas


def graficar_numerica(df_base, df_nuevo, variable):
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(
        df_base[variable].dropna(),
        bins=30,
        alpha=0.5,
        label="Histórico",
        density=True
    )

    ax.hist(
        df_nuevo[variable].dropna(),
        bins=30,
        alpha=0.5,
        label="Nuevo",
        density=True
    )

    ax.set_title(f"Distribución de {variable}")
    ax.set_xlabel(variable)
    ax.set_ylabel("Densidad")
    ax.legend()

    return fig


def graficar_categorica(df_base, df_nuevo, variable):
    historico = df_base[variable].astype(str).value_counts(normalize=True)
    nuevo = df_nuevo[variable].astype(str).value_counts(normalize=True)

    categorias = sorted(set(historico.index).union(set(nuevo.index)))

    comparacion = pd.DataFrame({
        "Histórico": [historico.get(cat, 0) for cat in categorias],
        "Nuevo": [nuevo.get(cat, 0) for cat in categorias]
    }, index=categorias)

    fig, ax = plt.subplots(figsize=(8, 5))
    comparacion.plot(kind="bar", ax=ax)

    ax.set_title(f"Distribución categórica de {variable}")
    ax.set_xlabel(variable)
    ax.set_ylabel("Proporción")
    ax.legend()

    plt.xticks(rotation=45)
    plt.tight_layout()

    return fig


def analisis_temporal(df_nuevo):
    if "fecha_prestamo" not in df_nuevo.columns:
        return None

    df_temp = df_nuevo.copy()
    df_temp["fecha_prestamo"] = pd.to_datetime(df_temp["fecha_prestamo"], errors="coerce")
    df_temp = df_temp.dropna(subset=["fecha_prestamo"])

    if df_temp.empty:
        return None

    df_temp["mes"] = df_temp["fecha_prestamo"].dt.to_period("M").astype(str)

    conteo_mensual = df_temp.groupby("mes").size().reset_index(name="cantidad_registros")

    return conteo_mensual


def main():
    st.set_page_config(page_title="Monitoreo de Data Drift", layout="wide")

    st.title("Monitoreo de Data Drift")
    st.write(
        "Dashboard para comparar datos históricos vs datos nuevos, "
        "identificar drift y generar alertas de monitoreo."
    )

    df_base = cargarDatos("Base_de_datos_historica.xlsx")
    df_nuevo = cargarDatos("Base_de_datos_nueva.xlsx")

    modelo = cargar_modelo()

    st.subheader("Resumen de datasets")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Registros históricos", df_base.shape[0])

    with col2:
        st.metric("Registros nuevos", df_nuevo.shape[0])

    with col3:
        st.metric("Variables", df_base.shape[1])

    st.divider()

    st.subheader("Pronósticos sobre datos nuevos")

    if modelo is not None:
        df_pred, X_pred = preparar_datos_para_prediccion(df_nuevo)

        predicciones = modelo.predict(X_pred)
        probabilidades = modelo.predict_proba(X_pred)[:, 1]

        df_resultado = df_pred.copy()
        df_resultado["pronostico_pago_atiempo"] = predicciones
        df_resultado["probabilidad_pago_atiempo"] = probabilidades

        st.write("Tabla de datos nuevos con pronósticos del modelo.")
        st.dataframe(df_resultado.head(50))

    else:
        st.warning("No se encontró modelo guardado. Ejecuta primero model_train_evaluation.py.")

    st.divider()

    st.subheader("Métricas de Data Drift")

    drift_df, columnas_numericas, columnas_categoricas = generar_tabla_drift(df_base, df_nuevo)

    if not drift_df.empty:
        st.dataframe(drift_df)

        total_drift = int(drift_df["drift_detectado"].sum())

        if total_drift > 0:
            st.warning(f"Se detectó drift en {total_drift} variables.")
        else:
            st.success("No se detectó drift significativo.")

        st.subheader("Semáforo de riesgo")
        riesgo_counts = drift_df["riesgo"].value_counts()
        st.write(riesgo_counts)

    else:
        st.info("No fue posible calcular métricas de drift.")

    st.divider()

    st.subheader("Visualización de distribuciones")

    tipo_variable = st.radio(
        "Selecciona tipo de variable",
        ["Numérica", "Categórica"]
    )

    if tipo_variable == "Numérica":
        variable = st.selectbox("Selecciona una variable numérica", columnas_numericas)
        fig = graficar_numerica(df_base, df_nuevo, variable)
        st.pyplot(fig)

    else:
        variable = st.selectbox("Selecciona una variable categórica", columnas_categoricas)
        fig = graficar_categorica(df_base, df_nuevo, variable)
        st.pyplot(fig)

    st.divider()

    st.subheader("Análisis temporal")

    temporal_df = analisis_temporal(df_nuevo)

    if temporal_df is not None:
        st.write("Cantidad de registros nuevos por mes.")
        st.dataframe(temporal_df)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(temporal_df["mes"], temporal_df["cantidad_registros"], marker="o")
        ax.set_title("Evolución temporal de registros nuevos")
        ax.set_xlabel("Mes")
        ax.set_ylabel("Cantidad de registros")
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.info("No se encontró información temporal válida.")

    st.divider()

    st.subheader("Recomendaciones automáticas")

    if not drift_df.empty:
        variables_con_drift = drift_df[drift_df["drift_detectado"]]

        if not variables_con_drift.empty:
            for _, row in variables_con_drift.iterrows():
                st.warning(
                    f"{row['variable']} ({row['riesgo']}): {row['recomendacion']}"
                )
        else:
            st.success("No se requieren acciones inmediatas.")
    else:
        st.info("No hay resultados de drift para generar recomendaciones.")


if __name__ == "__main__":
    main()