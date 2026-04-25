import pandas as pd
import matplotlib.pyplot as plt

from ft_engineering import prepararDatos

from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier
from catboost import CatBoostClassifier


def build_model(model, preprocessor):
    """
    Construye un pipeline completo con preprocesamiento y modelo.
    """
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    return pipeline


def summarize_classification(model, X_train, X_test, y_train, y_test):
    """
    Entrena un modelo y retorna métricas de evaluación enfocadas en la clase 0.
    Clase 0 = no paga.
    """

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    report = classification_report(
        y_test,
        y_pred,
        output_dict=True,
        zero_division=0
    )

    roc_auc = roc_auc_score(y_test, y_proba)

    results = {
        "precision_0": report["0"]["precision"],
        "recall_0": report["0"]["recall"],
        "f1_0": report["0"]["f1-score"],
        "accuracy": report["accuracy"],
        "roc_auc": roc_auc
    }

    return results


def entrenarEvaluarModelos():
    """
    Prepara los datos, entrena varios modelos supervisados
    y genera una tabla comparativa de resultados.
    """

    X_train, X_test, y_train, y_test, preprocessor = prepararDatos()

    # Peso para clase minoritaria: 0 = no paga
    clase_0 = (y_train == 0).sum()
    clase_1 = (y_train == 1).sum()
    peso_clase_0 = clase_1 / clase_0

    class_weights_catboost = [peso_clase_0, 1]

    models = {
        "Logistic Regression": build_model(
            LogisticRegression(
                max_iter=5000,
                class_weight="balanced",
                random_state=42
            ),
            preprocessor
        ),

        "Random Forest": build_model(
            RandomForestClassifier(
                n_estimators=200,
                max_depth=None,
                random_state=42,
                class_weight="balanced"
            ),
            preprocessor
        ),

        "XGBoost": build_model(
            XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                eval_metric="logloss"
            ),
            preprocessor
        ),

        "CatBoost": build_model(
            CatBoostClassifier(
                iterations=200,
                depth=6,
                learning_rate=0.1,
                random_state=42,
                verbose=0,
                class_weights=class_weights_catboost
            ),
            preprocessor
        )
    }

    results = {}

    for name, model in models.items():
        print(f"Entrenando modelo: {name}")
        results[name] = summarize_classification(
            model,
            X_train,
            X_test,
            y_train,
            y_test
        )

    results_df = pd.DataFrame(results).T
    results_df = results_df.sort_values(by="roc_auc", ascending=False)

    return results_df


def graficarResultados(results_df):
    """
    Genera gráfico comparativo de Recall clase 0 y ROC-AUC.
    """

    metrics_to_plot = results_df[["recall_0", "roc_auc"]]

    metrics_to_plot.plot(kind="bar", figsize=(10, 6))

    plt.title("Comparación de modelos: Recall clase 0 y ROC-AUC")
    plt.xlabel("Modelo")
    plt.ylabel("Valor de la métrica")
    plt.xticks(rotation=45)
    plt.legend(title="Métricas")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    results_df = entrenarEvaluarModelos()

    print("\nTabla comparativa de modelos:")
    print(results_df)

    graficarResultados(results_df)