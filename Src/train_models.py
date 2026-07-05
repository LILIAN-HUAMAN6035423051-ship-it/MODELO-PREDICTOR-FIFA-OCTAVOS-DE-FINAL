# ============================================================
# ENTRENAMIENTO Y COMPARACIÓN DE MODELOS
# ============================================================

import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from xgboost import XGBClassifier


# ============================================================
# MODELOS QUE REQUIEREN ESCALADO
# ============================================================

MODELOS_ESCALADOS = [
    "Regresión Logística",
    "KNN",
    "SVM"
]


# ============================================================
# CREAR MODELOS
# ============================================================

def crear_modelos():
    """
    Crea los seis algoritmos utilizados en la comparación.
    """

    modelos = {

        "Regresión Logística": LogisticRegression(
            max_iter=5000,
            random_state=42
        ),

        "Árbol de Decisión": DecisionTreeClassifier(
            random_state=42
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=42
        ),

        "KNN": KNeighborsClassifier(
            n_neighbors=5
        ),

        "SVM": SVC(
            probability=True,
            random_state=42
        ),

        "XGBoost": XGBClassifier(
            random_state=42,
            eval_metric="logloss"
        )

    }

    return modelos


# ============================================================
# ESCALAR DATOS
# ============================================================

def escalar_datos(
    X_train,
    X_test
):
    """
    Estandariza las variables predictoras.

    El escalador se ajusta únicamente con los datos
    de entrenamiento.
    """

    scaler = StandardScaler()

    X_train_escalado = scaler.fit_transform(
        X_train
    )

    X_test_escalado = scaler.transform(
        X_test
    )

    return (
        X_train_escalado,
        X_test_escalado,
        scaler
    )


# ============================================================
# ENTRENAR Y COMPARAR MODELOS
# ============================================================

def entrenar_y_comparar_modelos(
    X_train,
    X_test,
    y_train,
    y_test
):
    """
    Entrena los seis algoritmos y compara su rendimiento
    mediante Accuracy, Precision, Recall y F1-score.
    """

    modelos = crear_modelos()

    (
        X_train_escalado,
        X_test_escalado,
        scaler

    ) = escalar_datos(
        X_train,
        X_test
    )


    resultados = []

    modelos_entrenados = {}


    for nombre, modelo in modelos.items():

        print(
            f"Entrenando: {nombre}"
        )


        # ====================================================
        # SELECCIONAR DATOS
        # ====================================================

        if nombre in MODELOS_ESCALADOS:

            datos_train = X_train_escalado

            datos_test = X_test_escalado

        else:

            datos_train = X_train

            datos_test = X_test


        # ====================================================
        # ENTRENAMIENTO
        # ====================================================

        modelo.fit(
            datos_train,
            y_train
        )


        # ====================================================
        # PREDICCIÓN
        # ====================================================

        predicciones = modelo.predict(
            datos_test
        )


        # ====================================================
        # CÁLCULO DE MÉTRICAS
        # ====================================================

        accuracy = accuracy_score(
            y_test,
            predicciones
        )

        precision = precision_score(
            y_test,
            predicciones,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            predicciones,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            predicciones,
            zero_division=0
        )


        resultados.append({

            "Modelo": nombre,

            "Accuracy": accuracy,

            "Precision": precision,

            "Recall": recall,

            "F1-score": f1

        })


        modelos_entrenados[
            nombre
        ] = modelo


    # ========================================================
    # CREAR TABLA DE COMPARACIÓN
    # ========================================================

    resultados = pd.DataFrame(
        resultados
    )


    resultados = resultados.sort_values(
        by="F1-score",
        ascending=False
    ).reset_index(
        drop=True
    )


    return (
        resultados,
        modelos_entrenados,
        scaler
    )


# ============================================================
# SELECCIONAR MEJOR MODELO
# ============================================================

def seleccionar_mejor_modelo(
    resultados,
    modelos_entrenados
):
    """
    Selecciona automáticamente el modelo con mayor F1-score.
    """

    mejor_nombre = resultados.iloc[0][
        "Modelo"
    ]


    mejor_modelo = modelos_entrenados[
        mejor_nombre
    ]


    mejor_accuracy = resultados.iloc[0][
        "Accuracy"
    ]


    mejor_f1 = resultados.iloc[0][
        "F1-score"
    ]


    return (
        mejor_nombre,
        mejor_modelo,
        mejor_accuracy,
        mejor_f1
    )