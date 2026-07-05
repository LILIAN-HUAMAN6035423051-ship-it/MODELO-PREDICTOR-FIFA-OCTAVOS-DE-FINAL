# ============================================================
# EVALUACIÓN DEL MODELO
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    roc_auc_score
)

from Src.train_models import MODELOS_ESCALADOS


# ============================================================
# PREPARACIÓN DE DATOS PARA EL MODELO
# ============================================================

def preparar_datos_modelo(
    nombre_modelo,
    X,
    scaler
):
    """
    Prepara las variables predictoras según los requerimientos
    del modelo seleccionado.
    """

    if nombre_modelo in MODELOS_ESCALADOS:

        return scaler.transform(X)

    return X


# ============================================================
# OBTENER PREDICCIONES
# ============================================================

def obtener_predicciones(
    nombre_modelo,
    modelo,
    X,
    scaler
):
    """
    Genera las predicciones del modelo seleccionado.
    """

    X_modelo = preparar_datos_modelo(
        nombre_modelo,
        X,
        scaler
    )

    predicciones = modelo.predict(
        X_modelo
    )

    return predicciones


# ============================================================
# OBTENER PROBABILIDADES
# ============================================================

def obtener_probabilidades(
    nombre_modelo,
    modelo,
    X,
    scaler
):
    """
    Obtiene las probabilidades estimadas por el modelo.
    """

    X_modelo = preparar_datos_modelo(
        nombre_modelo,
        X,
        scaler
    )

    probabilidades = modelo.predict_proba(
        X_modelo
    )

    return probabilidades


# ============================================================
# MATRIZ DE CONFUSIÓN
# ============================================================

def graficar_matriz_confusion(
    y_real,
    y_predicho,
    nombre_modelo
):
    """
    Genera la matriz de confusión del modelo.
    """

    matriz = confusion_matrix(
        y_real,
        y_predicho
    )

    visualizacion = ConfusionMatrixDisplay(
        confusion_matrix=matriz,
        display_labels=[
            "No clasifica local",
            "Clasifica local"
        ]
    )

    visualizacion.plot()

    plt.title(
        f"Matriz de confusión - {nombre_modelo}"
    )

    plt.tight_layout()

    plt.savefig(
        "Results/matriz_confusion_svm.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.show()

    return matriz


# ============================================================
# CURVA ROC
# ============================================================

def graficar_curva_roc(
    y_real,
    probabilidades,
    nombre_modelo
):
    """
    Genera la curva ROC y calcula el valor AUC.
    """

    probabilidad_positiva = probabilidades[:, 1]

    falsos_positivos, verdaderos_positivos, _ = roc_curve(
        y_real,
        probabilidad_positiva
    )

    auc = roc_auc_score(
        y_real,
        probabilidad_positiva
    )

    plt.figure(
        figsize=(9, 7)
    )

    plt.plot(
        falsos_positivos,
        verdaderos_positivos,
        label=f"{nombre_modelo} (AUC = {auc:.4f})"
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Clasificación aleatoria"
    )

    plt.xlabel(
        "Tasa de falsos positivos"
    )

    plt.ylabel(
        "Tasa de verdaderos positivos"
    )

    plt.title(
        f"Curva ROC - {nombre_modelo}"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        "Results/curva_roc_svm.png",
        dpi=300,
        bbox_inches="tight"
    )
    
    plt.show()

    return auc


# ============================================================
# VALIDACIÓN PARTIDO POR PARTIDO
# ============================================================

def validar_partidos(
    datos_partidos,
    predicciones,
    probabilidades
):
    """
    Compara la predicción del modelo con el clasificado real
    de cada partido.
    """

    validacion = datos_partidos[
        [
            "fase",
            "home_team",
            "away_team",
            "clasifica_local"
        ]
    ].copy()

    validacion["prediccion_modelo"] = (
        predicciones
    )

    validacion["prob_equipo_1"] = (
        probabilidades[:, 1]
    )

    validacion["prob_equipo_2"] = (
        probabilidades[:, 0]
    )

    validacion["clasificado_real"] = np.where(
        validacion["clasifica_local"] == 1,
        validacion["home_team"],
        validacion["away_team"]
    )

    validacion["clasificado_predicho"] = np.where(
        validacion["prediccion_modelo"] == 1,
        validacion["home_team"],
        validacion["away_team"]
    )

    validacion["resultado_prediccion"] = np.where(
        validacion["clasificado_real"]
        ==
        validacion["clasificado_predicho"],
        "CORRECTO",
        "INCORRECTO"
    )

    return validacion


# ============================================================
# RESUMEN DE VALIDACIÓN
# ============================================================

def resumir_validacion(
    validacion
):
    """
    Resume los aciertos y errores obtenidos durante
    la validación partido por partido.
    """

    total = len(validacion)

    aciertos = (
        validacion["resultado_prediccion"]
        == "CORRECTO"
    ).sum()

    errores = total - aciertos

    porcentaje_acierto = (
        aciertos / total
    ) * 100

    resumen = {
        "total_partidos": total,
        "aciertos": aciertos,
        "errores": errores,
        "porcentaje_acierto": porcentaje_acierto
    }

    return resumen

# ============================================================
# GRÁFICA COMPARATIVA DE MODELOS
# ============================================================

def graficar_comparacion_modelos(
    resultados,
    ruta_guardado="Results/comparacion_modelos.png"
):
    """
    Genera y guarda una gráfica comparativa del rendimiento
    de los modelos evaluados.
    """

    metricas = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-score"
    ]

    datos_grafica = resultados.set_index(
        "Modelo"
    )[metricas]


    ax = datos_grafica.plot(
        kind="bar",
        figsize=(14, 7)
    )


    ax.set_title(
        "Comparación del rendimiento de los modelos de Machine Learning"
    )

    ax.set_xlabel(
        "Modelo"
    )

    ax.set_ylabel(
        "Valor de la métrica"
    )

    ax.set_ylim(
        0,
        1
    )


    plt.xticks(
        rotation=30,
        ha="right"
    )


    plt.legend(
        title="Métrica"
    )


    plt.tight_layout()


    plt.savefig(
        ruta_guardado,
        dpi=300,
        bbox_inches="tight"
    )


    plt.show()


    print(
        f"Gráfica comparativa guardada en: "
        f"{ruta_guardado}"
    )