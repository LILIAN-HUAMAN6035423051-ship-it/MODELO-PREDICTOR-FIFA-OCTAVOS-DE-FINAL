# ============================================================
# PREDICCIÓN DE PARTIDOS DEL MUNDIAL 2026
# ============================================================

import numpy as np
import pandas as pd

from Src.features import construir_caracteristicas_partido
from Src.evaluate import preparar_datos_modelo


# ============================================================
# VARIABLES UTILIZADAS POR LOS MODELOS
# ============================================================

VARIABLES_MODELO = [
    "prom_goles_local",
    "prom_goles_visitante",
    "prom_recibidos_local",
    "prom_recibidos_visitante",
    "victorias_ult5_local",
    "victorias_ult5_visitante",
    "puntos_ult5_local",
    "puntos_ult5_visitante",
    "dif_goles",
    "dif_recibidos",
    "dif_victorias",
    "dif_puntos"
]


# ============================================================
# CONSTRUIR DATOS PARA PREDICCIÓN
# ============================================================

def construir_datos_prediccion(
    cruces,
    fecha_referencia,
    base_partidos
):
    """
    Construye las características históricas de los partidos
    que serán predichos.
    """

    registros = []

    fecha_referencia = pd.to_datetime(
        fecha_referencia
    )


    for equipo_1, equipo_2 in cruces:

        caracteristicas = construir_caracteristicas_partido(
            equipo_1,
            equipo_2,
            fecha_referencia,
            base_partidos
        )


        registro = {

            "equipo_1": equipo_1,

            "equipo_2": equipo_2

        }


        registro.update(
            caracteristicas
        )


        registros.append(
            registro
        )


    datos_prediccion = pd.DataFrame(
        registros
    )


    return datos_prediccion


# ============================================================
# PREDECIR CLASIFICADOS
# ============================================================

def predecir_clasificados(
    cruces,
    fecha_referencia,
    base_partidos,
    nombre_modelo,
    modelo,
    scaler
):
    """
    Utiliza el modelo seleccionado para estimar el equipo
    clasificado en cada enfrentamiento.
    """

    datos_prediccion = construir_datos_prediccion(
        cruces,
        fecha_referencia,
        base_partidos
    )


    X_prediccion = datos_prediccion[
        VARIABLES_MODELO
    ].copy()


    X_modelo = preparar_datos_modelo(
        nombre_modelo,
        X_prediccion,
        scaler
    )


    # ========================================================
    # PREDICCIÓN DE CLASE
    # ========================================================

    predicciones = modelo.predict(
        X_modelo
    )


    # ========================================================
    # PROBABILIDADES
    # ========================================================

    probabilidades = modelo.predict_proba(
        X_modelo
    )


    resultados = datos_prediccion[
        [
            "equipo_1",
            "equipo_2"
        ]
    ].copy()


    resultados["prediccion"] = (
        predicciones
    )


    resultados["prob_equipo_1"] = (
        probabilidades[:, 1]
    )


    resultados["prob_equipo_2"] = (
        probabilidades[:, 0]
    )


    resultados["clasificado_predicho"] = np.where(
        resultados["prediccion"] == 1,
        resultados["equipo_1"],
        resultados["equipo_2"]
    )


    return resultados


# ============================================================
# TRADUCIR NOMBRES DE EQUIPOS
# ============================================================

def traducir_resultados(
    resultados,
    traduccion_equipos
):
    """
    Traduce los nombres internos de las selecciones para la
    presentación final de resultados.
    """

    tabla_final = resultados.copy()


    tabla_final["Equipo 1"] = (
        tabla_final["equipo_1"]
        .replace(
            traduccion_equipos
        )
    )


    tabla_final["Equipo 2"] = (
        tabla_final["equipo_2"]
        .replace(
            traduccion_equipos
        )
    )


    tabla_final["Probabilidad Equipo 1"] = (
        tabla_final["prob_equipo_1"]
    )


    tabla_final["Probabilidad Equipo 2"] = (
        tabla_final["prob_equipo_2"]
    )


    tabla_final["Clasificado predicho"] = (
        tabla_final["clasificado_predicho"]
        .replace(
            traduccion_equipos
        )
    )


    tabla_final = tabla_final[
        [
            "Equipo 1",
            "Equipo 2",
            "Probabilidad Equipo 1",
            "Probabilidad Equipo 2",
            "Clasificado predicho"
        ]
    ]


    return tabla_final


# ============================================================
# ANALIZAR DIFERENCIA DE PROBABILIDADES
# ============================================================

def analizar_diferencia_probabilidades(
    tabla_predicciones
):
    """
    Analiza la separación existente entre las probabilidades
    estimadas para ambas selecciones.
    """

    analisis = tabla_predicciones.copy()


    analisis["Diferencia probabilística"] = abs(
        analisis["Probabilidad Equipo 1"]
        -
        analisis["Probabilidad Equipo 2"]
    )


    def clasificar_diferencia(valor):

        if valor >= 0.40:

            return "Diferencia amplia"

        elif valor >= 0.20:

            return "Diferencia moderada"

        else:

            return "Diferencia reducida"


    analisis["Nivel de diferencia"] = (
        analisis[
            "Diferencia probabilística"
        ].apply(
            clasificar_diferencia
        )
    )


    analisis = analisis.sort_values(
        by="Diferencia probabilística",
        ascending=False
    ).reset_index(
        drop=True
    )


    return analisis