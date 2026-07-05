# ============================================================
# PROCESAMIENTO Y CARGA DE DATOS
# ============================================================

import pandas as pd


def cargar_datos(ruta_resultados, ruta_penales):
    """
    Carga las bases históricas de resultados y tandas de penales.
    """

    resultados = pd.read_csv(ruta_resultados)
    penales = pd.read_csv(ruta_penales)

    return resultados, penales


def preparar_resultados(resultados):
    """
    Prepara la base histórica de partidos internacionales.
    """

    datos = resultados.copy()

    # Convertir la fecha al formato datetime
    datos["date"] = pd.to_datetime(
        datos["date"]
    )

    # Eliminar registros sin marcador
    datos = datos.dropna(
        subset=[
            "home_score",
            "away_score"
        ]
    )

    # Filtrar partidos desde el año 2002
    datos = datos[
        datos["date"].dt.year >= 2002
    ].copy()

    # Ordenar cronológicamente
    datos = datos.sort_values(
        by="date"
    )

    # Reiniciar índices
    datos = datos.reset_index(
        drop=True
    )

    return datos


def preparar_penales(penales):
    """
    Prepara la base histórica de tandas de penales.
    """

    datos_penales = penales.copy()

    datos_penales["date"] = pd.to_datetime(
        datos_penales["date"]
    )

    datos_penales = datos_penales.sort_values(
        by="date"
    )

    datos_penales = datos_penales.reset_index(
        drop=True
    )

    return datos_penales