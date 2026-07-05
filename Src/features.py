# ============================================================
# CONSTRUCCIÓN DE CARACTERÍSTICAS TEMPORALES
# ============================================================

import pandas as pd


# ============================================================
# OBTENER ÚLTIMOS PARTIDOS DE UNA SELECCIÓN
# ============================================================

def obtener_ultimos_partidos(
    equipo,
    fecha_referencia,
    base_partidos,
    cantidad=5
):
    """
    Obtiene los últimos partidos disputados por una selección
    antes de una fecha determinada.
    """

    fecha_referencia = pd.to_datetime(
        fecha_referencia
    )

    partidos_equipo = base_partidos[
        (
            (
                base_partidos["home_team"] == equipo
            )
            |
            (
                base_partidos["away_team"] == equipo
            )
        )
        &
        (
            base_partidos["date"] < fecha_referencia
        )
    ].copy()

    partidos_equipo = partidos_equipo.sort_values(
        by="date"
    )

    ultimos_partidos = partidos_equipo.tail(
        cantidad
    )

    return ultimos_partidos


# ============================================================
# CALCULAR ESTADÍSTICAS DE UNA SELECCIÓN
# ============================================================

def calcular_estadisticas_equipo(
    equipo,
    fecha_referencia,
    base_partidos
):
    """
    Calcula el rendimiento de una selección utilizando
    exclusivamente sus últimos cinco partidos anteriores.
    """

    partidos = obtener_ultimos_partidos(
        equipo,
        fecha_referencia,
        base_partidos,
        cantidad=5
    )

    goles_anotados = []

    goles_recibidos = []

    victorias = 0

    puntos = 0


    for _, partido in partidos.iterrows():

        # ----------------------------------------------------
        # EQUIPO REGISTRADO COMO LOCAL
        # ----------------------------------------------------

        if partido["home_team"] == equipo:

            goles_favor = partido["home_score"]

            goles_contra = partido["away_score"]


        # ----------------------------------------------------
        # EQUIPO REGISTRADO COMO VISITANTE
        # ----------------------------------------------------

        else:

            goles_favor = partido["away_score"]

            goles_contra = partido["home_score"]


        goles_anotados.append(
            goles_favor
        )

        goles_recibidos.append(
            goles_contra
        )


        # ----------------------------------------------------
        # RESULTADO DEL PARTIDO
        # ----------------------------------------------------

        if goles_favor > goles_contra:

            victorias += 1

            puntos += 3

        elif goles_favor == goles_contra:

            puntos += 1


    # ========================================================
    # CONTROL DE EQUIPOS SIN HISTORIAL
    # ========================================================

    if len(partidos) == 0:

        return {
            "prom_goles": 0.0,
            "prom_recibidos": 0.0,
            "victorias_ult5": 0,
            "puntos_ult5": 0
        }


    # ========================================================
    # ESTADÍSTICAS FINALES
    # ========================================================

    estadisticas = {

        "prom_goles": sum(
            goles_anotados
        ) / len(partidos),

        "prom_recibidos": sum(
            goles_recibidos
        ) / len(partidos),

        "victorias_ult5": victorias,

        "puntos_ult5": puntos

    }

    return estadisticas


# ============================================================
# CONSTRUIR CARACTERÍSTICAS DE UN PARTIDO
# ============================================================

def construir_caracteristicas_partido(
    equipo_local,
    equipo_visitante,
    fecha_partido,
    base_partidos
):
    """
    Construye las variables predictoras de un enfrentamiento.
    """

    estadisticas_local = calcular_estadisticas_equipo(
        equipo_local,
        fecha_partido,
        base_partidos
    )

    estadisticas_visitante = calcular_estadisticas_equipo(
        equipo_visitante,
        fecha_partido,
        base_partidos
    )


    caracteristicas = {

        "prom_goles_local":
        estadisticas_local["prom_goles"],

        "prom_goles_visitante":
        estadisticas_visitante["prom_goles"],

        "prom_recibidos_local":
        estadisticas_local["prom_recibidos"],

        "prom_recibidos_visitante":
        estadisticas_visitante["prom_recibidos"],

        "victorias_ult5_local":
        estadisticas_local["victorias_ult5"],

        "victorias_ult5_visitante":
        estadisticas_visitante["victorias_ult5"],

        "puntos_ult5_local":
        estadisticas_local["puntos_ult5"],

        "puntos_ult5_visitante":
        estadisticas_visitante["puntos_ult5"]

    }


    # ========================================================
    # VARIABLES DE DIFERENCIA
    # ========================================================

    caracteristicas["dif_goles"] = (
        caracteristicas["prom_goles_local"]
        -
        caracteristicas["prom_goles_visitante"]
    )

    caracteristicas["dif_recibidos"] = (
        caracteristicas["prom_recibidos_local"]
        -
        caracteristicas["prom_recibidos_visitante"]
    )

    caracteristicas["dif_victorias"] = (
        caracteristicas["victorias_ult5_local"]
        -
        caracteristicas["victorias_ult5_visitante"]
    )

    caracteristicas["dif_puntos"] = (
        caracteristicas["puntos_ult5_local"]
        -
        caracteristicas["puntos_ult5_visitante"]
    )


    return caracteristicas


# ============================================================
# CONSTRUIR DATASET DE MACHINE LEARNING
# ============================================================

def construir_dataset_caracteristicas(
    partidos_objetivo,
    base_partidos
):
    """
    Construye las características temporales de todos los
    partidos utilizados por los modelos de Machine Learning.
    """

    registros = []


    for _, partido in partidos_objetivo.iterrows():

        caracteristicas = construir_caracteristicas_partido(
            partido["home_team"],
            partido["away_team"],
            partido["date"],
            base_partidos
        )


        registro = {

            "fecha": partido["date"],

            "home_team": partido["home_team"],

            "away_team": partido["away_team"]

        }


        registro.update(
            caracteristicas
        )


        registros.append(
            registro
        )


    dataset = pd.DataFrame(
        registros
    )


    return dataset