# ============================================================
# MODELO DE PREDICCIÓN DE CLASIFICACIÓN
# COPA MUNDIAL 2026
# ============================================================

import os
import joblib
import numpy as np
import pandas as pd

from Src.data_processing import (
    cargar_datos,
    preparar_resultados,
    preparar_penales
)

from Src.features import (
    construir_dataset_caracteristicas
)

from Src.train_models import (
    entrenar_y_comparar_modelos,
    seleccionar_mejor_modelo
)

from Src.evaluate import (
    obtener_predicciones,
    obtener_probabilidades,
    graficar_matriz_confusion,
    graficar_curva_roc,
    validar_partidos,
    resumir_validacion,
    graficar_comparacion_modelos
)

from Src.predict import (
    VARIABLES_MODELO,
    predecir_clasificados,
    traducir_resultados,
    analizar_diferencia_probabilidades
)


# ============================================================
# CONFIGURACIÓN DE RUTAS
# ============================================================

RUTA_RESULTADOS = os.path.join(
    "Data",
    "raw",
    "results.csv"
)

RUTA_PENALES = os.path.join(
    "Data",
    "raw",
    "shootouts.csv"
)

CARPETA_MODELOS = "Models"

CARPETA_RESULTADOS = "Results"


# ============================================================
# MUNDIALES UTILIZADOS
# ============================================================

ANIOS_MUNDIALES = [
    2002,
    2006,
    2010,
    2014,
    2018,
    2022
]


# ============================================================
# FUNCIÓN PARA IDENTIFICAR FASES
# ============================================================

def asignar_fases_eliminatorias(partidos):
    """
    Asigna la fase eliminatoria según el orden cronológico
    de los últimos 15 partidos de cada Mundial.
    """

    partidos = partidos.copy()

    partidos = partidos.sort_values(
        "date"
    ).reset_index(
        drop=True
    )

    fases = (
        ["Octavos de final"] * 8
        +
        ["Cuartos de final"] * 4
        +
        ["Semifinal"] * 2
        +
        ["Final"] * 1
    )

    partidos["fase"] = fases

    return partidos


# ============================================================
# OBTENER PARTIDOS ELIMINATORIOS
# ============================================================

def obtener_eliminatorias_mundiales(
    base_partidos
):
    """
    Obtiene los 15 partidos eliminatorios principales de cada
    Mundial entre 2002 y 2022.

    Se excluye el partido por el tercer puesto.
    """

    eliminatorias = []

    for anio in ANIOS_MUNDIALES:

        partidos_mundial = base_partidos[
            (
                base_partidos["tournament"]
                ==
                "FIFA World Cup"
            )
            &
            (
                base_partidos["date"].dt.year
                ==
                anio
            )
        ].copy()


        partidos_mundial = partidos_mundial.sort_values(
            "date"
        ).reset_index(
            drop=True
        )


        # ====================================================
        # ÚLTIMOS 16 PARTIDOS DEL MUNDIAL
        # ====================================================

        fase_final = partidos_mundial.tail(
            16
        ).copy()


        # ====================================================
        # IDENTIFICAR FINAL
        # ====================================================

        partido_final = fase_final.tail(
            1
        ).copy()


        # ====================================================
        # EXCLUIR PARTIDO POR TERCER PUESTO
        # ====================================================

        partidos_previos = fase_final.iloc[
            :-2
        ].copy()


        # ====================================================
        # CONSTRUIR 15 PARTIDOS ELIMINATORIOS
        # ====================================================

        partidos_mundial = pd.concat(
            [
                partidos_previos,
                partido_final
            ],
            ignore_index=True
        )


        if len(partidos_mundial) != 15:

            raise ValueError(
                f"El Mundial {anio} no contiene "
                f"15 partidos eliminatorios principales."
            )


        partidos_mundial = (
            asignar_fases_eliminatorias(
                partidos_mundial
            )
        )


        partidos_mundial["mundial"] = anio


        eliminatorias.append(
            partidos_mundial
        )


    dataset_eliminatorias = pd.concat(
        eliminatorias,
        ignore_index=True
    )


    return dataset_eliminatorias


# ============================================================
# IDENTIFICAR GANADOR DE PENALES
# ============================================================

def buscar_ganador_penales(
    partido,
    base_penales
):
    """
    Busca el ganador de una tanda de penales.
    """

    tanda = base_penales[
        (
            base_penales["date"]
            ==
            partido["date"]
        )
        &
        (
            (
                (
                    base_penales["home_team"]
                    ==
                    partido["home_team"]
                )
                &
                (
                    base_penales["away_team"]
                    ==
                    partido["away_team"]
                )
            )
            |
            (
                (
                    base_penales["home_team"]
                    ==
                    partido["away_team"]
                )
                &
                (
                    base_penales["away_team"]
                    ==
                    partido["home_team"]
                )
            )
        )
    ]

    if len(tanda) > 0:

        return tanda.iloc[0]["winner"]

    return np.nan


# ============================================================
# CREAR VARIABLE OBJETIVO
# ============================================================

def construir_variable_objetivo(
    partidos,
    base_penales
):
    """
    Construye la variable clasifica_local.

    1 = clasifica el equipo local.
    0 = clasifica el equipo visitante.
    """

    datos = partidos.copy()

    clasificados = []

    for _, partido in datos.iterrows():

        if (
            partido["home_score"]
            >
            partido["away_score"]
        ):

            clasificado = partido["home_team"]

        elif (
            partido["away_score"]
            >
            partido["home_score"]
        ):

            clasificado = partido["away_team"]

        else:

            clasificado = buscar_ganador_penales(
                partido,
                base_penales
            )

        clasificados.append(
            clasificado
        )

    datos["clasificado_real"] = (
        clasificados
    )

    if datos["clasificado_real"].isna().any():

        partidos_error = datos[
            datos["clasificado_real"].isna()
        ]

        print(
            "\nSe encontraron partidos sin clasificado:"
        )

        print(
            partidos_error[
                [
                    "date",
                    "home_team",
                    "away_team"
                ]
            ]
        )

        raise ValueError(
            "No fue posible identificar "
            "todos los clasificados."
        )

    datos["clasifica_local"] = np.where(
        datos["clasificado_real"]
        ==
        datos["home_team"],
        1,
        0
    )

    return datos


# ============================================================
# FUNCIÓN PRINCIPAL
# ============================================================

def main():

    print("\n")
    print("=" * 100)
    print("MODELO DE PREDICCIÓN DE CLASIFICACIÓN")
    print("COPA MUNDIAL 2026")
    print("=" * 100)


    # ========================================================
    # 1. CARGAR BASES DE DATOS
    # ========================================================

    print("\n[1] CARGANDO BASES DE DATOS...")

    resultados, penales = cargar_datos(
        RUTA_RESULTADOS,
        RUTA_PENALES
    )

    resultados = preparar_resultados(
        resultados
    )

    penales = preparar_penales(
        penales
    )

    print(
        f"Partidos históricos disponibles: "
        f"{len(resultados)}"
    )

    print(
        f"Tandas de penales disponibles: "
        f"{len(penales)}"
    )


    # ========================================================
    # 2. IDENTIFICAR ELIMINATORIAS
    # ========================================================

    print("\n[2] IDENTIFICANDO PARTIDOS ELIMINATORIOS...")

    eliminatorias = obtener_eliminatorias_mundiales(
        resultados
    )

    eliminatorias = construir_variable_objetivo(
        eliminatorias,
        penales
    )

    print(
        f"Partidos eliminatorios identificados: "
        f"{len(eliminatorias)}"
    )

    print("\nPartidos por Mundial:")

    print(
        eliminatorias[
            "mundial"
        ].value_counts().sort_index()
    )


    # ========================================================
    # 3. CONSTRUIR CARACTERÍSTICAS
    # ========================================================

    print("\n[3] CONSTRUYENDO CARACTERÍSTICAS...")

    dataset_ml = construir_dataset_caracteristicas(
        eliminatorias,
        resultados
    )

    dataset_ml["fase"] = (
        eliminatorias["fase"].values
    )

    dataset_ml["mundial"] = (
        eliminatorias["mundial"].values
    )

    dataset_ml["clasifica_local"] = (
        eliminatorias["clasifica_local"].values
    )

    print(
        f"Dimensión del dataset: "
        f"{dataset_ml.shape}"
    )


    # ========================================================
    # 4. DIVISIÓN TEMPORAL
    # ========================================================

    print("\n[4] REALIZANDO DIVISIÓN TEMPORAL...")

    datos_train = dataset_ml[
        dataset_ml["mundial"] < 2022
    ].copy()

    datos_test = dataset_ml[
        dataset_ml["mundial"] == 2022
    ].copy()

    print(
        f"Partidos de entrenamiento: "
        f"{len(datos_train)}"
    )

    print(
        f"Partidos de prueba: "
        f"{len(datos_test)}"
    )

    print(
        "\nEntrenamiento: Mundiales 2002 - 2018"
    )

    print(
        "Prueba independiente: Mundial 2022"
    )


    # ========================================================
    # 5. VARIABLES PREDICTORAS Y OBJETIVO
    # ========================================================

    X_train = datos_train[
        VARIABLES_MODELO
    ].copy()

    X_test = datos_test[
        VARIABLES_MODELO
    ].copy()

    y_train = datos_train[
        "clasifica_local"
    ].copy()

    y_test = datos_test[
        "clasifica_local"
    ].copy()


    # ========================================================
    # 6. ENTRENAR Y COMPARAR MODELOS
    # ========================================================

    print("\n[5] ENTRENANDO MODELOS...")

    (
        comparacion_modelos,
        modelos_entrenados,
        scaler

    ) = entrenar_y_comparar_modelos(
        X_train,
        X_test,
        y_train,
        y_test
    )


    print("\n")
    print("=" * 100)
    print("COMPARACIÓN DE MODELOS")
    print("=" * 100)

    print(
        comparacion_modelos.to_string(
            index=False
        )
    )


    # ============================================================
    # GRÁFICA COMPARATIVA DE LOS MODELOS
    # ============================================================

    graficar_comparacion_modelos(
        comparacion_modelos
    )


    # ============================================================
    # 7. SELECCIONAR MEJOR MODELO
    # ============================================================

    (
        mejor_nombre,
        mejor_modelo,
        mejor_accuracy,
        mejor_f1

    ) = seleccionar_mejor_modelo(
        comparacion_modelos,
        modelos_entrenados
    )


    print("\n")
    print("=" * 100)
    print("MODELO SELECCIONADO")
    print("=" * 100)

    print(
        f"\nModelo: {mejor_nombre}"
    )

    print(
        f"Accuracy: {mejor_accuracy:.4f}"
    )

    print(
        f"F1-score: {mejor_f1:.4f}"
    )


    # ========================================================
    # 8. EVALUACIÓN DEL MODELO
    # ========================================================

    print("\n[6] EVALUANDO MODELO SELECCIONADO...")

    predicciones_test = obtener_predicciones(
        mejor_nombre,
        mejor_modelo,
        X_test,
        scaler
    )

    probabilidades_test = obtener_probabilidades(
        mejor_nombre,
        mejor_modelo,
        X_test,
        scaler
    )


    # ========================================================
    # MATRIZ DE CONFUSIÓN
    # ========================================================

    matriz = graficar_matriz_confusion(
        y_test,
        predicciones_test,
        mejor_nombre
    )


    # ========================================================
    # CURVA ROC
    # ========================================================

    auc = graficar_curva_roc(
        y_test,
        probabilidades_test,
        mejor_nombre
    )

    print(
        f"\nAUC del modelo seleccionado: "
        f"{auc:.4f}"
    )


    # ========================================================
    # 9. VALIDACIÓN PARTIDO POR PARTIDO
    # ========================================================

    validacion_2022 = validar_partidos(
        datos_test,
        predicciones_test,
        probabilidades_test
    )

    resumen_validacion = resumir_validacion(
        validacion_2022
    )


    print("\n")
    print("=" * 100)
    print("VALIDACIÓN MUNDIAL 2022")
    print("=" * 100)

    print(
        validacion_2022[
            [
                "fase",
                "home_team",
                "away_team",
                "clasificado_real",
                "clasificado_predicho",
                "resultado_prediccion"
            ]
        ].to_string(
            index=False
        )
    )


    print(
        f"\nPartidos evaluados: "
        f"{resumen_validacion['total_partidos']}"
    )

    print(
        f"Predicciones correctas: "
        f"{resumen_validacion['aciertos']}"
    )

    print(
        f"Predicciones incorrectas: "
        f"{resumen_validacion['errores']}"
    )

    print(
        f"Porcentaje de acierto: "
        f"{resumen_validacion['porcentaje_acierto']:.2f}%"
    )


    # ========================================================
    # 10. CRUCES DEL MUNDIAL 2026
    # ========================================================

    cruces_2026 = [

    ("Portugal", "Spain"),

    ("United States", "Belgium"),

    ("Brazil", "Norway"),

    ("Mexico", "England"),

    ("Argentina", "Egypt"),

    ("Switzerland", "Colombia"),

    ("Canada", "Morocco"),

    ("France", "Paraguay")

]

    traduccion_equipos = {

    "Portugal": "Portugal",

    "Spain": "España",

    "United States": "Estados Unidos",

    "Belgium": "Bélgica",

    "Brazil": "Brasil",

    "Norway": "Noruega",

    "Mexico": "México",

    "England": "Inglaterra",

    "Argentina": "Argentina",

    "Egypt": "Egipto",

    "Switzerland": "Suiza",

    "Colombia": "Colombia",

    "Canada": "Canadá",

    "Morocco": "Marruecos",

    "France": "Francia",

    "Paraguay": "Paraguay"

}


    # ========================================================
    # 11. PREDICCIÓN MUNDIAL 2026
    # ========================================================

    print("\n[7] GENERANDO PREDICCIONES MUNDIAL 2026...")

    resultados_2026 = predecir_clasificados(
        cruces=cruces_2026,
        fecha_referencia="2026-07-05",
        base_partidos=resultados,
        nombre_modelo=mejor_nombre,
        modelo=mejor_modelo,
        scaler=scaler
    )

    tabla_final = traducir_resultados(
        resultados_2026,
        traduccion_equipos
    ) 
    # ========================================================
    # ESTIMACIÓN DEL MARCADOR
    # ========================================================

    def estimar_marcador(prob1, prob2):

        diferencia = abs(prob1 - prob2)

        if diferencia >= 0.45:
            return "3-0"
        elif diferencia >= 0.30:
            return "2-0"
        elif diferencia >= 0.15:
            return "2-1"
        else:
            return "1-0"

    tabla_final["Marcador estimado"] = tabla_final.apply(
        lambda fila: estimar_marcador(
            fila["Probabilidad Equipo 1"],
            fila["Probabilidad Equipo 2"]
        ),
        axis=1
    )

    print("\n")
    print("=" * 100)
    print("PREDICCIONES OCTAVOS DE FINAL - MUNDIAL 2026")
    print("=" * 100)


    tabla_impresion = tabla_final.copy()

    tabla_impresion[
        "Probabilidad Equipo 1"
    ] = tabla_impresion[
        "Probabilidad Equipo 1"
    ].map(
        lambda valor: f"{valor:.2%}"
    )

    tabla_impresion[
        "Probabilidad Equipo 2"
    ] = tabla_impresion[
        "Probabilidad Equipo 2"
    ].map(
        lambda valor: f"{valor:.2%}"
    )


    print(
        tabla_impresion.to_string(
            index=False
        )
    )


    # ========================================================
    # 12. ANÁLISIS DE DIFERENCIAS
    # ========================================================

    analisis_diferencias = (
        analizar_diferencia_probabilidades(
            tabla_final
        )
    )


    print("\n")
    print("=" * 100)
    print("ANÁLISIS DE DIFERENCIAS PROBABILÍSTICAS")
    print("=" * 100)


    analisis_impresion = (
        analisis_diferencias.copy()
    )

    analisis_impresion[
        "Diferencia probabilística"
    ] = analisis_impresion[
        "Diferencia probabilística"
    ].map(
        lambda valor: f"{valor:.2%}"
    )


    print(
        analisis_impresion[
            [
                "Equipo 1",
                "Equipo 2",
                "Clasificado predicho",
                "Diferencia probabilística",
                "Nivel de diferencia"
            ]
        ].to_string(
            index=False
        )
    )


    # ========================================================
    # 13. GUARDAR MODELO Y RESULTADOS
    # ========================================================

    print("\n[8] GUARDANDO RESULTADOS...")

    os.makedirs(
        CARPETA_MODELOS,
        exist_ok=True
    )

    os.makedirs(
        CARPETA_RESULTADOS,
        exist_ok=True
    )


    joblib.dump(
        mejor_modelo,
        os.path.join(
            CARPETA_MODELOS,
            "modelo_clasificacion_mundial.pkl"
        )
    )


    joblib.dump(
        scaler,
        os.path.join(
            CARPETA_MODELOS,
            "escalador_variables.pkl"
        )
    )


    comparacion_modelos.to_csv(
        os.path.join(
            CARPETA_RESULTADOS,
            "comparacion_modelos.csv"
        ),
        index=False,
        encoding="utf-8-sig"
    )


    validacion_2022.to_csv(
        os.path.join(
            CARPETA_RESULTADOS,
            "validacion_mundial_2022.csv"
        ),
        index=False,
        encoding="utf-8-sig"
    )


    tabla_final.to_csv(
        os.path.join(
            CARPETA_RESULTADOS,
            "predicciones_octavos_2026.csv"
        ),
        index=False,
        encoding="utf-8-sig"
    )


    # ========================================================
    # 14. RESUMEN FINAL
    # ========================================================

    print("\n")
    print("=" * 100)
    print("RESUMEN FINAL DEL PROYECTO")
    print("=" * 100)


    resumen_tecnico = f"""
Modelo seleccionado:
{mejor_nombre}

Accuracy:
{mejor_accuracy:.4f}

F1-score:
{mejor_f1:.4f}

AUC:
{auc:.4f}

Validación Mundial 2022:
{resumen_validacion['aciertos']} aciertos de
{resumen_validacion['total_partidos']} partidos.

Porcentaje de acierto:
{resumen_validacion['porcentaje_acierto']:.2f}%

El modelo y los resultados fueron guardados correctamente.
""" 
    print(
        resumen_tecnico
    )

    # ============================================================
    # GUARDAR RESUMEN TÉCNICO DEL MODELO
    # ============================================================

    ruta_resumen = "Results/resumen_tecnico_modelo.txt"

    with open(
        ruta_resumen,
        "w",
        encoding="utf-8"
        
    ) as archivo_resumen:
        archivo_resumen.write(
            resumen_tecnico.strip()
        )

    print(
        "\nResumen técnico guardado en:",
        ruta_resumen
    )
# ============================================================
# EJECUCIÓN DEL PROGRAMA
# ============================================================

if __name__ == "__main__":

    main()