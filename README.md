# Modelo de Predicción de Clasificación - Copa Mundial 2026

## Descripción del proyecto

El presente proyecto desarrolla un modelo de Machine Learning orientado a la predicción de equipos clasificados en partidos de eliminación directa de la Copa Mundial 2026.

El estudio utiliza información histórica de partidos internacionales de fútbol y analiza el rendimiento reciente de cada selección antes de un enfrentamiento.

El objetivo principal es comparar diferentes algoritmos de clasificación y determinar cuál presenta el mejor rendimiento para predecir el equipo que avanza a la siguiente fase.

---

## Objetivo general

Desarrollar y comparar modelos de Machine Learning para predecir la clasificación de selecciones nacionales en partidos de eliminación directa, utilizando información histórica y características de rendimiento reciente.

---

## Fuente de datos

El proyecto utiliza una base histórica de resultados de partidos internacionales de fútbol.

Los principales archivos utilizados son:

- `results.csv`: contiene los resultados históricos de partidos internacionales.
- `shootouts.csv`: contiene información sobre los ganadores de tandas de penales.

La base original contiene información histórica desde años anteriores a 2002.

Sin embargo, para el desarrollo del modelo se utilizan los Mundiales disputados entre 2002 y 2022.

---

## Justificación del periodo de estudio

Se seleccionaron los Mundiales comprendidos entre los años 2002 y 2022:

- Mundial 2002
- Mundial 2006
- Mundial 2010
- Mundial 2014
- Mundial 2018
- Mundial 2022

La selección de este periodo busca trabajar con una etapa relativamente reciente del fútbol internacional.

Los Mundiales demasiado antiguos pueden representar contextos deportivos diferentes debido a cambios en las selecciones, estilos de juego, preparación física, estrategias y evolución del fútbol.

Por esta razón, se estableció el año 2002 como punto inicial del análisis.

---

## Partidos analizados

Para cada Mundial se analizaron los partidos correspondientes a las principales fases de eliminación directa:

- 8 partidos de octavos de final.
- 4 partidos de cuartos de final.
- 2 partidos de semifinal.
- 1 partido de la final.

Esto representa un total de 15 partidos por Mundial.

Se analizaron seis ediciones de la Copa Mundial:

15 partidos × 6 Mundiales = 90 partidos eliminatorios.

El partido por el tercer puesto fue excluido del análisis debido a que no determina la clasificación hacia una siguiente fase ni la obtención del campeonato.

---

## Construcción de características

Para cada partido se utilizaron únicamente los encuentros disputados antes de la fecha del enfrentamiento analizado.

Esta metodología evita utilizar información futura durante la construcción de las variables.

Se analizaron los últimos cinco partidos de cada selección.

Las variables utilizadas por los modelos fueron:

1. Promedio de goles anotados por el equipo local.
2. Promedio de goles anotados por el equipo visitante.
3. Promedio de goles recibidos por el equipo local.
4. Promedio de goles recibidos por el equipo visitante.
5. Victorias del equipo local en sus últimos cinco partidos.
6. Victorias del equipo visitante en sus últimos cinco partidos.
7. Puntos obtenidos por el equipo local en sus últimos cinco partidos.
8. Puntos obtenidos por el equipo visitante en sus últimos cinco partidos.
9. Diferencia de promedio de goles anotados.
10. Diferencia de promedio de goles recibidos.
11. Diferencia de victorias recientes.
12. Diferencia de puntos recientes.

Las variables de diferencia permiten representar la ventaja o desventaja reciente existente entre ambas selecciones.

---

## Variable objetivo

La variable objetivo utilizada fue:

`clasifica_local`

La codificación utilizada es:

- `1`: clasifica el equipo registrado como local.
- `0`: clasifica el equipo registrado como visitante.

En los partidos empatados se utilizó la información de las tandas de penales para identificar correctamente al equipo clasificado.

---

## División temporal de los datos

El proyecto utiliza una división temporal de los datos.

### Datos de entrenamiento

Se utilizaron los Mundiales de:

- 2002
- 2006
- 2010
- 2014
- 2018

Esto representa 75 partidos de entrenamiento.

### Datos de prueba

El Mundial 2022 fue utilizado como conjunto de prueba independiente.

Esto representa 15 partidos.

La división temporal permite entrenar los modelos utilizando información histórica y posteriormente evaluar su comportamiento sobre un Mundial más reciente.

De esta manera se busca representar de forma más cercana el objetivo real del proyecto: utilizar información del pasado para realizar predicciones sobre partidos futuros.

---

## Modelos de Machine Learning comparados

Se compararon seis algoritmos de clasificación:

1. Regresión Logística.
2. Árbol de Decisión.
3. Random Forest.
4. K-Nearest Neighbors (KNN).
5. Support Vector Machine (SVM).
6. XGBoost.

Los modelos fueron entrenados utilizando las mismas variables predictoras y evaluados sobre el Mundial 2022.

---

## Escalado de variables

Los modelos:

- Regresión Logística.
- KNN.
- SVM.

utilizan variables estandarizadas mediante `StandardScaler`.

El escalador fue ajustado únicamente con los datos de entrenamiento.

Posteriormente, la transformación aprendida fue aplicada al conjunto de prueba.

Este procedimiento evita incorporar información del conjunto de prueba durante el entrenamiento.

---

## Métricas de evaluación

Para comparar los modelos se utilizaron las siguientes métricas:

### Accuracy

Representa la proporción total de predicciones correctas realizadas por el modelo.

### Precision

Indica qué proporción de las predicciones positivas realizadas por el modelo fueron correctas.

### Recall

Mide la capacidad del modelo para identificar correctamente los casos positivos reales.

### F1-score

Representa el equilibrio entre Precision y Recall.

Esta métrica fue utilizada como criterio principal para seleccionar el mejor modelo.

### AUC

El área bajo la curva ROC permite analizar la capacidad del modelo para diferenciar entre las clases.

---

## Selección del mejor modelo

Los seis algoritmos fueron entrenados y comparados automáticamente.

El modelo seleccionado fue:

**Support Vector Machine (SVM)**

Resultados obtenidos:

- Accuracy: `0.7333`
- F1-score: `0.8333`
- AUC: `0.5556`

El modelo fue seleccionado automáticamente debido a que obtuvo el mayor F1-score dentro de la comparación realizada.

---

## Validación con el Mundial 2022

El modelo seleccionado fue evaluado utilizando los 15 partidos eliminatorios principales del Mundial 2022.

Resultados:

- Partidos evaluados: `15`
- Predicciones correctas: `11`
- Predicciones incorrectas: `4`
- Porcentaje de acierto: `73.33 %`

Estos resultados corresponden a un conjunto de prueba temporal independiente que no fue utilizado durante el entrenamiento de los modelos.

---

## Interpretación de los resultados

El modelo SVM presentó el mejor equilibrio entre Precision y Recall según el F1-score obtenido.

Durante la validación con el Mundial 2022, el modelo clasificó correctamente 11 de los 15 partidos analizados.

Sin embargo, el valor AUC obtenido fue de 0.5556.

Este resultado indica que la capacidad de separación probabilística entre las clases es limitada.

Por esta razón, las probabilidades generadas por el modelo no deben interpretarse como certezas absolutas sobre el resultado de un partido.

El modelo representa una aproximación estadística basada exclusivamente en las variables históricas utilizadas.

---

## Predicción de los octavos de final del Mundial 2026

Después de seleccionar el mejor algoritmo, el modelo fue utilizado para analizar los partidos pendientes de octavos de final del Mundial 2026.

Los enfrentamientos analizados fueron:

- Portugal vs España.
- Estados Unidos vs Bélgica.
- Brasil vs Noruega.
- México vs Inglaterra.
- Argentina vs Egipto.
- Suiza vs Colombia.

Para cada enfrentamiento se calcularon nuevamente las 12 características utilizando los partidos históricos anteriores a la fecha de referencia establecida para la predicción.

Posteriormente, las características fueron procesadas por el modelo SVM seleccionado.

---

## Estructura del proyecto

```text
PREDICCION 16.AVOS
│
├── Data
│   └── raw
│       ├── results.csv
│       └── shootouts.csv
│
├── Models
│   ├── modelo_clasificacion_mundial.pkl
│   └── escalador_variables.pkl
│
├── Notebooks
│
├── Results
│   ├── comparacion_modelos.csv
│   ├── validacion_mundial_2022.csv
│   └── predicciones_octavos_2026.csv
│
├── Src
│   ├── __init__.py
│   ├── data_processing.py
│   ├── evaluate.py
│   ├── features.py
│   ├── predict.py
│   └── train_models.py
│
├── .gitignore
├── main.py
├── README.md
└── requirements.txt

Descripción de los módulos
data_processing.py

Realiza la carga y preparación inicial de las bases de datos.

features.py

Construye las características temporales de cada selección utilizando sus últimos cinco partidos anteriores.

train_models.py

Crea, entrena y compara los seis algoritmos de Machine Learning.

evaluate.py

Realiza la evaluación del modelo seleccionado mediante predicciones, matriz de confusión, curva ROC, AUC y validación partido por partido.

predict.py

Construye las características de los partidos del Mundial 2026 y genera las predicciones del equipo clasificado.

main.py

Integra todos los módulos y ejecuta el flujo completo del proyecto.

Ejecución del proyecto

Primero se deben instalar las dependencias:

python -m pip install -r requirements.txt

Posteriormente, el proyecto puede ejecutarse mediante:

python main.py

El programa realiza automáticamente las siguientes etapas:

Carga de las bases históricas.
Identificación de los partidos eliminatorios.
Construcción de características.
División temporal de los datos.
Entrenamiento de los modelos.
Comparación de métricas.
Selección del mejor modelo.
Evaluación con el Mundial 2022.
Predicción de partidos del Mundial 2026.
Almacenamiento del modelo y los resultados.


Limitaciones

El modelo utiliza únicamente variables relacionadas con el rendimiento histórico reciente de las selecciones.

No se incluyen variables como:

Lesiones de jugadores.
Alineaciones titulares.
Cambios de entrenador.
Ranking FIFA.
Valor de mercado de los jugadores.
Posesión de balón.
Tiros al arco.
Expected Goals (xG).
Condiciones climáticas.
Fatiga acumulada.
Sanciones.

Por esta razón, los resultados representan una estimación basada exclusivamente en la información disponible dentro de la base histórica utilizada.



Conclusión

La comparación de seis algoritmos de Machine Learning permitió identificar a SVM como el modelo con mejor F1-score para el conjunto de datos analizado.

La utilización de una división temporal permitió evaluar el modelo sobre el Mundial 2022 sin utilizar dichos partidos durante el entrenamiento.

El modelo alcanzó un porcentaje de acierto de 73.33 %, clasificando correctamente 11 de los 15 partidos evaluados.

Finalmente, el modelo seleccionado fue utilizado para generar predicciones sobre los partidos pendientes de octavos de final del Mundial 2026.


El proyecto demuestra la aplicación de técnicas de procesamiento de datos, construcción de variables, clasificación, comparación de algoritmos y evaluación de modelos dentro de un problema relacionado con la predicción deportiva.