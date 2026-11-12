# Sistema de agrupación de casas (California Housing)

Pipeline implementado en [house_clustering.py](./house_clustering.py).

## Datos

- Dataset: California Housing (censo de California de 1990), 20.640 filas.
- Variables usadas: `Latitude`, `Longitude` y `MedInc`.
- División: 80% train / 20% test (`random_state=42`).

## Resultados

### Paso 2: K-Means (6 clusters)

Los clusters separan regiones geográficas:

- El norte y el interior forman un grupo.
- El centro (Sierra Nevada y valle central) forma otro.
- El sur (Los Ángeles) tiene grupos propios.
- La Bahía de San Francisco forma un cluster compacto.

### Paso 3: Predicción sobre test

Los puntos de test se asignan a las mismas regiones que los de train. Sobre el gráfico de train, los puntos de test (marcados con `x`) quedan dentro de sus regiones, lo que confirma que la predicción es coherente.

### Paso 4: Clasificador supervisado

Modelo: `RandomForestClassifier(n_estimators=200, random_state=42)` entrenado para reproducir los clusters.

- Accuracy en test: 0,995.
- F1 macro en test: 0,99.
- La clase con menos soporte (cluster 1, 89 muestras) tiene el recall más bajo (0,97).
- Los errores se producen entre clusters vecinos.

**Advertencia:** estas métricas son altas por construcción. El clasificador aprende a reproducir clusters generados con las mismas tres variables (latitud, longitud e ingreso), así que no mide la capacidad de generalizar a otra variable.

## Modelos

Los modelos se guardan en `models/` con `joblib`:

- `kmeans_model.pkl`: K-Means entrenado.
- `classifier_model.pkl`: RandomForest entrenado.

Estos ficheros no están versionados en este repositorio por la política de contenido de la organización. Para generarlos de nuevo, ejecuta el pipeline con el CSV en `data/raw/housing.csv` o con la URL de la descripción del proyecto.
