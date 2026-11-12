"""Sistema de agrupación de casas (California Housing).

Descripción de resultados y advertencias: ver house_clustering.md en esta carpeta.
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "housing.csv"
DATA_URL = "https://breathecode.herokuapp.com/asset/internal-link?id=809&path=housing.csv"
MODELS_DIR = ROOT / "models"
FEATURES = ["Latitude", "Longitude", "MedInc"]
N_CLUSTERS = 6
RANDOM_STATE = 42


def load_data():
    source = RAW_PATH if RAW_PATH.exists() else DATA_URL
    df = pd.read_csv(source)
    return df[FEATURES].copy()


def plot_clusters(df, title, overlay=None):
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(df["Longitude"], df["Latitude"], c=df["cluster"].cat.codes,
                         cmap="tab10", vmin=0, vmax=N_CLUSTERS - 1, s=5, alpha=0.6)
    if overlay is not None:
        ax.scatter(overlay["Longitude"], overlay["Latitude"], c=overlay["cluster"].cat.codes,
                   cmap="tab10", vmin=0, vmax=N_CLUSTERS - 1, s=8, marker="x", alpha=0.8)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title)
    fig.colorbar(scatter, ax=ax, label="cluster")
    plt.show()


def main():
    # Paso 1: carga y división del dataset
    df = load_data()
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=RANDOM_STATE)

    # Paso 2: K-Means entrenado con train
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=10)
    train_df = train_df.copy()
    train_df["cluster"] = pd.Categorical(kmeans.fit_predict(train_df[FEATURES]),
                                         categories=range(N_CLUSTERS))
    plot_clusters(train_df, "K-Means (train) - 6 clusters")

    # Paso 3: predicción sobre test, añadida al gráfico de train
    test_df = test_df.copy()
    test_df["cluster"] = pd.Categorical(kmeans.predict(test_df[FEATURES]),
                                        categories=range(N_CLUSTERS))
    plot_clusters(train_df, "K-Means: train (o) y predicciones sobre test (x)", overlay=test_df)

    # Paso 4: clasificación supervisada usando los clusters como etiqueta
    X_train, y_train = train_df[FEATURES], train_df["cluster"].cat.codes
    X_test, y_test = test_df[FEATURES], test_df["cluster"].cat.codes

    classifier = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    print(f"Accuracy en test: {accuracy_score(y_test, y_pred):.4f}")
    print("Matriz de confusión:")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    # Paso 5: guardar ambos modelos
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(kmeans, MODELS_DIR / "kmeans_model.pkl")
    joblib.dump(classifier, MODELS_DIR / "classifier_model.pkl")
    print(f"Modelos guardados en {MODELS_DIR}")


if __name__ == "__main__":
    main()
