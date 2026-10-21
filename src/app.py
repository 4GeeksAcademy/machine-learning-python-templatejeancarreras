"""Agrupacion de California Housing y clasificacion de sus pseudoetiquetas."""

import argparse
import json
import platform
from pathlib import Path
from importlib.metadata import version

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
from sklearn.cluster import KMeans
from sklearn.datasets import fetch_california_housing
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    silhouette_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ["Latitude", "Longitude", "MedInc"]
SEED = 42
N_CLUSTERS = 6


def load_data(csv_path=None):
    """Carga un CSV local o descarga la fuente oficial mediante scikit-learn."""
    if csv_path is not None:
        data = pd.read_csv(csv_path)
    else:
        candidates = [ROOT / "housing.csv", ROOT / "data/raw/housing.csv"]
        existing = next((path for path in candidates if path.exists()), None)
        if existing is not None:
            data = pd.read_csv(existing)
        else:
            data = fetch_california_housing(as_frame=True).data.loc[:, FEATURES]
            destination = ROOT / "data/raw/housing.csv"
            destination.parent.mkdir(parents=True, exist_ok=True)
            data.to_csv(destination, index=False)
    missing = set(FEATURES) - set(data.columns)
    if missing:
        raise ValueError(f"Faltan columnas obligatorias: {sorted(missing)}")
    features = data.loc[:, FEATURES].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(features.to_numpy()).all():
        raise ValueError("Las variables seleccionadas contienen nulos o infinitos.")
    if len(features) < 30 or len(features.drop_duplicates()) < N_CLUSTERS:
        raise ValueError("Se necesitan al menos 30 filas y seis puntos distintos.")
    features.index.name = "row_id"
    return features


def fit_models(train):
    """Ajusta ambos modelos sin recibir datos de prueba."""
    clustering = Pipeline([
        ("scaler", StandardScaler()),
        ("kmeans", KMeans(n_clusters=N_CLUSTERS, n_init=20, random_state=SEED)),
    ])
    labels = clustering.fit_predict(train)
    counts = np.bincount(labels, minlength=N_CLUSTERS)
    if counts.min() < 3:
        raise ValueError("Cada cluster de train necesita al menos tres filas para CV.")
    search = GridSearchCV(
        DecisionTreeClassifier(random_state=SEED),
        {"max_depth": [6, 10, None], "min_samples_leaf": [1, 5, 10]},
        scoring="f1_macro",
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED),
        n_jobs=1,
    )
    search.fit(train, labels)
    return clustering, search


def evaluate(clustering, classifier, train, test):
    """Mide separacion y fidelidad del clasificador a las etiquetas de K-Means."""
    train_labels = clustering.predict(train)
    test_labels = clustering.predict(test)
    predicted = classifier.predict(test)
    baseline = DummyClassifier(strategy="most_frequent").fit(train, train_labels)
    metrics = {
        "seed": SEED,
        "features": FEATURES,
        "n_clusters": N_CLUSTERS,
        "train_rows": len(train),
        "test_rows": len(test),
        "best_tree_parameters": classifier.get_params(),
        "train_accuracy": accuracy_score(train_labels, classifier.predict(train)),
        "test_accuracy": accuracy_score(test_labels, predicted),
        "test_f1_macro": f1_score(test_labels, predicted, average="macro"),
        "baseline_test_accuracy": accuracy_score(test_labels, baseline.predict(test)),
        "classification_report": classification_report(
            test_labels, predicted, labels=list(range(N_CLUSTERS)),
            output_dict=True, zero_division=0,
        ),
        "train_inertia": float(clustering.named_steps["kmeans"].inertia_),
    }
    for name, frame, labels in [
        ("train", train, train_labels), ("test", test, test_labels),
    ]:
        scaled = clustering.named_steps["scaler"].transform(frame)
        sample_size = min(2000, len(frame))
        sampled = np.random.RandomState(SEED).choice(len(frame), sample_size, replace=False)
        sample_labels = labels[sampled]
        metrics[f"{name}_silhouette_sample_rows"] = sample_size
        metrics[f"{name}_silhouette"] = (
            float(silhouette_score(scaled[sampled], sample_labels))
            if 1 < len(np.unique(sample_labels)) < sample_size else None
        )
    return metrics


def save_results(data, train, test, clustering, search, output):
    """Guarda modelos completos, datos etiquetados, graficos e informe."""
    models = output / "models"
    processed = output / "data/processed"
    reports = output / "reports"
    for directory in [models, processed, reports]:
        directory.mkdir(parents=True, exist_ok=True)
    environment = {"python": platform.python_version()}
    environment.update({
        package: version(package)
        for package in ["scikit-learn", "numpy", "pandas", "matplotlib", "joblib"]
    })
    (reports / "environment.json").write_text(
        json.dumps(environment, indent=2) + "\n", encoding="utf-8"
    )
    classifier = search.best_estimator_
    metrics = evaluate(clustering, classifier, train, test)
    metrics["cv_best_f1_macro"] = float(search.best_score_)
    metrics["cv_best_parameters"] = search.best_params_
    labeled = {}
    for name, frame in [("train", train), ("test", test)]:
        labeled[name] = frame.assign(cluster=clustering.predict(frame))
        labeled[name]["cluster"] = pd.Categorical(
            labeled[name]["cluster"], categories=range(N_CLUSTERS)
        )
        labeled[name].to_csv(processed / f"housing_{name}_clusters.csv")
    labeled["test"].assign(predicted_cluster=classifier.predict(test)).to_csv(
        processed / "housing_test_predictions.csv"
    )
    data.describe().to_csv(reports / "data_summary.csv")
    data.isna().sum().to_csv(reports / "missing_values.csv", header=["missing"])
    data.corr().to_csv(reports / "correlations.csv")
    profile = labeled["train"].groupby("cluster", observed=False).agg(
        count=("MedInc", "size"),
        latitude_mean=("Latitude", "mean"),
        longitude_mean=("Longitude", "mean"),
        income_mean=("MedInc", "mean"),
        income_median=("MedInc", "median"),
    )
    profile.to_csv(reports / "cluster_profiles.csv")
    centers = pd.DataFrame(
        clustering.named_steps["scaler"].inverse_transform(
            clustering.named_steps["kmeans"].cluster_centers_
        ), columns=FEATURES,
    ).rename_axis("cluster")
    centers.to_csv(reports / "cluster_centers.csv")
    pd.DataFrame(search.cv_results_).to_csv(reports / "tree_cross_validation.csv", index=False)
    joblib.dump(clustering, models / "kmeans_pipeline.joblib")
    joblib.dump(classifier, models / "decision_tree.joblib")
    (reports / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    colors = ListedColormap(["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00", "#56B4E9"])
    norm = BoundaryNorm(np.arange(-0.5, 6.5), colors.N)
    figure, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)
    for axis in axes:
        scatter = axis.scatter(
            train["Longitude"], train["Latitude"], c=labeled["train"]["cluster"].astype(int),
            cmap=colors, norm=norm, s=5, alpha=0.3, label="Train",
        )
        axis.scatter(centers["Longitude"], centers["Latitude"], marker="X", s=100,
                     color="black", edgecolors="white", label="Centroides")
        axis.set(xlabel="Longitud", ylabel="Latitud", aspect="equal")
    axes[0].set_title("K-Means: seis clusters de entrenamiento")
    axes[1].scatter(
        test["Longitude"], test["Latitude"], c=labeled["test"]["cluster"].astype(int),
        cmap=colors, norm=norm, s=18, marker="^", alpha=0.65, label="Test",
    )
    axes[1].set_title("Train y predicciones de test")
    axes[1].legend()
    figure.colorbar(scatter, ax=axes, ticks=range(N_CLUSTERS), label="Cluster")
    figure.savefig(reports / "clusters.png", dpi=150, bbox_inches="tight")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(9, 5))
    labeled["train"].boxplot(column="MedInc", by="cluster", ax=axis)
    axis.set(xlabel="Cluster", ylabel="Ingreso mediano (decenas de miles de USD de 1990)")
    axis.set_title("Distribucion del ingreso por cluster (train)")
    figure.suptitle("")
    figure.tight_layout()
    figure.savefig(reports / "income_by_cluster.png", dpi=150)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay.from_predictions(
        labeled["test"]["cluster"].astype(int), classifier.predict(test),
        labels=list(range(N_CLUSTERS)), ax=axis, cmap="Blues", colorbar=False,
    )
    axis.set(xlabel="Cluster del arbol", ylabel="Cluster de K-Means", title="Test: matriz de confusion")
    figure.tight_layout()
    figure.savefig(reports / "confusion_matrix.png", dpi=150)
    plt.close(figure)

    high_income = int(profile["income_mean"].idxmax())
    north = int(profile["latitude_mean"].idxmax())
    south = int(profile["latitude_mean"].idxmin())
    report = f"""# Resultados: agrupacion de California Housing

## Datos y protocolo

Se analizaron {len(data):,} grupos de bloques censales, no viviendas individuales.
Variables: Latitude, Longitude y MedInc. No se utiliza el precio como predictor.
MedInc expresa el ingreso mediano en decenas de miles de dolares de 1990.
Valores nulos en estas variables: {int(data.isna().sum().sum())}.
Filas duplicadas en estas tres variables: {int(data.duplicated().sum())}; se conservan,
pues no demuestran que sean el mismo bloque censal.
Particion aleatoria 80/20, semilla {SEED}: {len(train):,} train y {len(test):,} test.
StandardScaler y K-Means se ajustaron exclusivamente con train.
Se fijo k=6 por el enunciado; no se afirma que sea el numero optimo.

## Interpretacion de los clusters

El mapa muestra separacion geografica, pero no toda la informacion: el ingreso
es la tercera dimension y por eso pueden solaparse colores en latitud/longitud.
Cada variable tiene el mismo peso tras estandarizar; los grados no son distancias
geodesicas. Los centroides del mapa son proyecciones de centros tridimensionales.
El cluster {north} tiene la latitud media mas alta y el {south} la mas baja.
El cluster {high_income} tiene el ingreso medio mas alto:
{profile.loc[high_income, 'income_mean']:.2f} decenas de miles de USD.
Los perfiles completos y tamanos estan en cluster_profiles.csv; las etiquetas
0 a 5 son categorias nominales, no un ranking de calidad ni de ingreso.

Silhouette train (muestra de hasta 2000): {metrics['train_silhouette']}.
Silhouette test (muestra de hasta 2000): {metrics['test_silhouette']}.
Esta medida va de -1 a 1: mayor implica grupos mas compactos y separados.
La superposicion de test permite inspeccionar continuidad espacial, pero no
demuestra que los grupos representen regiones administrativas reales.

## Clasificacion supervisada

Se eligio un arbol de decision por su interpretabilidad y por poder separar
regiones no lineales combinando las tres variables. Sus cortes alineados con los
ejes solo aproximan las fronteras oblicuas de los grupos de K-Means.
Se compararon nueve configuraciones mediante CV estratificada de tres folds
solo en train, con F1 macro. Las pseudoetiquetas se fijaron antes de esta CV;
la puntuacion CV evalua la imitacion del agrupador, no todo el proceso de clustering.
Mejores parametros: {search.best_params_}.
F1 macro CV: {search.best_score_:.4f}.
Accuracy train: {metrics['train_accuracy']:.4f}.
Accuracy test: {metrics['test_accuracy']:.4f}.
F1 macro test: {metrics['test_f1_macro']:.4f}.
Accuracy del baseline mayoritario en test: {metrics['baseline_test_accuracy']:.4f}.
Brecha accuracy train-test: {metrics['train_accuracy'] - metrics['test_accuracy']:.4f}.
Precision, recall, F1 y soporte por clase estan en metrics.json.

Estas cifras miden concordancia con K-Means, no exactitud frente a etiquetas
reales. Un resultado alto indica que el arbol reproduce bien esa particion.
No se puede concluir capacidad de predecir precios o clases socioeconomicas.
La division aleatoria puede compartir vecindarios entre train y test; para
generalizar a regiones nuevas se necesitaria una evaluacion espacial por bloques.
K-Means ya permite predecir directamente; el arbol aporta una aproximacion
interpretable, no nuevas etiquetas independientes.

## Modelos

models/kmeans_pipeline.joblib incluye el escalador y el agrupador.
models/decision_tree.joblib recibe Latitude, Longitude y MedInc sin escalar.
Ambos se guardan sin reajustar con test. Cargar solo archivos joblib de confianza.
"""
    (reports / "analysis.md").write_text(report, encoding="utf-8")
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, help="CSV con Latitude, Longitude y MedInc")
    parser.add_argument("--output", type=Path, default=ROOT, help="Directorio de resultados")
    arguments = parser.parse_args()
    data = load_data(arguments.data)
    train, test = train_test_split(data, test_size=0.2, random_state=SEED)
    clustering, search = fit_models(train)
    metrics = save_results(data, train, test, clustering, search, arguments.output)
    print(f"Train: {len(train)} | Test: {len(test)} | Clusters: {N_CLUSTERS}")
    print(f"Accuracy test: {metrics['test_accuracy']:.4f}")
    print(f"F1 macro test: {metrics['test_f1_macro']:.4f}")
    print(f"Informe: {arguments.output / 'reports/analysis.md'}")


if __name__ == "__main__":
    main()
