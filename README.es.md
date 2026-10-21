# Proyecto Tutorial K-Medias

Agrupación de California Housing por ubicación e ingreso, seguida de un clasificador
supervisado que aprende las etiquetas generadas por K-Means.

## Ejecución

Desde la raíz del repositorio, con Python 3.11 o posterior:

```bash
pip install -r requirements.txt
python src/app.py
```

No se necesita una base de datos ni configurar variables de entorno para este
ejercicio. El programa busca `housing.csv` en la raíz y en `data/raw/`. Si no lo
encuentra, utiliza `fetch_california_housing` de scikit-learn y guarda las tres
variables en `data/raw/housing.csv`. Esa primera descarga necesita conexión a
Internet. También puedes indicar tu propio CSV:

```bash
python src/app.py --data ruta/housing.csv --output .
```

El CSV debe incluir exactamente estos nombres de columnas; puede contener otras,
pero se ignoran: `Latitude`, `Longitude` y `MedInc`. Se rechazan valores nulos,
infinitos y no numéricos. La fuente representa grupos de bloques censales de
1990, no viviendas individuales. `MedInc` se expresa en decenas de miles de USD.

## Tutorial y Método

Abre [src/housing.ipynb](src/housing.ipynb), que ya incluye resultados ejecutados.
Para volver a ejecutarlo automáticamente:

```bash
python -m nbconvert --to notebook --execute --inplace src/housing.ipynb
```

1. Exploración de tipos, nulos, duplicados, estadísticas y correlaciones.
2. División aleatoria 80/20 con semilla 42 antes de ajustar el escalador.
3. `Pipeline` con `StandardScaler` y K-Means (`n_clusters=6`, `n_init=20`).
4. Asignación de `cluster` como categoría nominal a train y test sin reajustar.
5. Mapa de train, superposición de test, perfiles e ingresos por cluster.
6. Árbol de decisión: nueve configuraciones, CV estratificada de tres folds,
   selección por F1 macro únicamente con train y comparación con baseline.
7. Evaluación en test y persistencia de ambos modelos sin incorporar test al ajuste.

Se elige un árbol por su interpretabilidad. Sus cortes alineados con los ejes
aproximan las fronteras de K-Means. Las pseudoetiquetas de CV provienen del
agrupador fijo entrenado con todo train: esa CV evalúa su imitación, no el
proceso completo de agrupación.

## Resultados

Ejecución verificada con 20.640 registros: 16.512 train y 4.128 test.

| Medida | Resultado |
| --- | ---: |
| Accuracy del árbol en test | 0,9944 |
| F1 macro del árbol en test | 0,9906 |
| Accuracy del baseline mayoritario | 0,3331 |
| Silueta de train, muestra de 2.000 | 0,4006 |
| Silueta de test, muestra de 2.000 | 0,3981 |

El árbol elegido tiene `max_depth=10` y `min_samples_leaf=1`.
Consulta [reports/analysis.md](reports/analysis.md) para la interpretación y
[reports/metrics.json](reports/metrics.json) para precision, recall, F1 y soporte
por clase. El mapa está en [reports/clusters.png](reports/clusters.png).

La silueta indica separación parcial, no clusters perfectos. Los grupos combinan
geografía e ingreso y pueden solaparse en el mapa bidimensional. Las etiquetas
0 a 5 no tienen orden. La accuracy mide concordancia con K-Means, no clasificación
contra etiquetas reales ni capacidad para predecir precios. La partición
aleatoria comparte vecindarios; generalizar a regiones nuevas requiere validación
espacial. El árbol no es necesario para predecir clusters: aporta una aproximación
interpretable al agrupador.

## Artefactos

- [models/kmeans_pipeline.joblib](models/kmeans_pipeline.joblib): escalador y K-Means.
- [models/decision_tree.joblib](models/decision_tree.joblib): clasificador supervisado.
- [data/processed/housing_train_clusters.csv](data/processed/housing_train_clusters.csv)
  y [data/processed/housing_test_clusters.csv](data/processed/housing_test_clusters.csv):
  variables originales, índice `row_id` y etiquetas, para mantener trazabilidad.
- [data/processed/housing_test_predictions.csv](data/processed/housing_test_predictions.csv):
  comparación de etiquetas de K-Means y predicciones del árbol.
- [reports/cluster_profiles.csv](reports/cluster_profiles.csv): tamaños e ingresos.
- [reports/confusion_matrix.png](reports/confusion_matrix.png): errores por cluster.
- [reports/income_by_cluster.png](reports/income_by_cluster.png): distribución del ingreso.

Los CSV no conservan tipos categóricos: al recargarlos para análisis, convierte
`cluster` a `category`. No lo incluyas entre las variables predictoras.

## Predicción con los Modelos Guardados

Ejemplo desde la raíz del repositorio:

```python
import joblib
import pandas as pd

points = pd.DataFrame({
    "Latitude": [34.05], "Longitude": [-118.24], "MedInc": [4.0]
})
kmeans = joblib.load("models/kmeans_pipeline.joblib")
tree = joblib.load("models/decision_tree.joblib")
print(kmeans.predict(points))
print(tree.predict(points))
```

Ambos reciben las tres columnas originales en ese orden. El pipeline aplica el
escalado automáticamente. Carga solo archivos joblib de confianza y con versiones
compatibles de scikit-learn. Las versiones usadas se registran en
[reports/environment.json](reports/environment.json).

## Pruebas y Entrega

```bash
python -m unittest discover -s tests -v
```

Las cinco pruebas usan datos sintéticos sin red y verifican separación de índices,
escalado solo con train, seis clusters, validación de entradas, exportación y
recarga de modelos. El script y el notebook también se ejecutaron con datos reales.

Revisa los cambios, realiza el commit y el push a tu repositorio, y entrega su URL
en 4Geeks. Los archivos de reseñas y el notebook de exploración anteriores se
conservan como material previo; no participan en este proyecto.

<details>
<summary>Referencia de la plantilla original (no necesaria para este ejercicio)</summary>

## Estructura

El proyecto está organizado de la siguiente manera:

- **`src/app.py`** → Script principal de Python donde correrá tu proyecto.
- **`src/explore.ipynb`** → Notebook para exploración y pruebas. Una vez finalizada la exploración, migra el código limpio a `app.py`.
- **`src/utils.py`** → Funciones auxiliares, como conexión a bases de datos.
- **`requirements.txt`** → Lista de paquetes de Python necesarios.
- **`models/`** → Contendrá tus clases de modelos SQLAlchemy.
- **`data/`** → Almacena los datasets en diferentes etapas:
  - **`data/raw/`** → Datos sin procesar.
  - **`data/interim/`** → Datos transformados temporalmente.
  - **`data/processed/`** → Datos listos para análisis.


## ⚡ Configuración Inicial en Codespaces (Recomendado)

No es necesario realizar ninguna configuración manual, ya que **Codespaces se configura automáticamente** con los archivos predefinidos que ha creado la academia para ti. Simplemente sigue estos pasos:

1. **Espera a que el entorno se configure automáticamente**.
   - Todos los paquetes necesarios y la base de datos se instalarán por sí mismos.
   - El `username` y `db_name` creados automáticamente están en el archivo **`.env`** en la raíz del proyecto.
2. **Una vez que Codespaces esté listo, puedes comenzar a trabajar inmediatamente**.


## 💻 Configuración en Local (Solo si no puedes usar Codespaces)

**Prerrequisitos**

Asegúrate de tener Python 3.11+ instalado en tu máquina. También necesitarás pip para instalar los paquetes de Python.

**Instalación**

Clona el repositorio del proyecto en tu máquina local.

Navega hasta el directorio del proyecto e instala los paquetes de Python requeridos:

```bash
pip install -r requirements.txt
```

**Crear una base de datos (si es necesario)**

Crea una nueva base de datos dentro del motor Postgres personalizando y ejecutando el siguiente comando: 

```bash
$ psql -U postgres -c "DO \$\$ BEGIN 
    CREATE USER mi_usuario WITH PASSWORD 'mi_contraseña'; 
    CREATE DATABASE mi_base_de_datos OWNER mi_usuario; 
END \$\$;"
```
Conéctate al motor Postgres para usar tu base de datos, manipular tablas y datos: 

```bash
$ psql -U mi_usuario -d mi_base_de_datos
```

¡Una vez que estés dentro de PSQL podrás crear tablas, hacer consultas, insertar, actualizar o eliminar datos y mucho más!

**Variables de entorno**

Crea un archivo .env en el directorio raíz del proyecto para almacenar tus variables de entorno, como tu cadena de conexión a la base de datos:

```makefile
DATABASE_URL="postgresql://<USUARIO>:<CONTRASEÑA>@<HOST>:<PUERTO>/<NOMBRE_BD>"

#example
DATABASE_URL="postgresql://mi_usuario:mi_contraseña@localhost:5432/mi_base_de_datos"
```

## Ejecutando la Aplicación

Para ejecutar la aplicación, ejecuta el script app.py desde la raíz del directorio del proyecto:

```bash
python src/app.py
```

## Añadiendo Modelos

Para añadir clases de modelos SQLAlchemy, crea nuevos archivos de script de Python dentro del directorio models/. Estas clases deben ser definidas de acuerdo a tu esquema de base de datos.

Definición del modelo de ejemplo (`models/example_model.py`):

```py
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column

Base = declarative_base()

class ExampleModel(Base):
    __tablename__ = 'example_table'
    id: Mapped[int] = mapped_column(primary_key=True)
    username: Mapped[str] = mapped_column(unique=True)
```

## Trabajando con Datos

Puedes colocar tus conjuntos de datos brutos en el directorio data/raw, conjuntos de datos intermedios en data/interim, y los conjuntos de datos procesados listos para el análisis en data/processed.

Para procesar datos, puedes modificar el script app.py para incluir tus pasos de procesamiento de datos, utilizando pandas para la manipulación y análisis de datos.

## Contribuyentes

Esta plantilla fue construida como parte del [Data Science and Machine Learning Bootcamp](https://4geeksacademy.com/us/coding-bootcamps/datascience-machine-learning) de 4Geeks Academy por [Alejandro Sanchez](https://twitter.com/alesanchezr) y muchos otros contribuyentes. Descubre más sobre [los programas BootCamp de 4Geeks Academy](https://4geeksacademy.com/us/programs) aquí.

Otras plantillas y recursos como este se pueden encontrar en la página de GitHub de la escuela.
</details>