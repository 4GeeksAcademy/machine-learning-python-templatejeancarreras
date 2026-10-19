# Clasificador de Imágenes: Gatos y Perros

Implementación del tutorial con EfficientNetB0, exploración de imágenes, entrenamiento base, optimización, evaluación y guardado. El código está en [src/image_classifier.py](src/image_classifier.py). La aplicación de sentimientos existente se conserva sin cambios.

## Estado

El pipeline se verificó con imágenes sintéticas: preparación, ambos modos de carga, entrenamiento, interrupción simulada, reanudación, checkpoints, recarga y predicciones. Las once pruebas pasan. El dataset de Microsoft se descargó y preparó localmente: 24.998 imágenes válidas y dos corruptas excluidas. **No se ha entrenado ni medido precisión sobre las imágenes reales**. Ni el dataset ni un modelo entrenado con datos reales se distribuyen con el repositorio.

## 1. Instalación y Dataset

Desde la raíz del repositorio, con Python 3.11–3.13:

```bash
python -m pip install -r requirements-image-classifier.txt
```

Las dependencias del clasificador están separadas de las de la aplicación anterior. Se ha verificado el proyecto con Python 3.13 y TensorFlow para CPU. Entrenar EfficientNetB0 desde cero sobre 25.000 imágenes en CPU puede tardar muchas horas; para el entrenamiento completo conviene un entorno con GPU y la distribución de TensorFlow adecuada para ese entorno.

Descarga los datos de [Dogs vs. Cats en Kaggle](https://www.kaggle.com/c/dogs-vs-cats/data), aceptando sus condiciones. No hacen falta credenciales en el código. El archivo de la competición puede contener otro ZIP llamado `train.zip`: extrae ese ZIP interior antes de usar el comando siguiente. El script acepta el `train.zip` que contiene directamente los JPG etiquetados, no el ZIP exterior.

```bash
python src/image_classifier.py --archive /ruta/train.zip --prepare-only
```

Alternativamente, descomprime las fotos en `data/raw/train/`, o indica su ubicación mediante `--data-dir`. Los nombres deben seguir el patrón `cat.0.jpg` y `dog.0.jpg`. La búsqueda es recursiva y las imágenes corruptas quedan registradas y excluidas. El `test1` sin etiquetas de Kaggle no sirve para calcular métricas y no se utiliza: el test etiquetado se reserva a partir de `train`.

No vuelvas a pasar `--archive` si ya has extraído esas mismas fotos: el script rechaza sobrescribir archivos. En las siguientes ejecuciones usa `--data-dir`.

## 2. Explorar y Preparar

```bash
python src/image_classifier.py --data-dir data/raw/train --prepare-only
```

Se generan dos figuras de 3×3 con las primeras nueve imágenes de cada clase en orden de nombre, conservando su aspecto original e indicando sus dimensiones. Durante la carga para el modelo, todas se convierten a RGB de **200×200 píxeles**.

La división estratificada y reproducible utiliza aproximadamente 70% para entrenamiento, 15% para validación y 15% para test, con semilla 42. Para el dataset completo de 25.000 imágenes son 17.500/3.750/3.750. El manifiesto permite comprobar que ningún archivo pertenece a más de una partición. Se crean directorios `train`, `validation` y `test`, con subdirectorios `cat` y `dog`, mediante enlaces simbólicos sin duplicar los originales.

El modo `--mode auto` considera tanto la RAM disponible como los límites del contenedor. Si supera 12 GiB y la estimación deja margen suficiente, carga cada partición mediante las utilidades de Keras en una tupla `(imágenes, etiquetas)`, conservando `uint8`. En caso contrario utiliza `ImageDataGenerator.flow_from_directory()` para leer por lotes. Puede forzarse `--mode directory`; `--mode memory` se rechaza si no hay memoria suficiente. Ambas rutas usan generadores con lotes `uint8` para evitar una copia completa a `float32`.

## 3. Entrenar y Optimizar

```bash
python src/image_classifier.py --data-dir data/raw/train --mode auto \
  --batch-size 16 --baseline-epochs 5 --epochs 30
```

La arquitectura es `EfficientNetB0(weights=None, include_top=False)` → `GlobalAveragePooling2D` → `Dense(128, relu)` → `Dense(2, softmax)`. Se usan Adam, etiquetas categóricas y `categorical_crossentropy`; `cat=0` y `dog=1`.

El ejemplo del enunciado indica 224×224, pero el requisito de preparación pide 200×200. EfficientNetB0 admite 200×200 al utilizar `include_top=False`, por lo que se mantiene ese tamaño de extremo a extremo. No se aplica `rescale=1/255`: EfficientNetB0 ya incluye su reescalado y espera píxeles en rango 0–255.

Se comparan dos configuraciones, ambas inicializadas desde cero con la misma semilla:

| Configuración | Learning rate inicial | Aumento de datos | Épocas máximas por defecto |
| --- | --- | --- | --- |
| Base | 0.001 | No | 5 |
| Optimizada | 0.0001 | Rotación, desplazamiento, zoom y volteo horizontal | 30 |

Ambas usan `ModelCheckpoint(save_best_only=True)` y `EarlyStopping(patience=5, restore_best_weights=True)`. La optimizada añade `ReduceLROnPlateau`. Solo se aumentan las imágenes de entrenamiento. Se utiliza `model.fit()` porque `fit_generator()` ya no existe en Keras actual.

Se elige el checkpoint con menor pérdida en **validación** entre las dos configuraciones. Solo después se recarga el ganador para predecir el test, sin barajarlo. Una configuración llamada «optimizada» no garantiza un mejor resultado: si gana la base, se guarda esa como mejor modelo.

Para repetir experimentos sin mezclar salidas, usa una carpeta distinta:

```bash
python src/image_classifier.py --data-dir data/raw/train \
  --output models/experimento_02 --batch-size 8
```

### Reanudar Tras Una Interrupción

El entrenamiento guarda automáticamente un respaldo al finalizar cada época mediante `BackupAndRestore`. Este respaldo recupera pesos, variables del optimizador, learning rate y la época alcanzada. Se conserva además una copia del respaldo anterior. No es lo mismo que el checkpoint de mejor validación: reanudar debe continuar desde el último estado guardado, aunque ese estado no sea el mejor modelo.

Para iniciar un entrenamiento nuevo con los valores por defecto:

```bash
python src/image_classifier.py --data-dir data/raw/train
```

Después de una interrupción, vuelve a abrir Codespaces y ejecuta:

```bash
python src/image_classifier.py --data-dir data/raw/train --resume
```

Mantén el mismo `--output`, dataset, `--batch-size`, `--baseline-epochs` y `--epochs` que utilizaste al iniciar. Los límites de épocas son el total previsto, no épocas adicionales. Por ejemplo, para reanudar el experimento anterior con lotes de ocho:

```bash
python src/image_classifier.py --data-dir data/raw/train \
  --output models/experimento_02 --batch-size 8 --resume
```

El script comprueba rutas, etiquetas, tamaños y fechas de modificación de las imágenes, además de la configuración. Reconstruye las mismas particiones deterministas, conserva el historial, protege el mejor checkpoint y omite las etapas terminadas. Si ambas etapas terminaron, `--resume` solo repite la generación de los resultados finales. No se sobrescribe un entrenamiento existente sin indicar `--resume`; para empezar desde cero elige otra carpeta de salida.

Si el entorno se detiene a mitad de una época, se pierde el progreso posterior al último respaldo; si aún no terminó la primera época, esa etapa empieza desde el principio. Los contadores de paciencia de `EarlyStopping` y `ReduceLROnPlateau`, y la secuencia de aumentos aleatorios, se reinician al reanudar: no se garantiza una ejecución idéntica bit a bit a una ininterrumpida. La tasa de aprendizaje vigente sí se restaura con el optimizador.

No borres `backup/`, los historiales ni los JSON de estado de la carpeta de salida mientras necesites reanudar. Los modelos guardados antes de incorporar este mecanismo no contienen toda la información requerida para una reanudación automática. Este mecanismo no evita que GitHub detenga el Codespace.

## 4. Analizar y Guardar

Los archivos se guardan por defecto en `models/image_classifier/`:

| Archivo | Contenido |
| --- | --- |
| `examples_cat.png`, `examples_dog.png` | Muestras originales de cada clase |
| `dataset_summary.json`, `split_manifest.csv` | Conteos, memoria, etiquetas, particiones e imágenes rechazadas |
| `baseline.keras`, `optimized.keras` | Mejor checkpoint de cada entrenamiento |
| `best_model.keras` | Modelo elegido por validación, listo para recargar |
| `training_config.json` | Configuración e identidad del dataset para validar la reanudación |
| `backup/baseline/`, `backup/optimized/` | Últimos respaldos de modelo, optimizador y época |
| `baseline_history.csv`, `optimized_history.csv` | Métricas persistentes de cada época |
| `baseline_complete.json`, `optimized_complete.json` | Etapas completadas que no se deben repetir |
| `learning_curves.png` | Pérdida y accuracy de entrenamiento y validación |
| `confusion_matrix.png` | Errores por clase en test |
| `test_predictions.csv` | Archivo, etiqueta real, predicción y probabilidades |
| `metrics.json` | Historiales, configuración, selección y métricas finales |

Para interpretar los resultados, compara la pérdida de validación de ambas configuraciones. Si la pérdida de entrenamiento baja pero la de validación sube, hay indicios de sobreajuste. Revisa la matriz de confusión, precision, recall y F1 por clase para detectar errores desiguales entre gatos y perros. Las curvas comparan configuraciones y presupuestos de entrenamiento distintos, no demuestran qué cambio individual produjo la mejora. Cualquier ajuste adicional debe decidirse con validación, no reutilizando el test para escoger hiperparámetros. Entrenar desde cero puede requerir más datos o épocas; transfer learning con pesos de ImageNet sería un experimento posterior, no el modelo `weights=None` solicitado.

Los ZIP, las fotos, los pesos `.keras` y las carpetas de respaldo bajo `models/` se excluyen de Git para evitar subir archivos masivos; los modelos se conservan localmente en la carpeta indicada. Si necesitas publicar pesos o trasladar respaldos a otro entorno, utiliza Git LFS o un almacén de artefactos. Los metadatos y las gráficas sí pueden incluirse en la entrega tras el entrenamiento real.

## Pruebas

```bash
python -m unittest discover -s tests -v
RUN_TRAINING_SMOKE=1 python -m unittest discover -s tests -v
```

El primer comando prueba etiquetas, particiones sin solapamiento, memoria, ZIP, imágenes corruptas, ambos cargadores y una interrupción simulada usando una red pequeña para verificar pesos, optimizador, época y etapas completadas. El segundo incluye además entrenamiento, evaluación, persistencia y `--resume` por CLI con EfficientNetB0 y datos sintéticos temporales; las once pruebas tardaron unos 153 segundos en este entorno. Es una comprobación funcional, no una medida de calidad para gatos y perros.

## Plantilla Original

Esta plantilla está diseñada para impulsar proyectos de ciencia de datos proporcionando una configuración básica para conexiones de base de datos, procesamiento de datos, y desarrollo de modelos de aprendizaje automático. Incluye una organización estructurada de carpetas para tus conjuntos de datos y un conjunto de paquetes de Python predefinidos necesarios para la mayoría de las tareas de ciencia de datos.

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