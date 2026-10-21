# Resultados: agrupacion de California Housing

## Datos y protocolo

Se analizaron 20,640 grupos de bloques censales, no viviendas individuales.
Variables: Latitude, Longitude y MedInc. No se utiliza el precio como predictor.
MedInc expresa el ingreso mediano en decenas de miles de dolares de 1990.
Valores nulos en estas variables: 0.
Filas duplicadas en estas tres variables: 5; se conservan,
pues no demuestran que sean el mismo bloque censal.
Particion aleatoria 80/20, semilla 42: 16,512 train y 4,128 test.
StandardScaler y K-Means se ajustaron exclusivamente con train.
Se fijo k=6 por el enunciado; no se afirma que sea el numero optimo.

## Interpretacion de los clusters

El mapa muestra separacion geografica, pero no toda la informacion: el ingreso
es la tercera dimension y por eso pueden solaparse colores en latitud/longitud.
Cada variable tiene el mismo peso tras estandarizar; los grados no son distancias
geodesicas. Los centroides del mapa son proyecciones de centros tridimensionales.
El cluster 5 tiene la latitud media mas alta y el 1 la mas baja.
El cluster 0 tiene el ingreso medio mas alto:
10.44 decenas de miles de USD.
Los perfiles completos y tamanos estan en cluster_profiles.csv; las etiquetas
0 a 5 son categorias nominales, no un ranking de calidad ni de ingreso.

Silhouette train (muestra de hasta 2000): 0.40055299144426537.
Silhouette test (muestra de hasta 2000): 0.3981055303918927.
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
Mejores parametros: {'max_depth': 10, 'min_samples_leaf': 1}.
F1 macro CV: 0.9884.
Accuracy train: 0.9995.
Accuracy test: 0.9944.
F1 macro test: 0.9906.
Accuracy del baseline mayoritario en test: 0.3331.
Brecha accuracy train-test: 0.0051.
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
