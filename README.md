# Analizador de Desaprobación Escolar

Prototipo web para analizar la presencia y la tasa de desaprobación en registros
educativos agregados del Perú. La aplicación no realiza predicciones individuales
de estudiantes.

## Datos y modelos

El proyecto utiliza los archivos de Matriculación y Trayectoria Estudiantil del
MINEDU/SIAGIE. El periodo 2023 se emplea para desarrollo y entrenamiento; 2024 se
reserva para evaluación temporal.

Los siete predictores son `Edad`, `TotalEstudiantes`, `Discapacidad`,
`Venezolanos`, `Extranjeros`, `gestion` y `dsc_nivel`. El pipeline imputa las
variables numéricas con la mediana y las categóricas con la moda, escala los datos
numéricos y aplica One-Hot Encoding a las categorías.

- Clasificación: Random Forest para `HAY_DESAPROBACION`.
- Regresión: Regresión Lineal para `TASA_DESAPROBACION`.

Las variables académicas de resultado del mismo periodo se excluyen para evitar
data leakage. También se excluyen identificadores y nombres institucionales.

## Resultados principales

En la evaluación temporal de 2024, Random Forest obtuvo Accuracy 0.8960,
Precision 0.7086, Recall 0.3988, F1 0.5104 y ROC-AUC 0.8898. La Regresión Lineal
obtuvo MAE 5.8704, RMSE 13.3167 y R² 0.0854. La regresión se presenta como una
estimación complementaria por su baja capacidad explicativa.

## Instalación

Se verificó el proyecto con Python 3.13.3. Las dependencias fijadas también son
compatibles con versiones modernas de Python admitidas por Streamlit.

```bash
git clone <URL_DEL_REPOSITORIO>
cd proyecto-desaprobacion-ia
python -m venv .venv
```

En Windows:

```bash
.venv\Scripts\activate
```

En Linux o macOS:

```bash
source .venv/bin/activate
```

Instale las dependencias:

```bash
pip install -r requirements.txt
```

## Ejecución

```bash
streamlit run app.py
```

La aplicación incluye análisis de un registro, procesamiento de CSV, descarga de
resultados, métricas del proyecto e información metodológica.

Un ejemplo de entrada está disponible en `data/ejemplo_entrada.csv`.

## Reproducir el entrenamiento

Coloque los CSV originales de 2023 y 2024 en la carpeta que contiene este proyecto
y ejecute:

```bash
python -m src.train_models
```

También puede indicar otra carpeta sin introducir rutas en el código:

```bash
python -m src.train_models --data-dir <CARPETA_DE_LOS_CSV>
```

El script reproduce el split y los pipelines finales, guarda ambos modelos y
verifica que sus predicciones no cambien después de serializarlos.

## Uso del CSV

El archivo debe contener como mínimo:

```text
Edad,TotalEstudiantes,Discapacidad,Venezolanos,Extranjeros,gestion,dsc_nivel
```

Los conteos deben ser no negativos y no pueden superar `TotalEstudiantes`. Los
archivos se procesan en memoria y no se almacenan.

## Pruebas

```bash
python -m unittest discover -s tests -v
```

## Despliegue

La guía para GitHub y Streamlit Community Cloud se encuentra en
`docs/DEPLOYMENT.md`.

## Limitaciones

El conjunto presenta desbalance de clases y la clasificación tiene Recall
moderado. La tasa se concentra en cero y la regresión tiene R² bajo; además, una
Regresión Lineal puede producir valores fuera del intervalo 0 %–100 %. Los datos
son agregados, por lo que el sistema no permite conclusiones individuales ni
causales.

