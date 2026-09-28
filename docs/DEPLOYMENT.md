# Documentación de despliegue

## 1. Entorno

El prototipo se desarrolló y verificó en Windows con Python 3.13.3 dentro de un
entorno virtual `.venv`. El código usa rutas relativas y puede ejecutarse en
Windows, Linux o macOS.

## 2. Dependencias

Las dependencias de ejecución se encuentran fijadas en `requirements.txt`:
Streamlit, pandas, NumPy, scikit-learn, joblib y Matplotlib.

## 3. Instalación

```bash
git clone <URL_DEL_REPOSITORIO>
cd proyecto-desaprobacion-ia
python -m venv .venv
```

Active el entorno virtual e instale las dependencias:

```bash
pip install -r requirements.txt
```

## 4. Ejecución

La aplicación se inicia desde la raíz del proyecto:

```bash
streamlit run app.py
```

## 5. API

La versión actual del prototipo no utiliza una API independiente. La aplicación
Streamlit ejecuta directamente los pipelines de inferencia cargados mediante
joblib. Una API REST mediante FastAPI puede incorporarse como trabajo futuro si
el modelo debe ser consumido por sistemas externos.

## 6. Interfaz

- Tecnología: Streamlit.
- Tipo: aplicación web.
- Entradas: características contextuales de un registro agregado o un CSV.
- Procesamiento: preprocesamiento y modelos almacenados como pipelines.
- Salidas: clasificación, probabilidad estimada y tasa estimada.

## 7. Infraestructura

La arquitectura propuesta es sencilla:

```text
Código fuente en GitHub
        ↓
Streamlit Community Cloud
        ↓
Pipelines joblib incluidos en el repositorio
        ↓
Formulario o CSV cargado temporalmente por el usuario
```

No se utilizan contenedores, servicios de nube adicionales, microservicios ni
bases de datos.

## 8. Modelo

El clasificador es un `RandomForestClassifier` para `HAY_DESAPROBACION`. El modelo
complementario es una `LinearRegression` para `TASA_DESAPROBACION`. Ambos pipelines
incluyen imputación, escalado numérico y One-Hot Encoding de `gestion` y
`dsc_nivel`.

Los modelos reciben exclusivamente `Edad`, `TotalEstudiantes`, `Discapacidad`,
`Venezolanos`, `Extranjeros`, `gestion` y `dsc_nivel`. Se excluyen resultados
académicos del mismo periodo e identificadores institucionales.

## 9. Almacenamiento

Los modelos entrenados se almacenan como archivos joblib. Los CSV cargados se
procesan temporalmente en memoria durante la sesión y no se guardan de forma
persistente. El dataset de entrenamiento no se necesita para inferencia después
de serializar los pipelines.

## 10. Despliegue

1. Crear un repositorio en GitHub.
2. Subir esta carpeta sin `.venv` ni los CSV originales.
3. Confirmar que los dos archivos joblib respeten los límites de GitHub.
4. Entrar a Streamlit Community Cloud.
5. Conectar la cuenta de GitHub.
6. Seleccionar el repositorio y la rama `main`.
7. Indicar `app.py` como archivo principal.
8. Seleccionar una versión de Python compatible con las dependencias fijadas.
9. Desplegar y revisar los registros de instalación.

No se requieren secretos ni variables de entorno para esta versión.

## 11. Reproducibilidad

`src/train_models.py` reproduce el entrenamiento a partir de los CSV originales.
Mantiene `random_state=42`, el split 80/20 de 2023, el preprocesamiento documentado
y la evaluación temporal sobre 2024. Después de guardar los pipelines, compara
sus predicciones antes y después de cargarlos con joblib.

### Incidencia resuelta durante la reproducción

- **Problema encontrado:** una primera versión eliminaba filas repetidas después
  de leer únicamente los predictores y el objetivo.
- **Causa:** registros educativos distintos pueden compartir exactamente los
  mismos valores en ese subconjunto de columnas.
- **Consecuencia:** el entrenamiento quedaba reducido de forma incorrecta y las
  métricas no coincidían con el notebook.
- **Solución aplicada:** se conserva cada fila del CSV. La comprobación original
  de duplicados se realizó sobre las 27 columnas y no encontró filas totalmente
  idénticas. Con esta corrección, las métricas reproducen las del cuaderno.

## 12. Limitaciones

- La clase positiva está desbalanceada y el Recall es moderado.
- La Regresión Lineal tiene R² bajo y puede producir valores fuera de 0 %–100 %.
- La tasa de desaprobación se concentra en cero.
- Los datos son registros agregados y no contienen información académica individual.
- Los resultados no representan predicciones individuales ni relaciones causales.
- La primera versión no incluye búsqueda institucional para evitar incorporar un
  archivo auxiliar pesado al despliegue.
