"""Interfaz Streamlit del Analizador de Desaprobación Escolar."""

from __future__ import annotations

from io import StringIO
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.predict import load_models, predict_records
from src.validation import validate_and_prepare, validate_record


PROJECT_ROOT = Path(__file__).resolve().parent
ASSET_DIR = PROJECT_ROOT / "assets"

CLASSIFIER_RESULTS = pd.DataFrame(
    [
        ["Regresión Logística", 0.8769, 0.6904, 0.2211, 0.3350, 0.8512],
        ["KNN", 0.8720, 0.5655, 0.3759, 0.4516, 0.8005],
        ["Árbol de Decisión", 0.8815, 0.6449, 0.3445, 0.4491, 0.7959],
        ["Random Forest", 0.8835, 0.6476, 0.3709, 0.4717, 0.8586],
    ],
    columns=["Modelo", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"],
)

TEMPORAL_RESULTS = pd.DataFrame(
    [["Random Forest", 0.8960, 0.7086, 0.3988, 0.5104, 0.8898]],
    columns=["Modelo", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"],
)


st.set_page_config(
    page_title="Analizador de Desaprobación Escolar",
    page_icon="📊",
    layout="wide",
)

st.markdown(
    """
    <style>
    :root {
        --unfv-orange: #E26400;
        --ink: #1f1f1f;
        --muted: #5f6368;
        --surface: #ffffff;
        --soft: #f5f6f7;
        --border: #e2e4e7;
    }
    .stApp { background: #f8f9fa; color: var(--ink); }
    .block-container { max-width: 1180px; padding-top: 2rem; padding-bottom: 3rem; }
    h1, h2, h3, h4 { color: var(--ink); }
    h1 { letter-spacing: -0.02em; }
    p, label, .stCaption { color: #35383b; }
    .title-accent {
        width: 72px;
        height: 4px;
        margin: 0.6rem 0 1.2rem;
        background: var(--unfv-orange);
        border-radius: 999px;
    }
    div[data-testid="stForm"] {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 1rem;
    }
    .stButton > button,
    .stFormSubmitButton > button {
        background: var(--unfv-orange) !important;
        border-color: var(--unfv-orange) !important;
        color: #000000 !important;
        font-weight: 700;
    }
    .stButton > button p,
    .stFormSubmitButton > button p {
        color: #000000 !important;
    }
    .stButton > button:hover,
    .stFormSubmitButton > button:hover {
        background: #C95700;
        border-color: #C95700;
        color: #000000;
    }
    .stDownloadButton > button {
        border-color: var(--unfv-orange);
        color: var(--ink);
        font-weight: 600;
    }
    .stDownloadButton > button:hover {
        border-color: var(--unfv-orange);
        color: #000000;
        background: #fff7e8;
    }
    div[data-testid="stMetric"] {
        background: var(--surface);
        border: 1px solid var(--border);
        border-top: 3px solid var(--unfv-orange);
        border-radius: 8px;
        padding: 0.8rem 1rem;
    }
    .method-note {
        background: #fffaf0;
        border-left: 4px solid var(--unfv-orange);
        padding: 0.9rem 1rem;
        margin-top: 1rem;
        color: #343434;
    }
    .small-note { color: var(--muted); font-size: 0.92rem; }
    .academic-footer {
        margin-top: 2.5rem;
        padding: 1.25rem 1rem 0.3rem;
        border-top: 1px solid var(--border);
        color: #4f5356;
        text-align: center;
        font-size: 0.86rem;
        line-height: 1.65;
    }
    .academic-footer p { margin: 0.1rem 0; }
    .academic-footer .footer-disclaimer {
        margin-top: 0.45rem;
        color: #707478;
        font-size: 0.76rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_models():
    return load_models()


def methodological_note() -> None:
    st.markdown(
        """
        <div class="method-note">
        Los resultados corresponden a registros educativos agregados y no representan
        predicciones individuales de estudiantes. El prototipo constituye una herramienta
        de apoyo al análisis y no reemplaza la evaluación realizada por especialistas o
        autoridades educativas.
        </div>
        """,
        unsafe_allow_html=True,
    )


def academic_footer() -> None:
    st.markdown(
        """
        <footer class="academic-footer">
            <p>Proyecto académico desarrollado para el curso de Inteligencia Artificial – Universidad Nacional Federico Villarreal (UNFV).</p>
            <p>© 2026. Uso académico.</p>
            <p>Datos utilizados: Ministerio de Educación del Perú – SIAGIE.</p>
            <p class="footer-disclaimer">Este prototipo tiene fines académicos y de demostración. Los resultados corresponden a registros educativos agregados y no representan predicciones individuales de estudiantes.</p>
        </footer>
        """,
        unsafe_allow_html=True,
    )


def read_uploaded_csv(raw: bytes) -> pd.DataFrame:
    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            text = raw.decode(encoding)
            return pd.read_csv(StringIO(text), sep=None, engine="python")
        except (UnicodeDecodeError, pd.errors.ParserError) as exc:
            last_error = exc
    raise ValueError("No se pudo leer el CSV cargado.") from last_error


try:
    classifier, regressor, metadata = get_models()
except Exception as exc:
    st.error(
        "No se pudieron cargar los modelos. Ejecute primero "
        "`python -m src.train_models` desde la carpeta del proyecto."
    )
    st.exception(exc)
    st.stop()

categories = metadata["categories_2023"]

st.title("Analizador de Desaprobación Escolar")
st.caption(
    "Prototipo basado en aprendizaje automático para el análisis de registros "
    "educativos agregados del Perú"
)
st.markdown('<div class="title-accent"></div>', unsafe_allow_html=True)

tab_record, tab_csv, tab_results, tab_info = st.tabs(
    [
        "Analizar registro",
        "Analizar archivo CSV",
        "Modelo y resultados",
        "Información del proyecto",
    ]
)

with tab_record:
    st.subheader("Análisis de un registro agregado")
    st.write(
        "Ingrese las características contextuales del grupo. Los campos de nivel y "
        "gestión utilizan las categorías observadas en los datos de 2023."
    )

    with st.form("record_form"):
        left, right = st.columns(2)
        with left:
            educational_level = st.selectbox(
                "Nivel educativo", categories["dsc_nivel"]
            )
            management = st.selectbox("Gestión", categories["gestion"])
            age = st.number_input("Edad", min_value=0, value=9, step=1)
            total_students = st.number_input(
                "Total de estudiantes", min_value=1, value=20, step=1
            )
        with right:
            disability = st.number_input(
                "Estudiantes con discapacidad", min_value=0, value=0, step=1
            )
            venezuelan = st.number_input(
                "Estudiantes venezolanos", min_value=0, value=0, step=1
            )
            foreign = st.number_input(
                "Estudiantes extranjeros", min_value=0, value=0, step=1
            )
        submitted = st.form_submit_button("ANALIZAR REGISTRO", type="primary")

    if submitted:
        record = {
            "Edad": age,
            "TotalEstudiantes": total_students,
            "Discapacidad": disability,
            "Venezolanos": venezuelan,
            "Extranjeros": foreign,
            "gestion": management,
            "dsc_nivel": educational_level,
        }
        prepared, errors = validate_record(record, categories)
        if errors:
            for error in errors:
                st.error(error)
        else:
            result = predict_records(prepared, classifier, regressor).iloc[0]
            presence = "SÍ" if result["PRED_HAY_DESAPROBACION"] == 1 else "NO"
            probability = result["PROB_DESAPROBACION"] * 100
            estimated_rate = result["TASA_DESAPROBACION_ESTIMADA"]

            st.markdown("### Resultados del análisis")
            result_columns = st.columns(3)
            result_columns[0].metric(
                "Presencia estimada de desaprobación", presence
            )
            result_columns[1].metric("Probabilidad estimada", f"{probability:.1f} %")
            result_columns[2].metric(
                "Tasa estimada de desaprobación", f"{estimated_rate:.1f} %"
            )
            if estimated_rate < 0 or estimated_rate > 100:
                st.warning(
                    "La Regresión Lineal produjo un valor fuera del rango teórico de "
                    "0 % a 100 %. Esta limitación fue identificada durante la evaluación "
                    "y el valor se muestra sin modificar."
                )
            else:
                st.caption(
                    "La tasa es una estimación complementaria. El bajo R² del modelo "
                    "limita su capacidad explicativa."
                )

    methodological_note()
    academic_footer()

with tab_csv:
    st.subheader("Análisis por archivo CSV")
    st.write(
        "El archivo debe incluir las columnas: Edad, TotalEstudiantes, Discapacidad, "
        "Venezolanos, Extranjeros, gestion y dsc_nivel."
    )
    uploaded_file = st.file_uploader("Seleccione un archivo", type=["csv"])

    if uploaded_file is not None:
        try:
            uploaded_data = read_uploaded_csv(uploaded_file.getvalue())
        except ValueError as exc:
            st.error(str(exc))
        else:
            prepared, errors = validate_and_prepare(uploaded_data, categories)
            if errors:
                st.error("El archivo no superó la validación.")
                for error in errors:
                    st.write(f"- {error}")
            else:
                predictions = predict_records(prepared, classifier, regressor)
                output = uploaded_data.copy()
                for column in (
                    "PRED_HAY_DESAPROBACION",
                    "PROB_DESAPROBACION",
                    "TASA_DESAPROBACION_ESTIMADA",
                ):
                    output[column] = predictions[column].to_numpy()

                positive_count = int(output["PRED_HAY_DESAPROBACION"].sum())
                total_count = len(output)
                negative_count = total_count - positive_count
                positive_percentage = positive_count / total_count * 100
                average_rate = output["TASA_DESAPROBACION_ESTIMADA"].mean()

                st.success(f"Se procesaron {total_count:,} registros correctamente.")
                summary_columns = st.columns(5)
                summary_columns[0].metric("Registros", f"{total_count:,}")
                summary_columns[1].metric("Con presencia", f"{positive_count:,}")
                summary_columns[2].metric("Sin presencia", f"{negative_count:,}")
                summary_columns[3].metric("Clasificación positiva", f"{positive_percentage:.1f} %")
                summary_columns[4].metric("Tasa estimada media", f"{average_rate:.1f} %")

                st.dataframe(output, width="stretch", height=330)
                csv_bytes = output.to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    "Descargar resultados",
                    data=csv_bytes,
                    file_name="resultados_analisis.csv",
                    mime="text/csv",
                    type="primary",
                )

                chart_left, chart_right = st.columns(2)
                with chart_left:
                    fig, ax = plt.subplots(figsize=(5.8, 3.8))
                    ax.bar(
                        ["Sin presencia", "Con presencia"],
                        [negative_count, positive_count],
                        color=["#6b7075", "#E26400"],
                    )
                    ax.set_ylabel("Registros")
                    ax.set_title("Distribución de la clasificación")
                    st.pyplot(fig, clear_figure=True)
                with chart_right:
                    fig, ax = plt.subplots(figsize=(5.8, 3.8))
                    ax.hist(
                        output["PROB_DESAPROBACION"],
                        bins=20,
                        color="#E26400",
                        edgecolor="white",
                    )
                    ax.set_xlabel("Probabilidad estimada")
                    ax.set_ylabel("Registros")
                    ax.set_title("Distribución de probabilidades")
                    st.pyplot(fig, clear_figure=True)

                outside_range = output["TASA_DESAPROBACION_ESTIMADA"].lt(0) | output[
                    "TASA_DESAPROBACION_ESTIMADA"
                ].gt(100)
                if outside_range.any():
                    st.warning(
                        f"La tasa estimada quedó fuera de 0 % a 100 % en "
                        f"{int(outside_range.sum())} registro(s). Los valores se conservan "
                        "sin modificación para mostrar la limitación de la Regresión Lineal."
                    )

    methodological_note()
    academic_footer()

with tab_results:
    st.subheader("Evaluación temporal 2024")
    temporal_result = TEMPORAL_RESULTS.iloc[0]
    temporal_columns = st.columns(5)
    for column, metric_name in zip(
        temporal_columns,
        ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"],
    ):
        column.metric(metric_name, f"{temporal_result[metric_name]:.4f}")

    st.markdown("### Modelo seleccionado: Random Forest")
    st.info(
        "Random Forest fue seleccionado por presentar el mejor equilibrio general, "
        "especialmente en F1 y ROC-AUC."
    )

    st.markdown("### Comparación de algoritmos en validación 2023")
    st.dataframe(
        CLASSIFIER_RESULTS.style.format(precision=4),
        width="stretch",
        hide_index=True,
    )
    st.info(
        "La comparación entre algoritmos se realizó sobre una muestra común de "
        "entrenamiento y validación de 2023. Posteriormente, Random Forest fue "
        "reentrenado con el conjunto de entrenamiento 2023 y evaluado sobre los "
        "registros completos de 2024 como periodo posterior."
    )

    st.markdown("### Resultados gráficos de clasificación")
    image_left, image_right = st.columns(2)
    with image_left:
        st.image(ASSET_DIR / "matriz_confusion_2024.png", width="stretch")
    with image_right:
        st.image(ASSET_DIR / "curva_roc_2024.png", width="stretch")

    st.markdown("### Importancia de variables")
    st.image(
        ASSET_DIR / "importancia_variables.png",
        caption=(
            "Las importancias de las categorías codificadas se sumaron dentro de su "
            "variable original. No representan efectos causales."
        ),
        width="stretch",
    )

    st.markdown("### Estimación complementaria de la tasa de desaprobación")
    regression_columns = st.columns(3)
    regression_columns[0].metric("MAE", "5.8704")
    regression_columns[1].metric("RMSE", "13.3167")
    regression_columns[2].metric("R²", "0.0854")
    st.caption(
        "La Regresión Lineal supera la referencia basada en la media en MAE y RMSE, "
        "pero el bajo R² indica una capacidad explicativa limitada."
    )
    methodological_note()
    academic_footer()

with tab_info:
    st.subheader("Información del proyecto")
    information_left, information_right = st.columns(2)
    with information_left:
        with st.container(border=True):
            st.markdown("#### Problema y alcance")
            st.write(
                "Análisis de la presencia de desaprobación escolar en registros "
                "educativos agregados del Perú."
            )
            st.markdown("**Unidad de análisis:** registro educativo agregado.")
        with st.container(border=True):
            st.markdown("#### Fuente y periodos")
            st.write(
                "Ministerio de Educación del Perú (MINEDU), SIAGIE, "
                "Matriculación y Trayectoria Estudiantil."
            )
            st.markdown(
                "- **2023:** desarrollo, entrenamiento y validación.\n"
                "- **2024:** evaluación temporal posterior."
            )
    with information_right:
        with st.container(border=True):
            st.markdown("#### Técnicas evaluadas")
            st.markdown(
                "- Regresión Logística\n"
                "- KNN\n"
                "- Árbol de Decisión\n"
                "- Random Forest\n"
                "- Regresión Lineal"
            )
        with st.container(border=True):
            st.markdown("#### Limitaciones principales")
            st.markdown(
                "- Desbalance de clases y Recall moderado.\n"
                "- Baja capacidad explicativa de la regresión.\n"
                "- Concentración de la tasa de desaprobación en cero.\n"
                "- Datos agregados sin variables académicas individuales.\n"
                "- Los resultados no representan predicciones individuales ni "
                "relaciones causales."
            )
    methodological_note()
    academic_footer()
