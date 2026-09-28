"""Reproduce el entrenamiento final documentado en el notebook."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_DIR = PROJECT_ROOT.parent
MODEL_DIR = PROJECT_ROOT / "models"
ASSET_DIR = PROJECT_ROOT / "assets"

PREDICTORS = [
    "Edad",
    "TotalEstudiantes",
    "Discapacidad",
    "Venezolanos",
    "Extranjeros",
    "gestion",
    "dsc_nivel",
]
NUMERIC_COLUMNS = [
    "Edad",
    "TotalEstudiantes",
    "Discapacidad",
    "Venezolanos",
    "Extranjeros",
]
CATEGORICAL_COLUMNS = ["gestion", "dsc_nivel"]

CLASSIFICATION_REFERENCE = {
    "Accuracy": 0.8960,
    "Precision": 0.7086,
    "Recall": 0.3988,
    "F1": 0.5104,
    "ROC-AUC": 0.8898,
}
REGRESSION_REFERENCE = {
    "MAE": 5.8704,
    "RMSE": 13.3167,
    "R2": 0.0854,
}

CLASSIFIER_COMPARISON = pd.DataFrame(
    [
        ["Regresión Logística", 0.8769, 0.6904, 0.2211, 0.3350, 0.8512],
        ["KNN", 0.8720, 0.5655, 0.3759, 0.4516, 0.8005],
        ["Árbol de Decisión", 0.8815, 0.6449, 0.3445, 0.4491, 0.7959],
        ["Random Forest", 0.8835, 0.6476, 0.3709, 0.4717, 0.8586],
    ],
    columns=["Modelo", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"],
)


def detect_csv_format(path: Path) -> tuple[str, str]:
    sample_bytes = path.read_bytes()[:200_000]
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            sample = sample_bytes.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise RuntimeError(f"No se pudo detectar la codificación de {path.name}.")

    try:
        separator = csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        separator = max((",", ";", "\t", "|"), key=sample.count)
    return encoding, separator


def read_period(path: Path) -> pd.DataFrame:
    encoding, separator = detect_csv_format(path)
    required = PREDICTORS + ["Desaprobado"]
    data = pd.read_csv(
        path,
        sep=separator,
        encoding=encoding,
        usecols=required,
        low_memory=False,
    )
    for column in CATEGORICAL_COLUMNS:
        data[column] = data[column].astype("string").str.strip()
    # El notebook comprobó los duplicados sobre las 27 columnas completas y no
    # encontró filas idénticas. No se eliminan duplicados después de seleccionar
    # variables, porque registros distintos pueden compartir estos ocho valores.
    return data.reset_index(drop=True)


def build_preprocessor() -> ColumnTransformer:
    numeric_process = Pipeline(
        steps=[
            ("imputador", SimpleImputer(strategy="median")),
            ("escalador", StandardScaler()),
        ]
    )
    categorical_process = Pipeline(
        steps=[
            ("imputador", SimpleImputer(strategy="most_frequent")),
            ("codificador", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numericas", numeric_process, NUMERIC_COLUMNS),
            ("categoricas", categorical_process, CATEGORICAL_COLUMNS),
        ]
    )


def classification_metrics(y_true, prediction, probability) -> dict[str, float]:
    return {
        "Accuracy": accuracy_score(y_true, prediction),
        "Precision": precision_score(y_true, prediction, zero_division=0),
        "Recall": recall_score(y_true, prediction, zero_division=0),
        "F1": f1_score(y_true, prediction, zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, probability),
    }


def regression_metrics(y_true, prediction) -> dict[str, float]:
    return {
        "MAE": mean_absolute_error(y_true, prediction),
        "RMSE": np.sqrt(mean_squared_error(y_true, prediction)),
        "R2": r2_score(y_true, prediction),
    }


def aggregate_feature_importance(classifier: Pipeline) -> pd.DataFrame:
    preprocessor = classifier.named_steps["preprocesamiento"]
    names = preprocessor.get_feature_names_out()
    values = classifier.named_steps["modelo"].feature_importances_
    rows: list[dict[str, object]] = []
    for transformed_name, importance in zip(names, values, strict=True):
        clean_name = transformed_name.split("__", 1)[-1]
        if clean_name.startswith("gestion_"):
            original_name = "gestion"
        elif clean_name.startswith("dsc_nivel_"):
            original_name = "dsc_nivel"
        else:
            original_name = clean_name
        rows.append({"Variable": original_name, "Importancia": float(importance)})
    return (
        pd.DataFrame(rows)
        .groupby("Variable", as_index=False)["Importancia"]
        .sum()
        .sort_values("Importancia", ascending=True)
    )


def save_plots(
    y_class_2024: pd.Series,
    class_prediction: np.ndarray,
    class_probability: np.ndarray,
    importance: pd.DataFrame,
) -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    matrix = confusion_matrix(y_class_2024, class_prediction)
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    image = ax.imshow(matrix, cmap="Blues")
    for (row, column), value in np.ndenumerate(matrix):
        text_color = "white" if value > matrix.max() / 2 else "#20343a"
        ax.text(
            column,
            row,
            f"{value:,}",
            ha="center",
            va="center",
            fontsize=11,
            color=text_color,
        )
    ax.set_xticks([0, 1], ["No", "Sí"])
    ax.set_yticks([0, 1], ["No", "Sí"])
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Valor real")
    ax.set_title("Matriz de confusión en 2024")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(ASSET_DIR / "matriz_confusion_2024.png", dpi=160)
    plt.close(fig)

    false_positive, true_positive, _ = roc_curve(y_class_2024, class_probability)
    auc = roc_auc_score(y_class_2024, class_probability)
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    ax.plot(false_positive, true_positive, color="#1f6f8b", linewidth=2)
    ax.plot([0, 1], [0, 1], "--", color="#7f8c8d")
    ax.set_xlabel("Tasa de falsos positivos")
    ax.set_ylabel("Tasa de verdaderos positivos")
    ax.set_title(f"Curva ROC en 2024 (AUC = {auc:.4f})")
    fig.tight_layout()
    fig.savefig(ASSET_DIR / "curva_roc_2024.png", dpi=160)
    plt.close(fig)

    comparison = CLASSIFIER_COMPARISON.set_index("Modelo")[["Recall", "F1", "ROC-AUC"]]
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    comparison.plot(kind="bar", ax=ax, color=["#8fb9aa", "#356859", "#1f6f8b"])
    ax.set_ylim(0, 1)
    ax.set_xlabel("")
    ax.set_ylabel("Valor")
    ax.set_title("Comparación de clasificadores en validación 2023")
    ax.tick_params(axis="x", rotation=15)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(ASSET_DIR / "comparacion_clasificadores.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.barh(importance["Variable"], importance["Importancia"], color="#356859")
    ax.set_xlabel("Importancia acumulada")
    ax.set_title("Importancia de variables del Random Forest")
    fig.tight_layout()
    fig.savefig(ASSET_DIR / "importancia_variables.png", dpi=160)
    plt.close(fig)


def train(source_dir: Path) -> dict[str, object]:
    path_2023 = source_dir / "Matriculación y Trayectoria Estudiantil 2023.csv"
    path_2024 = source_dir / "Matriculación y Trayectoria Estudiantil 2024.csv"
    if not path_2023.is_file() or not path_2024.is_file():
        raise FileNotFoundError(
            "No se encontraron los CSV 2023 y 2024 en la carpeta indicada."
        )

    data_2023 = read_period(path_2023)
    data_2024 = read_period(path_2024)
    for data in (data_2023, data_2024):
        data["HAY_DESAPROBACION"] = data["Desaprobado"].gt(0).astype(int)
        data["TASA_DESAPROBACION"] = (
            data["Desaprobado"] / data["TotalEstudiantes"] * 100
        )

    x_2023 = data_2023[PREDICTORS]
    x_2024 = data_2024[PREDICTORS]
    y_class_2023 = data_2023["HAY_DESAPROBACION"]
    y_class_2024 = data_2024["HAY_DESAPROBACION"]
    y_rate_2023 = data_2023["TASA_DESAPROBACION"]
    y_rate_2024 = data_2024["TASA_DESAPROBACION"]

    x_train_class, _, y_train_class, _ = train_test_split(
        x_2023,
        y_class_2023,
        test_size=0.20,
        random_state=42,
        stratify=y_class_2023,
    )
    x_train_rate, _, y_train_rate, _ = train_test_split(
        x_2023,
        y_rate_2023,
        test_size=0.20,
        random_state=42,
    )

    classifier = Pipeline(
        steps=[
            ("preprocesamiento", build_preprocessor()),
            ("modelo", RandomForestClassifier(random_state=42, n_jobs=-1)),
        ]
    )
    regressor = Pipeline(
        steps=[
            ("preprocesamiento", build_preprocessor()),
            ("modelo", LinearRegression()),
        ]
    )
    classifier.fit(x_train_class, y_train_class)
    regressor.fit(x_train_rate, y_train_rate)

    class_prediction = classifier.predict(x_2024)
    class_probability = classifier.predict_proba(x_2024)[:, 1]
    rate_prediction = regressor.predict(x_2024)
    class_metrics = classification_metrics(
        y_class_2024, class_prediction, class_probability
    )
    rate_metrics = regression_metrics(y_rate_2024, rate_prediction)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    classifier_path = MODEL_DIR / "random_forest_desaprobacion.joblib"
    regressor_path = MODEL_DIR / "regresion_tasa_desaprobacion.joblib"
    joblib.dump(classifier, classifier_path, compress=3)
    joblib.dump(regressor, regressor_path, compress=3)

    loaded_classifier = joblib.load(classifier_path)
    loaded_regressor = joblib.load(regressor_path)
    verification_sample = x_2024.iloc[:250]
    classification_equal = np.array_equal(
        classifier.predict(verification_sample),
        loaded_classifier.predict(verification_sample),
    ) and np.allclose(
        classifier.predict_proba(verification_sample),
        loaded_classifier.predict_proba(verification_sample),
    )
    regression_equal = np.allclose(
        regressor.predict(verification_sample),
        loaded_regressor.predict(verification_sample),
    )
    if not classification_equal or not regression_equal:
        raise RuntimeError("La verificación posterior a la serialización no coincidió.")

    importance = aggregate_feature_importance(classifier)
    save_plots(y_class_2024, class_prediction, class_probability, importance)

    categories = {
        column: sorted(data_2023[column].dropna().astype(str).unique().tolist())
        for column in CATEGORICAL_COLUMNS
    }
    metadata: dict[str, object] = {
        "project": "Analizador de Desaprobación Escolar",
        "predictors": PREDICTORS,
        "numeric_columns": NUMERIC_COLUMNS,
        "categorical_columns": CATEGORICAL_COLUMNS,
        "categories_2023": categories,
        "training_period": 2023,
        "temporal_evaluation_period": 2024,
        "training_rows_classification": len(x_train_class),
        "training_rows_regression": len(x_train_rate),
        "evaluation_rows": len(x_2024),
        "classification_metrics_2024": class_metrics,
        "regression_metrics_2024": rate_metrics,
        "reference_metrics_notebook": {
            "classification": CLASSIFICATION_REFERENCE,
            "regression": REGRESSION_REFERENCE,
        },
        "serialization_verified": True,
        "feature_importance": importance.sort_values(
            "Importancia", ascending=False
        ).to_dict(orient="records"),
    }
    (MODEL_DIR / "model_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Entrena y serializa los pipelines finales del proyecto."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_SOURCE_DIR,
        help="Carpeta que contiene los CSV 2023 y 2024.",
    )
    args = parser.parse_args()
    metadata = train(args.data_dir.resolve())
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
