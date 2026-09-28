"""Carga de pipelines y funciones de inferencia."""

from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path

import joblib
import pandas as pd

from src.validation import PREDICTORS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models"
CLASSIFIER_PATH = MODEL_DIR / "random_forest_desaprobacion.joblib"
REGRESSOR_PATH = MODEL_DIR / "regresion_tasa_desaprobacion.joblib"
METADATA_PATH = MODEL_DIR / "model_metadata.json"


@lru_cache(maxsize=1)
def load_models():
    """Carga una sola vez los dos pipelines completos y sus metadatos."""

    classifier = joblib.load(CLASSIFIER_PATH)
    regressor = joblib.load(REGRESSOR_PATH)
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    return classifier, regressor, metadata


def predict_records(
    data: pd.DataFrame,
    classifier=None,
    regressor=None,
) -> pd.DataFrame:
    """Agrega clasificación, probabilidad y tasa estimada a los registros."""

    if classifier is None or regressor is None:
        classifier, regressor, _ = load_models()

    features = data[PREDICTORS]
    result = data.copy()
    result["PRED_HAY_DESAPROBACION"] = classifier.predict(features).astype(int)
    result["PROB_DESAPROBACION"] = classifier.predict_proba(features)[:, 1]
    result["TASA_DESAPROBACION_ESTIMADA"] = regressor.predict(features)
    return result

