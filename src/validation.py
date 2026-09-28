"""Validaciones compartidas por el formulario y el procesamiento de CSV."""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd


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
COUNT_COLUMNS = ["Discapacidad", "Venezolanos", "Extranjeros"]


def validate_and_prepare(
    data: pd.DataFrame,
    allowed_categories: Mapping[str, list[str]] | None = None,
) -> tuple[pd.DataFrame | None, list[str]]:
    """Valida columnas y valores, y devuelve una copia lista para inferencia."""

    errors: list[str] = []
    missing_columns = [column for column in PREDICTORS if column not in data.columns]
    if missing_columns:
        joined = ", ".join(missing_columns)
        return None, [f"Faltan columnas obligatorias: {joined}."]

    prepared = data[PREDICTORS].copy()

    for column in NUMERIC_COLUMNS:
        original = prepared[column]
        converted = pd.to_numeric(original, errors="coerce")
        invalid = original.notna() & converted.isna()
        if invalid.any():
            errors.append(
                f"La columna {column} contiene {int(invalid.sum())} valor(es) no numérico(s)."
            )
        prepared[column] = converted

    missing_numeric = prepared[NUMERIC_COLUMNS].isna().any(axis=1)
    if missing_numeric.any():
        errors.append(
            f"Hay {int(missing_numeric.sum())} fila(s) con valores numéricos faltantes."
        )

    missing_category = prepared[CATEGORICAL_COLUMNS].isna().any(axis=1)
    if missing_category.any():
        errors.append(
            f"Hay {int(missing_category.sum())} fila(s) con categorías faltantes."
        )

    for column in CATEGORICAL_COLUMNS:
        prepared[column] = prepared[column].astype("string").str.strip()

    if allowed_categories:
        for column in CATEGORICAL_COLUMNS:
            allowed = set(allowed_categories.get(column, []))
            unknown = prepared[column].dropna().loc[
                ~prepared[column].dropna().isin(allowed)
            ]
            if not unknown.empty:
                examples = ", ".join(map(str, unknown.drop_duplicates().head(3)))
                errors.append(
                    f"La columna {column} contiene {len(unknown)} categoría(s) no reconocida(s). "
                    f"Ejemplos: {examples}."
                )

    negative_age = prepared["Edad"].lt(0).fillna(False)
    if negative_age.any():
        errors.append(f"Edad tiene {int(negative_age.sum())} valor(es) menor(es) que 0.")

    invalid_total = prepared["TotalEstudiantes"].le(0).fillna(False)
    if invalid_total.any():
        errors.append(
            "TotalEstudiantes debe ser mayor que 0 "
            f"en {int(invalid_total.sum())} fila(s)."
        )

    for column in COUNT_COLUMNS:
        negative = prepared[column].lt(0).fillna(False)
        if negative.any():
            errors.append(
                f"{column} tiene {int(negative.sum())} valor(es) menor(es) que 0."
            )
        exceeds_total = prepared[column].gt(prepared["TotalEstudiantes"]).fillna(False)
        if exceeds_total.any():
            errors.append(
                f"{column} supera TotalEstudiantes en "
                f"{int(exceeds_total.sum())} fila(s)."
            )

    if errors:
        return None, errors
    return prepared, []


def validate_record(
    record: Mapping[str, object],
    allowed_categories: Mapping[str, list[str]] | None = None,
) -> tuple[pd.DataFrame | None, list[str]]:
    """Valida un registro individual usando las mismas reglas del CSV."""

    return validate_and_prepare(pd.DataFrame([record]), allowed_categories)

