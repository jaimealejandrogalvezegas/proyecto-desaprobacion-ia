"""Pruebas mínimas de carga e inferencia de los modelos."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.predict import load_models, predict_records
from src.validation import validate_and_prepare


class ModelSmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.classifier, cls.regressor, cls.metadata = load_models()
        categories = cls.metadata["categories_2023"]
        cls.valid_row = pd.DataFrame(
            [
                {
                    "Edad": 9,
                    "TotalEstudiantes": 20,
                    "Discapacidad": 1,
                    "Venezolanos": 0,
                    "Extranjeros": 0,
                    "gestion": categories["gestion"][0],
                    "dsc_nivel": categories["dsc_nivel"][0],
                }
            ]
        )

    def test_models_load_and_accept_one_row(self):
        prepared, errors = validate_and_prepare(
            self.valid_row, self.metadata["categories_2023"]
        )
        self.assertEqual(errors, [])
        classification = self.classifier.predict(prepared)
        probability = self.classifier.predict_proba(prepared)[:, 1]
        rate = self.regressor.predict(prepared)

        self.assertIn(int(classification[0]), (0, 1))
        self.assertGreaterEqual(float(probability[0]), 0.0)
        self.assertLessEqual(float(probability[0]), 1.0)
        self.assertTrue(np.issubdtype(rate.dtype, np.number))

    def test_batch_output_can_be_downloaded_as_csv(self):
        output = predict_records(
            self.valid_row,
            classifier=self.classifier,
            regressor=self.regressor,
        )
        self.assertIn("PRED_HAY_DESAPROBACION", output.columns)
        self.assertIn("PROB_DESAPROBACION", output.columns)
        self.assertIn("TASA_DESAPROBACION_ESTIMADA", output.columns)
        csv_bytes = output.to_csv(index=False).encode("utf-8-sig")
        self.assertGreater(len(csv_bytes), 20)


if __name__ == "__main__":
    unittest.main()

