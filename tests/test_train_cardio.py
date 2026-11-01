import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

import train_cardio


class CardioArchiveTests(unittest.TestCase):
    def test_loader_filters_implausible_measurements_and_maps_features(self):
        frame = pd.DataFrame([
            [1, 18000, 1, 165, 65, 120, 80, 1, 1, 0, 0, 1, 0],
            [2, 19000, 2, 170, 72, 140, 90, 2, 1, 0, 0, 1, 1],
            [3, 20000, 1, 160, 60, 130, 80, 1, 2, 1, 0, 0, 1],
            [4, 21000, 2, 170, 75, 16020, 11000, 1, 1, 0, 0, 1, 0],
        ],
            columns=train_cardio.EXPECTED_COLUMNS,
        )
        with tempfile.TemporaryDirectory() as directory:
            archive_path = Path(directory) / "cardio.zip"
            csv_data = frame.to_csv(index=False, sep=";")
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("cardio_train.csv", csv_data)

            cleaned, valid_rows = train_cardio.load_cardio_archive(archive_path)

        self.assertEqual(len(cleaned), 3)
        self.assertEqual(int((~valid_rows).sum()), 1)
        self.assertIn("age_years", cleaned.columns)
        self.assertIn("height_cm", cleaned.columns)
        self.assertIn("diastolic_bp", cleaned.columns)
        self.assertEqual(set(cleaned["cardio"]), {0, 1})
        self.assertAlmostEqual(cleaned.iloc[0]["age_years"], 18000 / 365.25)

    def test_prepare_data_keeps_split_disjoint_and_fits_scaler_on_training_rows(self):
        rows = []
        for index in range(12):
            rows.append([
                index + 1,
                18000 + index * 100,
                1 + index % 2,
                160 + index,
                60 + index,
                120 + index,
                75 + index,
                1,
                1,
                0,
                0,
                1,
                index % 2,
            ])
        frame = pd.DataFrame(rows, columns=train_cardio.EXPECTED_COLUMNS)
        with tempfile.TemporaryDirectory() as directory:
            archive_path = Path(directory) / "cardio.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr(
                    "cardio_train.csv",
                    frame.to_csv(index=False, sep=";"),
                )
            prepared = train_cardio.prepare_cardio_data(archive_path)

        train_indices = prepared["train_row_indices"]
        test_indices = prepared["test_row_indices"]
        self.assertEqual(len(set(train_indices) & set(test_indices)), 0)
        self.assertEqual(len(train_indices) + len(test_indices), len(frame))
        raw_train = prepared["frame"].iloc[train_indices].loc[
            :, list(train_cardio.FEATURE_NAMES)
        ].to_numpy(dtype=float)
        np.testing.assert_allclose(
            prepared["scaler"].mean_,
            raw_train.mean(axis=0),
        )
        np.testing.assert_allclose(
            prepared["X_train"].mean(axis=0),
            np.zeros(len(train_cardio.FEATURE_NAMES)),
            atol=1e-12,
        )


if __name__ == "__main__":
    unittest.main()
