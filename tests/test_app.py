import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import app
import main as project
import train_cardio


class DatasetSelectionTests(unittest.TestCase):
    def test_feature_schemas_map_to_exactly_the_two_dataset_types(self):
        self.assertEqual(
            app.dataset_type_for_features(tuple(project.HEART_FEATURE_NAMES)),
            app.DATASET_TYPES[0],
        )
        self.assertEqual(
            app.dataset_type_for_features(tuple(train_cardio.FEATURE_NAMES)),
            app.DATASET_TYPES[1],
        )
        self.assertIsNone(
            app.dataset_type_for_features(
                tuple(reversed(project.HEART_FEATURE_NAMES))
            )
        )
        self.assertIsNone(app.dataset_type_for_features(("unknown",)))

    def test_compatible_runs_are_filtered_by_model_and_summary_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            results_dir = Path(directory)
            for name, features in (
                ("cleveland", project.HEART_FEATURE_NAMES),
                ("cardio", train_cardio.FEATURE_NAMES),
                ("summary_mismatch", project.HEART_FEATURE_NAMES),
            ):
                run_dir = results_dir / name
                run_dir.mkdir()
                feature_names = np.asarray(features)
                np.savez(
                    run_dir / "global_model.npz",
                    weights=np.zeros(len(features)),
                    bias=np.array([0.0]),
                    feature_names=feature_names,
                    scaler_mean=np.zeros(len(features)),
                    scaler_scale=np.ones(len(features)),
                    imputation_medians=np.zeros(len(features)),
                )
                (run_dir / "model_comparison.csv").write_text(
                    "approach,accuracy\nFedAvg,0.5\n",
                    encoding="utf-8",
                )
                summary_features = (
                    train_cardio.FEATURE_NAMES
                    if name == "summary_mismatch"
                    else features
                )
                (run_dir / "training_summary.json").write_text(
                    json.dumps({
                        "features": list(summary_features),
                        "rows_used": 10,
                    }),
                    encoding="utf-8",
                )

            with patch.object(app, "RESULTS_DIR", results_dir):
                cleveland_runs = app.compatible_run_dirs(app.DATASET_TYPES[0])
                cardio_runs = app.compatible_run_dirs(app.DATASET_TYPES[1])

        self.assertEqual([path.name for path in cleveland_runs], ["cleveland"])
        self.assertEqual([path.name for path in cardio_runs], ["cardio"])

    def test_active_context_rejects_a_run_from_the_other_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            feature_names = np.asarray(train_cardio.FEATURE_NAMES)
            np.savez(
                run_dir / "global_model.npz",
                weights=np.zeros(len(feature_names)),
                bias=np.array([0.0]),
                feature_names=feature_names,
                scaler_mean=np.zeros(len(feature_names)),
                scaler_scale=np.ones(len(feature_names)),
                imputation_medians=np.zeros(len(feature_names)),
            )
            (run_dir / "training_summary.json").write_text(
                json.dumps({
                    "features": list(train_cardio.FEATURE_NAMES),
                    "rows_used": 68_615,
                }),
                encoding="utf-8",
            )
            with (
                patch.object(app, "RUN_DIR", run_dir),
                patch.object(app, "MODEL_PATH", run_dir / "global_model.npz"),
                patch.object(app, "SELECTED_DATASET", app.DATASET_TYPES[0]),
            ):
                with self.assertRaisesRegex(
                    app.DashboardDataError,
                    "not compatible",
                ):
                    app.load_active_run_context()

    def test_dataset_input_schemas_have_correct_feature_counts(self):
        self.assertEqual(len(app.DATASET_FEATURES[app.DATASET_TYPES[0]]), 13)
        self.assertEqual(len(app.DATASET_FEATURES[app.DATASET_TYPES[1]]), 11)
        for feature in project.HEART_FEATURE_NAMES:
            self.assertTrue(app.feature_spec(feature)[0])
        for feature in train_cardio.FEATURE_NAMES:
            self.assertTrue(app.feature_spec(feature)[0])


if __name__ == "__main__":
    unittest.main()
