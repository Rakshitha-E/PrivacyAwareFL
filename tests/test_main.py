import tempfile
import unittest
from pathlib import Path

import numpy as np

import main


class ZeroNoise:
    def normal(self, loc, scale, size):
        return np.zeros(size)


class FederatedLearningTests(unittest.TestCase):
    def test_non_iid_partitions_cover_every_row_for_supported_sizes(self):
        labels = np.tile([0, 1], 60)
        features = np.arange(len(labels)).reshape(-1, 1)

        for count in (2, 4, 6, 8):
            with self.subTest(client_count=count):
                clients = main.split_clients_non_iid(
                    features, labels, num_clients=count, seed=7
                )
                assigned = np.concatenate([
                    client_features[:, 0]
                    for client_features, _ in clients.values()
                ])
                self.assertEqual(len(assigned), len(labels))
                np.testing.assert_array_equal(
                    np.sort(assigned), np.arange(len(labels))
                )
                proportions = [
                    client_labels.mean()
                    for _, client_labels in clients.values()
                ]
                self.assertGreater(proportions[-1], proportions[0])

    def test_fedavg_uses_sample_size_weights(self):
        weights, bias = main.fedavg([
            (np.array([2.0]), 1.0, 1),
            (np.array([8.0]), 3.0, 3)
        ])
        np.testing.assert_allclose(weights, [6.5])
        self.assertAlmostEqual(bias, 2.5)

    def test_dp_aggregation_clips_complete_update_and_reports_sensitivity(self):
        weights, bias, sample_weights, sensitivity, noise_std = (
            main.aggregate_dp_client_updates(
                [
                    (1, np.array([3.0, 4.0]), 0.0, 1),
                    (2, np.array([0.0, 0.0]), 0.0, 1)
                ],
                np.zeros(2),
                0.0,
                clipping_norm=1.0,
                noise_multiplier=1.0,
                rng=ZeroNoise()
            )
        )
        np.testing.assert_allclose(weights, [0.3, 0.4])
        self.assertAlmostEqual(bias, 0.0)
        self.assertEqual(sample_weights, {1: 0.5, 2: 0.5})
        self.assertAlmostEqual(sensitivity, 1.0)
        self.assertAlmostEqual(noise_std, 1.0)

    def test_federated_aggregation_skips_dropout_and_keeps_training(self):
        features = np.array([[-1.0], [-0.5], [0.5], [1.0]])
        labels = np.array([0, 0, 1, 1])
        clients = {
            1: (features[:2], labels[:2]),
            2: (features[2:], labels[2:])
        }
        _, _, history = main.train_federated_model(
            clients,
            features,
            labels,
            num_rounds=2,
            dropout_round=2,
            dropout_client_id=2,
            verbose=False
        )
        self.assertEqual([row["active_clients"] for row in history], [2, 1])

    def test_rdp_epsilon_decreases_as_noise_increases(self):
        orders = np.array([2.0, 4.0, 8.0, 16.0])
        low_noise_rdp = orders / (2 * 1.0**2)
        high_noise_rdp = orders / (2 * 3.0**2)
        self.assertLess(
            main.rdp_to_epsilon(high_noise_rdp, orders, 1e-5),
            main.rdp_to_epsilon(low_noise_rdp, orders, 1e-5)
        )

    def test_dp_history_records_calibration_and_accounting(self):
        features = np.array([[-1.0], [-0.5], [0.5], [1.0]])
        labels = np.array([0, 0, 1, 1])
        clients = {
            1: (features[:2], labels[:2]),
            2: (features[2:], labels[2:])
        }
        _, _, history, privacy = main.train_dp_federated_model(
            clients,
            features,
            labels,
            num_rounds=1,
            clipping_norm=1.0,
            noise_multiplier=2.0,
            dropout_round=None,
            dropout_client_id=None,
            verbose=False
        )
        self.assertAlmostEqual(history[0]["sensitivity"], 1.0)
        self.assertAlmostEqual(history[0]["noise_std"], 2.0)
        self.assertAlmostEqual(history[0]["noise_multiplier"], 2.0)
        self.assertAlmostEqual(history[0]["epsilon"], privacy["epsilon"])

    def test_saved_model_imputes_missing_raw_features(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "model.npz"
            main.save_model_artifact(
                model_path,
                np.zeros(len(main.HEART_FEATURE_NAMES)),
                0.0,
                np.zeros(len(main.HEART_FEATURE_NAMES)),
                np.ones(len(main.HEART_FEATURE_NAMES)),
                np.arange(len(main.HEART_FEATURE_NAMES), dtype=float)
            )
            raw = np.full(len(main.HEART_FEATURE_NAMES), np.nan)
            result = main.predict_proba_from_raw(raw, model_path=model_path)
            self.assertEqual(result, 0.5)
            self.assertEqual(main.predict_from_raw(raw, model_path=model_path), 1)
            batch = main.predict_from_raw(
                np.vstack((raw, raw)), model_path=model_path
            )
            np.testing.assert_array_equal(batch, [1, 1])

    def test_saved_model_uses_its_own_feature_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "cardio_model.npz"
            np.savez(
                model_path,
                weights=np.array([1.0, -1.0]),
                bias=np.array([0.0]),
                feature_names=np.array(["age_years", "weight_kg"]),
                scaler_mean=np.zeros(2),
                scaler_scale=np.ones(2),
                imputation_medians=np.zeros(2),
            )

            probability = main.predict_proba_from_raw(
                np.array([1.0, 0.0]),
                model_path=model_path,
            )

            self.assertAlmostEqual(probability, main.sigmoid(1.0))
            with self.assertRaisesRegex(ValueError, "Expected records with 2 features"):
                main.predict_proba_from_raw(
                    np.array([1.0]),
                    model_path=model_path,
                )

    def test_client_scalability_plot_requires_four_client_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "scalability.png"
            metrics = {
                "accuracy": [0.72, 0.73],
                "precision": [0.71, 0.72],
                "recall": [0.70, 0.71],
                "f1_score": [0.705, 0.715],
            }
            data = main.pd.DataFrame({"num_clients": [2, 4], **metrics})

            main.plot_client_scalability(data, output_path)

            self.assertTrue(output_path.is_file())
            self.assertGreater(output_path.stat().st_size, 0)
            with self.assertRaisesRegex(ValueError, "must contain exactly one 4-client row"):
                main.plot_client_scalability(
                    data.loc[data["num_clients"] == 2],
                    output_path,
                )


if __name__ == "__main__":
    unittest.main()
