"""Pruebas del aislamiento de test y de los artefactos del tutorial."""

import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

from src.app import FEATURES, fit_models, load_data, save_results


class HousingWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        values, _ = make_blobs(
            n_samples=300, centers=6, n_features=3, cluster_std=0.3, random_state=42
        )
        cls.data = pd.DataFrame(values, columns=FEATURES).rename_axis("row_id")
        cls.train, cls.test = train_test_split(cls.data, test_size=0.2, random_state=42)
        cls.clustering, cls.search = fit_models(cls.train)

    def test_split_and_training_only_scaling(self):
        self.assertTrue(set(self.train.index).isdisjoint(self.test.index))
        scaler = self.clustering.named_steps["scaler"]
        self.assertEqual(scaler.n_samples_seen_, len(self.train))
        np.testing.assert_allclose(scaler.mean_, self.train.mean())
        centers = self.clustering.named_steps["kmeans"].cluster_centers_.copy()
        self.clustering.predict(self.test)
        np.testing.assert_array_equal(
            centers, self.clustering.named_steps["kmeans"].cluster_centers_
        )

    def test_six_clusters_and_classifier_features(self):
        self.assertEqual(set(self.clustering.predict(self.train)), set(range(6)))
        self.assertEqual(list(self.search.best_estimator_.feature_names_in_), FEATURES)
        self.assertGreater(self.search.score(self.test, self.clustering.predict(self.test)), 0.8)

    def test_exports_and_model_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            metrics = save_results(
                self.data, self.train, self.test, self.clustering, self.search, output
            )
            for filename, expected in [
                ("kmeans_pipeline.joblib", self.clustering.predict(self.test)),
                ("decision_tree.joblib", self.search.predict(self.test)),
            ]:
                restored = joblib.load(output / "models" / filename)
                np.testing.assert_array_equal(restored.predict(self.test), expected)
            exported = pd.read_csv(output / "data/processed/housing_test_clusters.csv")
            self.assertEqual(exported["row_id"].tolist(), self.test.index.tolist())
            np.testing.assert_array_equal(exported["cluster"], self.clustering.predict(self.test))
            self.assertEqual(metrics["test_rows"], len(self.test))
            for filename in ["clusters.png", "income_by_cluster.png", "confusion_matrix.png"]:
                self.assertGreater((output / "reports" / filename).stat().st_size, 1000)

    def test_load_data_selects_only_required_features(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "housing.csv"
            self.data.assign(unused=999).to_csv(path, index=False)
            loaded = load_data(path)
            self.assertEqual(list(loaded.columns), FEATURES)
            pd.testing.assert_frame_equal(loaded, self.data)

    def test_load_data_rejects_invalid_input(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.csv"
            invalid_frames = [
                self.data.drop(columns="MedInc"),
                self.data.assign(MedInc=np.nan),
                self.data.assign(MedInc=np.inf),
                self.data.assign(MedInc="not numeric"),
                self.data.head(5),
            ]
            for frame in invalid_frames:
                with self.subTest(columns=list(frame.columns), rows=len(frame)):
                    frame.to_csv(path, index=False)
                    with self.assertRaises(ValueError):
                        load_data(path)


if __name__ == "__main__":
    unittest.main()