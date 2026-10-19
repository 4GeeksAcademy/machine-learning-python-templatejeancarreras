import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from PIL import Image

from src.image_classifier import (
    ROOT, discover_images, extract_images, image_label, prepare_directories,
    prepare_training_run, read_training_history, select_loading_mode, split_records,
    train_models,
)


def create_images(directory, count=5):
    for label, color in (("cat", (240, 40, 30)), ("dog", (30, 220, 80))):
        for index in range(count):
            Image.new("RGB", (25 + index, 40), color).save(directory / f"{label}.{index}.jpg")


class DatasetTests(unittest.TestCase):
    def test_filename_labels(self):
        self.assertEqual(image_label("train/cat.12.jpg"), "cat")
        self.assertEqual(image_label("dog.3.jpg"), "dog")
        self.assertIsNone(image_label("123.jpg"))

    def test_reproducible_disjoint_stratified_splits(self):
        records = [(Path(f"{label}.{index}.jpg"), label)
                   for label in ("cat", "dog") for index in range(100)]
        splits = split_records(records)
        self.assertEqual(splits, split_records(records))
        self.assertNotEqual(splits, split_records(records, seed=7))
        self.assertEqual([len(rows) for rows in splits.values()], [140, 30, 30])
        sets = [{path for path, _ in rows} for rows in splits.values()]
        self.assertEqual(len(set.union(*sets)), 200)
        self.assertFalse(sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
        for rows in splits.values():
            self.assertEqual(sum(label == "cat" for _, label in rows), len(rows) // 2)

    def test_insufficient_data_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "al menos 5"):
            split_records([(Path("cat.0.jpg"), "cat")])

    def test_memory_selection(self):
        self.assertEqual(select_loading_mode("auto", 25000, 8 * 1024**3), "directory")
        self.assertEqual(select_loading_mode("auto", 25000, 16 * 1024**3), "memory")
        self.assertEqual(select_loading_mode("auto", 25000, 12 * 1024**3), "directory")
        self.assertEqual(select_loading_mode("directory", 25000, 32 * 1024**3), "directory")
        with self.assertRaises(ValueError):
            select_loading_mode("memory", 25000, 8 * 1024**3)

    def test_corrupt_images_and_links(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            create_images(directory)
            (directory / "cat.corrupt.jpg").write_bytes(b"not an image")
            Image.new("RGB", (10, 10)).save(directory / "123.jpg")
            records, rejected = discover_images(directory)
            self.assertEqual(len(records), 10)
            self.assertEqual(len(rejected), 1)
            root = prepare_directories(split_records(records), directory / "split")
            links = list(root.glob("*/*/*.jpg"))
            self.assertEqual(len(links), 10)
            self.assertTrue(all(path.is_symlink() and path.exists() for path in links))

    def test_archive_does_not_trust_member_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            archive = directory / "train.zip"
            with zipfile.ZipFile(archive, "w") as compressed:
                compressed.writestr("../../cat.0.jpg", b"example")
                compressed.writestr("train/dog.0.jpg", b"example")
                compressed.writestr("../../outside.txt", b"ignored")
            destination = directory / "images"
            extract_images(archive, destination)
            self.assertEqual({path.name for path in destination.iterdir()}, {"cat.0.jpg", "dog.0.jpg"})
            self.assertFalse((directory / "outside.txt").exists())
            with self.assertRaisesRegex(ValueError, "duplicado"):
                extract_images(archive, destination)

    def test_both_loading_modes(self):
        os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")
        os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
        from src.image_classifier import load_arrays, make_generators

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            create_images(directory)
            records, _ = discover_images(directory)
            splits = split_records(records)
            split_root = prepare_directories(splits, directory / "split")
            arrays = load_arrays(splits)
            self.assertEqual(arrays["test"][0].dtype.name, "uint8")
            for source in (None, arrays):
                generators = make_generators(split_root, source, 4, augment=True)
                images, labels = generators["test"][0]
                self.assertEqual(images.shape, (2, 200, 200, 3))
                self.assertEqual(images.dtype.name, "uint8")
                self.assertEqual(labels.argmax(axis=1).tolist(), [0, 1])
                self.assertGreater(float(images.max()), 1)
                self.assertLessEqual(float(images.max()), 255)
                self.assertTrue(generators["train"].shuffle)
                self.assertFalse(generators["validation"].shuffle)
                self.assertFalse(generators["test"].shuffle)

    def test_resume_configuration_guards(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            create_images(directory)
            records, _ = discover_images(directory)
            splits = split_records(records)
            with self.assertRaisesRegex(ValueError, "No hay un entrenamiento"):
                prepare_training_run(directory, splits, 4, 1, 2, resume=True)
            prepare_training_run(directory, splits, 4, 1, 2)
            prepare_training_run(directory, splits, 4, 1, 2, resume=True)
            with self.assertRaisesRegex(ValueError, "Ya hay un entrenamiento"):
                prepare_training_run(directory, splits, 4, 1, 2)
            with self.assertRaisesRegex(ValueError, "mantener el dataset"):
                prepare_training_run(directory, splits, 8, 1, 2, resume=True)
            Image.new("RGB", (70, 70)).save(directory / "cat.0.jpg")
            with self.assertRaisesRegex(ValueError, "mantener el dataset"):
                prepare_training_run(directory, splits, 4, 1, 2, resume=True)

    def test_resumed_history_replaces_repeated_epochs(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "history.csv"
            self.assertEqual(read_training_history(path), {})
            path.write_text("epoch,loss,val_loss\n0,1.0,1.2\n1,0.8,1.1\n1,0.7,0.9\n")
            self.assertEqual(read_training_history(path),
                             {"loss": [1.0, 0.7], "val_loss": [1.2, 0.9]})

    def test_training_resume_restores_state_and_skips_completed_stage(self):
        os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")
        os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
        import numpy as np
        import tensorflow as tf
        from tensorflow.keras.callbacks import BackupAndRestore

        saved, resumed_epochs = {}, []

        def tiny_model(learning_rate):
            model = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(200, 200, 3)),
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Rescaling(1 / 255),
                tf.keras.layers.Dense(2, activation="softmax"),
            ])
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
                          loss="categorical_crossentropy", metrics=["accuracy"])
            return model

        class InterruptAfterBackup(BackupAndRestore):
            def on_epoch_end(self, epoch, logs=None):
                super().on_epoch_end(epoch, logs)
                if Path(self.backup_dir).name == "optimized":
                    saved["weights"] = self.model.get_weights()
                    saved["optimizer"] = [value.numpy().copy() for value in self.model.optimizer.variables]
                    raise InterruptedError("Interrupcion simulada despues de la primera epoca")

        class ObserveRestoration(BackupAndRestore):
            def on_train_begin(self, logs=None):
                super().on_train_begin(logs)
                for actual, expected in zip(self.model.get_weights(), saved["weights"]):
                    np.testing.assert_allclose(actual, expected)
                for actual, expected in zip(self.model.optimizer.variables, saved["optimizer"]):
                    np.testing.assert_allclose(actual.numpy(), expected)

            def on_epoch_begin(self, epoch, logs=None):
                resumed_epochs.append(epoch)
                self.model.set_weights([np.zeros_like(value) for value in self.model.get_weights()])

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            raw = directory / "raw"
            raw.mkdir()
            create_images(raw)
            records, _ = discover_images(raw)
            splits = split_records(records)
            split_root = prepare_directories(splits, directory / "split")
            output = directory / "models"
            output.mkdir()
            prepare_training_run(output, splits, 6, 1, 2)
            with patch("src.image_classifier.build_model", side_effect=tiny_model), \
                 patch("tensorflow.keras.callbacks.BackupAndRestore", InterruptAfterBackup):
                with self.assertRaisesRegex(InterruptedError, "Interrupcion simulada"):
                    train_models(split_root, None, output, 6, 1, 2)
            self.assertTrue((output / "baseline_complete.json").exists())
            self.assertFalse((output / "optimized_complete.json").exists())
            previous_checkpoint = (output / "optimized.keras").read_bytes()
            best_loss = min(read_training_history(output / "optimized_history.csv")["val_loss"])
            prepare_training_run(output, splits, 6, 1, 2, resume=True)
            with patch("src.image_classifier.build_model", side_effect=tiny_model) as builder, \
                 patch("tensorflow.keras.callbacks.BackupAndRestore", ObserveRestoration):
                histories, losses, winner = train_models(split_root, None, output, 6, 1, 2, resume=True)
            self.assertEqual(builder.call_count, 1)
            self.assertEqual(resumed_epochs, [1])
            self.assertEqual(len(histories["baseline"]["loss"]), 1)
            self.assertEqual(len(histories["optimized"]["loss"]), 2)
            self.assertLessEqual(losses["optimized"], best_loss + 1e-6)
            self.assertEqual((output / "optimized.keras").read_bytes(), previous_checkpoint)
            self.assertIn(winner, ("baseline", "optimized"))
            with patch("src.image_classifier.build_model") as builder:
                repeated = train_models(split_root, None, output, 6, 1, 2, resume=True)
            builder.assert_not_called()
            self.assertEqual(repeated, (histories, losses, winner))

    @unittest.skipUnless(os.environ.get("RUN_TRAINING_SMOKE") == "1", "Prueba de entrenamiento opcional")
    def test_training_round_trip(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            raw = directory / "raw"
            raw.mkdir()
            create_images(raw)
            output = directory / "models"
            result = subprocess.run(
                [sys.executable, str(ROOT / "src/image_classifier.py"),
                 "--data-dir", str(raw), "--output", str(output),
                 "--split-dir", str(directory / "split"), "--mode", "directory",
                 "--epochs", "1", "--baseline-epochs", "1", "--batch-size", "6"],
                cwd=ROOT, capture_output=True, text=True, timeout=300,
                env={**os.environ, "MPLBACKEND": "Agg"},
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            for name in ("baseline.keras", "optimized.keras", "best_model.keras",
                         "learning_curves.png", "confusion_matrix.png", "examples_cat.png",
                         "examples_dog.png", "split_manifest.csv", "dataset_summary.json"):
                self.assertTrue((output / name).is_file(), name)
            metrics = json.loads((output / "metrics.json").read_text())
            self.assertIn(metrics["selected_model"], ("baseline", "optimized"))
            self.assertEqual(metrics["test"]["macro avg"]["support"], 2)
            with (output / "test_predictions.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 2)
            self.assertEqual({row["actual"] for row in rows}, {"cat", "dog"})
            resumed = subprocess.run(
                [sys.executable, str(ROOT / "src/image_classifier.py"),
                 "--data-dir", str(raw), "--output", str(output),
                 "--split-dir", str(directory / "split"), "--mode", "directory",
                 "--epochs", "1", "--baseline-epochs", "1", "--batch-size", "6", "--resume"],
                cwd=ROOT, capture_output=True, text=True, timeout=120,
                env={**os.environ, "MPLBACKEND": "Agg"},
            )
            self.assertEqual(resumed.returncode, 0, resumed.stdout + resumed.stderr)
            self.assertIn("Etapa baseline ya terminada", resumed.stdout)
            self.assertIn("Etapa optimized ya terminada", resumed.stdout)
            self.assertEqual(json.loads((output / "metrics.json").read_text())["history"], metrics["history"])


if __name__ == "__main__":
    unittest.main()