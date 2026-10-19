"""Clasificador reproducible de gatos y perros con EfficientNetB0."""

import argparse
import csv
import hashlib
import json
import random
import shutil
import tempfile
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CLASSES = ("cat", "dog")
IMAGE_SIZE = (200, 200)
SEED = 42


def image_label(path):
    """Obtiene la clase de los nombres originales de Asirra."""
    label = Path(path).name.split(".")[0].lower()
    return label if label in CLASSES else None


def extract_images(archive, destination):
    """Extrae solo JPG etiquetados, sin confiar en las rutas del ZIP."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as compressed:
        for member in compressed.infolist():
            name = Path(member.filename).name
            if image_label(name) is None or Path(name).suffix.lower() not in {".jpg", ".jpeg"}:
                continue
            target = destination / name
            if target.exists():
                raise ValueError(f"Archivo duplicado al extraer: {target}")
            with compressed.open(member) as source, target.open("xb") as output:
                shutil.copyfileobj(source, output)


def discover_images(directory):
    """Localiza JPG etiquetados y descarta imagenes corruptas."""
    from PIL import Image

    records, rejected = [], []
    seen = set()
    for path in sorted(Path(directory).rglob("*")):
        label = image_label(path)
        if not path.is_file() or path.suffix.lower() not in {".jpg", ".jpeg"} or label is None:
            continue
        if path.name in seen:
            raise ValueError(f"Nombre duplicado en el dataset: {path.name}")
        seen.add(path.name)
        try:
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                image.convert("RGB").load()
        except (OSError, ValueError, Image.DecompressionBombError) as error:
            rejected.append({"path": str(path), "error": str(error)})
            continue
        records.append((path.resolve(), label))
    if not records:
        raise ValueError("No hay imagenes cat.*.jpg o dog.*.jpg validas en la carpeta indicada.")
    return records, rejected


def split_records(records, seed=SEED):
    """Particion estratificada 70/15/15, reservando test para el final."""
    partitions = {"train": [], "validation": [], "test": []}
    generator = random.Random(seed)
    for label in CLASSES:
        group = sorted((record for record in records if record[1] == label), key=lambda record: str(record[0]))
        if len(group) < 5:
            raise ValueError(f"Se necesitan al menos 5 imagenes validas de {label}.")
        generator.shuffle(group)
        held_out = max(1, round(len(group) * 0.15))
        partitions["test"].extend(group[:held_out])
        partitions["validation"].extend(group[held_out : 2 * held_out])
        partitions["train"].extend(group[2 * held_out :])
    return partitions


def prepare_directories(partitions, destination):
    """Crea directorios por clase mediante enlaces, sin duplicar las fotos."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    split_root = Path(tempfile.mkdtemp(prefix="split-", dir=destination))
    for split, records in partitions.items():
        for label in CLASSES:
            (split_root / split / label).mkdir(parents=True)
        for path, label in records:
            (split_root / split / label / path.name).symlink_to(path.resolve())
    return split_root


def available_memory():
    """Tiene en cuenta los limites del contenedor, no solo la RAM del host."""
    import psutil

    memory = psutil.virtual_memory()
    available = min(memory.total, memory.available)
    for limit_path, used_path in (
        ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.current"),
        ("/sys/fs/cgroup/memory/memory.limit_in_bytes", "/sys/fs/cgroup/memory/memory.usage_in_bytes"),
    ):
        try:
            limit = int(Path(limit_path).read_text().strip())
            used = int(Path(used_path).read_text().strip())
            available = min(available, max(0, limit - used))
        except (OSError, ValueError):
            continue
    return available


def select_loading_mode(requested, count, memory_bytes):
    required = count * IMAGE_SIZE[0] * IMAGE_SIZE[1] * 3 * 2 + 3 * 1024**3
    enough = memory_bytes > 12 * 1024**3 and required < memory_bytes
    if requested == "memory" and not enough:
        raise ValueError("RAM insuficiente para --mode memory; utiliza --mode directory.")
    return "memory" if requested != "directory" and enough else "directory"


def plot_examples(records, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    for label in CLASSES:
        figure, axes = plt.subplots(3, 3, figsize=(9, 9))
        examples = [path for path, category in records if category == label][:9]
        for axis in axes.flat:
            axis.axis("off")
        for axis, path in zip(axes.flat, examples):
            with Image.open(path) as image:
                axis.imshow(image.convert("RGB"))
                axis.set_title(f"{path.name}\n{image.width} x {image.height}")
        figure.suptitle(f"Primeras nueve imagenes: {label}")
        figure.tight_layout()
        figure.savefig(Path(output) / f"examples_{label}.png")
        plt.close(figure)


def load_arrays(partitions):
    import numpy as np
    from tensorflow.keras.utils import img_to_array, load_img

    arrays = {}
    for split, records in partitions.items():
        images = np.empty((len(records), *IMAGE_SIZE, 3), dtype=np.uint8)
        labels = np.zeros((len(records), 2), dtype=np.float32)
        for index, (path, label) in enumerate(records):
            image = load_img(path, target_size=IMAGE_SIZE, color_mode="rgb")
            try:
                images[index] = img_to_array(image, dtype="uint8")
            finally:
                image.close()
            labels[index, CLASSES.index(label)] = 1
        arrays[split] = (images, labels)
    return arrays


def make_generators(split_root, arrays, batch_size, augment=False):
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    generators = {}
    for split in ("train", "validation", "test"):
        settings = {}
        if split == "train" and augment:
            settings = {"rotation_range": 15, "width_shift_range": 0.1,
                        "height_shift_range": 0.1, "zoom_range": 0.15,
                        "horizontal_flip": True, "fill_mode": "nearest"}
        processor = ImageDataGenerator(dtype="uint8", **settings)
        options = {"batch_size": batch_size, "shuffle": split == "train", "seed": SEED}
        if arrays is None:
            generators[split] = processor.flow_from_directory(
                str(split_root / split), target_size=IMAGE_SIZE, color_mode="rgb",
                classes=list(CLASSES), class_mode="categorical", **options,
            )
        else:
            images, labels = arrays[split]
            generators[split] = processor.flow(images, labels, **options)
    return generators


def build_model(learning_rate=1e-3):
    from tensorflow.keras.applications import EfficientNetB0
    from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.optimizers import Adam

    model = Sequential([
        EfficientNetB0(include_top=False, weights=None, input_shape=(*IMAGE_SIZE, 3)),
        GlobalAveragePooling2D(),
        Dense(128, activation="relu"),
        Dense(2, activation="softmax"),
    ])
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def write_json_atomic(path, value):
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def prepare_training_run(output, partitions, batch_size, baseline_epochs, epochs, resume=False):
    fingerprint = hashlib.sha256()
    for split, records in partitions.items():
        for path, label in records:
            info = path.stat()
            record = [split, str(path.resolve()), label, info.st_size, info.st_mtime_ns]
            fingerprint.update(json.dumps(record).encode())
    configuration = {
        "version": 1, "dataset_fingerprint": fingerprint.hexdigest(),
        "batch_size": batch_size, "baseline_epochs": baseline_epochs, "epochs": epochs,
        "seed": SEED, "image_size": list(IMAGE_SIZE), "classes": list(CLASSES),
    }
    path = output / "training_config.json"
    if resume:
        if not path.exists():
            raise ValueError("No hay un entrenamiento reanudable en --output. Inicia uno sin --resume.")
        if json.loads(path.read_text()) != configuration:
            raise ValueError("Para reanudar debes mantener el dataset, batch-size y limites de epocas originales.")
    else:
        if path.exists() or any(output.glob("*.keras")) or (output / "backup").exists():
            raise ValueError("Ya hay un entrenamiento en --output. Utiliza --resume o elige otra carpeta.")
        write_json_atomic(path, configuration)


def read_training_history(path):
    if not path.exists():
        return {}
    with path.open(newline="") as stream:
        rows = {int(row["epoch"]): row for row in csv.DictReader(stream)}
    if not rows:
        return {}
    return {key: [float(rows[epoch][key]) for epoch in sorted(rows)]
            for key in next(iter(rows.values())) if key != "epoch"}


def train_models(split_root, arrays, output, batch_size, baseline_epochs, epochs, resume=False):
    import tensorflow as tf
    from tensorflow.keras.callbacks import (
        BackupAndRestore, CSVLogger, EarlyStopping, LambdaCallback,
        ModelCheckpoint, ReduceLROnPlateau,
    )
    from tensorflow.keras.models import load_model

    histories, validation_losses = {}, {}
    for name, augment, learning_rate, epoch_count in (
        ("baseline", False, 1e-3, baseline_epochs),
        ("optimized", True, 1e-4, epochs),
    ):
        checkpoint = output / f"{name}.keras"
        completed = output / f"{name}_complete.json"
        history_path = output / f"{name}_history.csv"
        if resume and completed.exists():
            if not checkpoint.exists():
                raise ValueError(f"Falta el checkpoint de la etapa terminada: {checkpoint}")
            saved = json.loads(completed.read_text())
            histories[name] = saved["history"]
            validation_losses[name] = saved["validation_loss"]
            print(f"Etapa {name} ya terminada; se conserva sin volver a entrenar.")
            continue
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(SEED)
        generators = make_generators(split_root, arrays, batch_size, augment)
        threshold = None
        if resume and checkpoint.exists():
            previous_best = load_model(checkpoint)
            threshold = float(previous_best.evaluate(generators["validation"], verbose=0)[0])
            del previous_best
        model = build_model(learning_rate)
        pending = output / f".{name}_pending.keras"
        pending.unlink(missing_ok=True)

        def commit_checkpoint(epoch, logs):
            if pending.exists():
                pending.replace(checkpoint)

        callbacks = [
            ModelCheckpoint(str(pending), monitor="val_loss", save_best_only=True,
                            initial_value_threshold=threshold),
            LambdaCallback(on_epoch_end=commit_checkpoint),
            EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        ]
        if augment:
            callbacks.append(ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                             patience=2, min_lr=1e-6))
        history_logger = CSVLogger(str(history_path), append=resume)
        callbacks.extend([
            history_logger,
            BackupAndRestore(backup_dir=str(output / "backup" / name), save_freq="epoch",
                             double_checkpoint=True, delete_checkpoint=False),
        ])
        print(f"\n{'Reanudacion' if resume else 'Entrenamiento'}: {name}")
        try:
            model.fit(generators["train"], validation_data=generators["validation"],
                      epochs=epoch_count, callbacks=callbacks)
        finally:
            if getattr(history_logger, "csv_file", None) is not None:
                history_logger.csv_file.close()
        histories[name] = read_training_history(history_path)
        model = load_model(checkpoint)
        validation_losses[name] = float(model.evaluate(generators["validation"], verbose=0)[0])
        write_json_atomic(completed, {"history": histories[name],
                                     "validation_loss": validation_losses[name]})
        del model, generators
    winner = min(validation_losses, key=validation_losses.get)
    shutil.copy2(output / f"{winner}.keras", output / "best_model.keras")
    return histories, validation_losses, winner


def evaluate_model(model, generator, paths, output):
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix

    generator.reset()
    probabilities = model.predict(generator, verbose=1)
    if hasattr(generator, "classes"):
        expected = np.asarray(generator.classes)
        paths = generator.filepaths
    else:
        expected = np.argmax(generator.y, axis=1)
    predicted = np.argmax(probabilities, axis=1)
    report = classification_report(expected, predicted, labels=[0, 1],
                                   target_names=list(CLASSES), output_dict=True, zero_division=0)
    report["log_loss"] = float(-np.log(np.clip(probabilities[np.arange(len(expected)), expected],
                                              1e-7, 1)).mean())
    matrix = confusion_matrix(expected, predicted, labels=[0, 1])
    display = ConfusionMatrixDisplay(matrix, display_labels=CLASSES)
    display.plot(cmap="Greens")
    display.figure_.savefig(output / "confusion_matrix.png", bbox_inches="tight")
    plt.close(display.figure_)
    with (output / "test_predictions.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["path", "actual", "predicted", "probability_cat", "probability_dog"])
        for path, actual, prediction, scores in zip(paths, expected, predicted, probabilities):
            writer.writerow([str(path), CLASSES[actual], CLASSES[prediction], *scores.tolist()])
    return report


def plot_history(histories, output):
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    for name, history in histories.items():
        for axis, metric in zip(axes, ("loss", "accuracy")):
            epochs = range(1, len(history[metric]) + 1)
            axis.plot(epochs, history[metric], label=f"{name}: train")
            axis.plot(epochs, history[f"val_{metric}"], linestyle="--", label=f"{name}: validation")
            axis.set(xlabel="Epoca", ylabel=metric)
            axis.legend()
    figure.tight_layout()
    figure.savefig(output / "learning_curves.png")
    plt.close(figure)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/raw/train")
    parser.add_argument("--archive", type=Path, help="ZIP train.zip con imagenes etiquetadas")
    parser.add_argument("--output", type=Path, default=ROOT / "models/image_classifier")
    parser.add_argument("--split-dir", type=Path, default=ROOT / "data/interim/cats_dogs")
    parser.add_argument("--mode", choices=("auto", "memory", "directory"), default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--baseline-epochs", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--resume", action="store_true", help="Continuar el entrenamiento de --output")
    parser.add_argument("--prepare-only", action="store_true", help="Explorar y particionar sin entrenar")
    args = parser.parse_args()
    if min(args.batch_size, args.baseline_epochs, args.epochs) < 1:
        parser.error("Los tamanos de lote y las epocas deben ser positivos.")
    if args.resume and (args.prepare_only or args.archive):
        parser.error("--resume no se combina con --prepare-only ni --archive.")
    return args


def main():
    args = parse_args()
    if args.archive:
        extract_images(args.archive, args.data_dir)
    if not args.data_dir.is_dir():
        raise SystemExit(f"Falta el dataset: {args.data_dir}. Indica --data-dir o --archive train.zip.")
    records, rejected = discover_images(args.data_dir)
    partitions = split_records(records)
    memory = available_memory()
    mode = select_loading_mode(args.mode, len(records), memory)
    args.output.mkdir(parents=True, exist_ok=True)
    if not args.prepare_only:
        try:
            prepare_training_run(args.output, partitions, args.batch_size,
                                 args.baseline_epochs, args.epochs, args.resume)
        except ValueError as error:
            raise SystemExit(str(error)) from error
    split_root = prepare_directories(partitions, args.split_dir)
    plot_examples(records, args.output)
    summary = {
        "seed": SEED, "image_size": list(IMAGE_SIZE), "class_indices": {"cat": 0, "dog": 1},
        "mode": mode, "available_memory_gib": round(memory / 1024**3, 2),
        "splits": {split: {label: sum(category == label for _, category in rows)
                            for label in CLASSES} for split, rows in partitions.items()},
        "split_directory": str(split_root), "rejected": rejected,
    }
    with (args.output / "split_manifest.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["path", "label", "split"])
        for split, rows in partitions.items():
            writer.writerows((str(path), label, split) for path, label in rows)
    (args.output / "dataset_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    if args.prepare_only:
        return

    import os

    os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")
    os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
    import tensorflow as tf
    from tensorflow.keras.models import load_model

    tf.keras.utils.set_random_seed(SEED)
    arrays = load_arrays(partitions) if mode == "memory" else None
    histories, validation_losses, winner = train_models(
        split_root, arrays, args.output, args.batch_size, args.baseline_epochs, args.epochs,
        resume=args.resume,
    )
    plot_history(histories, args.output)
    tf.keras.backend.clear_session()
    model = load_model(args.output / "best_model.keras")
    test = make_generators(split_root, arrays, args.batch_size)["test"]
    report = evaluate_model(model, test, [path for path, _ in partitions["test"]], args.output)
    results = {"selected_model": winner, "validation_losses": validation_losses,
               "test": report, "history": histories,
               "training": {"weights": None, "batch_size": args.batch_size,
                            "baseline_epochs": args.baseline_epochs, "optimized_epochs": args.epochs},
               "tensorflow_version": tf.__version__}
    (args.output / "metrics.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()