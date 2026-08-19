"""Entrena una CNN en Fashion-MNIST y evalúa su robustez con FGSM.

La implementación conserva el protocolo documentado en el informe del proyecto:
semilla 42, dos bloques convolucionales, salida en logits, callbacks de
entrenamiento y evaluación adversaria para siete valores de epsilon.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

import tensorflow as tf
from tensorflow.keras import layers, models


SEED = 42
EPSILON_VALUES = (0.00, 0.01, 0.03, 0.05, 0.10, 0.20, 0.30)
LOSS_FN = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)


def configure_reproducibility(seed: int = SEED) -> None:
    """Configura la semilla y activa determinismo cuando está disponible."""

    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        # Algunos entornos o dispositivos no ofrecen determinismo completo.
        pass


def load_fashion_mnist():
    """Carga Fashion-MNIST y normaliza sus píxeles al rango [0, 1]."""

    (x_train, y_train), (x_test, y_test) = (
        tf.keras.datasets.fashion_mnist.load_data()
    )

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    return (
        x_train[..., tf.newaxis],
        y_train,
        x_test[..., tf.newaxis],
        y_test,
    )


def build_model() -> tf.keras.Model:
    """Construye la CNN utilizada en el experimento."""

    model = models.Sequential(
        [
            layers.Input(shape=(28, 28, 1)),
            layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.50),
            # Sin Softmax: la capa devuelve logits.
            layers.Dense(10),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=LOSS_FN,
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    return model


def train_model(
    model: tf.keras.Model,
    x_train,
    y_train,
    *,
    epochs: int = 20,
    batch_size: int = 128,
    verbose: int = 1,
):
    """Entrena la CNN con callbacks de estabilidad."""

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=4,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-5,
        ),
    ]

    return model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        shuffle=True,
        callbacks=callbacks,
        verbose=verbose,
    )


@tf.function
def fgsm_attack(
    model: tf.keras.Model,
    images: tf.Tensor,
    labels: tf.Tensor,
    epsilon: float,
) -> tf.Tensor:
    """Genera copias adversarias mediante FGSM sin modificar el modelo."""

    images = tf.cast(images, tf.float32)
    epsilon = tf.cast(epsilon, tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(images)
        logits = model(images, training=False)
        loss = LOSS_FN(labels, logits)

    gradient = tape.gradient(loss, images)
    signed_gradient = tf.sign(gradient)
    adversarial_images = images + epsilon * signed_gradient
    adversarial_images = tf.clip_by_value(adversarial_images, 0.0, 1.0)

    return tf.stop_gradient(adversarial_images)


def evaluate_fgsm(
    model: tf.keras.Model,
    images,
    labels,
    epsilon: float,
    *,
    batch_size: int = 256,
) -> tuple[float, float]:
    """Calcula accuracy robusta y ASR para un valor de epsilon."""

    dataset = tf.data.Dataset.from_tensor_slices((images, labels)).batch(
        batch_size
    )

    total = 0
    adversarial_correct = 0
    initially_correct = 0
    fooled_from_correct = 0

    for image_batch, label_batch in dataset:
        clean_logits = model(image_batch, training=False)
        clean_predictions = tf.argmax(
            clean_logits,
            axis=1,
            output_type=tf.int64,
        )

        adversarial_batch = fgsm_attack(
            model,
            image_batch,
            label_batch,
            epsilon,
        )
        adversarial_logits = model(adversarial_batch, training=False)
        adversarial_predictions = tf.argmax(
            adversarial_logits,
            axis=1,
            output_type=tf.int64,
        )

        labels_int64 = tf.cast(label_batch, tf.int64)
        clean_correct_mask = tf.equal(clean_predictions, labels_int64)
        adversarial_correct_mask = tf.equal(
            adversarial_predictions,
            labels_int64,
        )
        fooled_mask = tf.logical_and(
            clean_correct_mask,
            tf.logical_not(adversarial_correct_mask),
        )

        total += int(tf.size(label_batch).numpy())
        adversarial_correct += int(
            tf.reduce_sum(tf.cast(adversarial_correct_mask, tf.int32)).numpy()
        )
        initially_correct += int(
            tf.reduce_sum(tf.cast(clean_correct_mask, tf.int32)).numpy()
        )
        fooled_from_correct += int(
            tf.reduce_sum(tf.cast(fooled_mask, tf.int32)).numpy()
        )

    robust_accuracy = adversarial_correct / total
    attack_success_rate = (
        fooled_from_correct / initially_correct
        if initially_correct > 0
        else 0.0
    )
    return robust_accuracy, attack_success_rate


def evaluate_epsilons(
    model: tf.keras.Model,
    images,
    labels,
    epsilons: Iterable[float] = EPSILON_VALUES,
    *,
    batch_size: int = 256,
) -> list[dict[str, float]]:
    """Repite la evaluación FGSM para cada epsilon solicitado."""

    results = []
    for epsilon in epsilons:
        robust_accuracy, attack_success_rate = evaluate_fgsm(
            model,
            images,
            labels,
            float(epsilon),
            batch_size=batch_size,
        )
        results.append(
            {
                "epsilon": float(epsilon),
                "robust_accuracy": robust_accuracy,
                "attack_success_rate": attack_success_rate,
            }
        )
    return results


def save_metrics(results: list[dict[str, float]], output_path: Path) -> None:
    """Guarda las métricas adversarias en CSV."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("epsilon", "robust_accuracy", "attack_success_rate"),
        )
        writer.writeheader()
        writer.writerows(results)


def save_history(history, output_path: Path) -> None:
    """Guarda el historial de entrenamiento como JSON."""

    serializable = {
        key: [float(value) for value in values]
        for key, values in history.history.items()
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(serializable, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def plot_metrics(results: list[dict[str, float]], output_path: Path) -> None:
    """Genera una gráfica con accuracy robusta y ASR."""

    import matplotlib.pyplot as plt

    epsilons = [row["epsilon"] for row in results]
    robust = [100 * row["robust_accuracy"] for row in results]
    asr = [100 * row["attack_success_rate"] for row in results]

    figure, axis = plt.subplots(figsize=(9, 5))
    axis.plot(epsilons, robust, marker="o", label="Accuracy robusta")
    axis.plot(epsilons, asr, marker="o", label="ASR")
    axis.set_xlabel("Epsilon")
    axis.set_ylabel("Porcentaje")
    axis.set_xticks(epsilons)
    axis.set_ylim(0, 100)
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Entrena una CNN en Fashion-MNIST y evalúa FGSM."
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/generated"),
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce la salida del entrenamiento.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_reproducibility()

    x_train, y_train, x_test, y_test = load_fashion_mnist()
    model = build_model()
    history = train_model(
        model,
        x_train,
        y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=0 if args.quiet else 1,
    )

    _, clean_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Accuracy limpia: {clean_accuracy:.4f}")

    results = evaluate_epsilons(
        model,
        x_test,
        y_test,
        batch_size=args.eval_batch_size,
    )

    print("\nEvaluación adversaria")
    print("epsilon | accuracy robusta | attack success rate")
    for row in results:
        print(
            f"{row['epsilon']:7.2f} | "
            f"{row['robust_accuracy']:16.4f} | "
            f"{row['attack_success_rate']:19.4f}"
        )

    save_metrics(results, args.output_dir / "fgsm_metrics.csv")
    save_history(history, args.output_dir / "training_history.json")
    plot_metrics(results, args.output_dir / "fgsm_metrics.png")
    print(f"\nResultados guardados en: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
