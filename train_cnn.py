"""
train_cnn.py
------------
Treina a CNN 2D sobre mel-spectrogramas — trata o som como imagem.

Modelo mais profundo que o MLP: as camadas Conv2D aprendem padroes locais
de tempo-frequencia (ataque de bateria, harmonia, distorcao) em vez de
depender de medias temporais como o fingerprint 26-dim.

Artefatos salvos em models/:  cnn.keras  +  cnn_norm.npz
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras

from audio_features import GENRES, MODELS_DIR
from data_utils import load_dataset, split_by_song

EPOCHS = 120
BATCH_SIZE = 32


def build_model(input_shape: tuple, n_classes: int) -> keras.Model:
    """CNN com 4 blocos convolucionais + GlobalAveragePooling."""
    model = keras.Sequential([
        keras.layers.Input(shape=input_shape),

        keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D(2),

        keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D(2),

        keras.layers.Conv2D(128, 3, padding="same", activation="relu"),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D(2),

        keras.layers.Conv2D(128, 3, padding="same", activation="relu"),
        keras.layers.BatchNormalization(),

        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dropout(0.4),
        keras.layers.Dense(n_classes, activation="softmax"),
    ], name="cnn_gtzan")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    tf.random.set_seed(42)

    _, X_cnn, y, track_ids = load_dataset()
    tr, va, te = split_by_song(track_ids, y)
    print(f"Split por musica -> treino {len(tr)} | val {len(va)} | teste {len(te)}")

    # Padroniza os mel-spectrogramas com media/desvio do conjunto de treino.
    mean = X_cnn[tr].mean()
    std = X_cnn[tr].std()
    Xn = ((X_cnn - mean) / std)[..., np.newaxis]  # adiciona canal -> (N,128,130,1)

    model = build_model(Xn.shape[1:], len(GENRES))
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=20, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=8, min_lr=1e-5),
    ]

    model.fit(
        Xn[tr], y[tr],
        validation_data=(Xn[va], y[va]),
        epochs=EPOCHS, batch_size=BATCH_SIZE,
        callbacks=callbacks, verbose=2,
    )

    loss, acc = model.evaluate(Xn[te], y[te], verbose=0)
    print(f"\n[CNN] Acuracia no conjunto de teste: {acc:.4f}  (loss {loss:.4f})")

    MODELS_DIR.mkdir(exist_ok=True)
    model.save(MODELS_DIR / "cnn.keras")
    np.savez(MODELS_DIR / "cnn_norm.npz", mean=mean, std=std)
    print(f"Modelo salvo em {MODELS_DIR / 'cnn.keras'}")
    print(f"Normalizacao salva em {MODELS_DIR / 'cnn_norm.npz'}")


if __name__ == "__main__":
    main()
