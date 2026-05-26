"""
train_mlp.py
------------
Treina o MLP (rede neural densa) sobre o fingerprint 26-dim.

Este e o modelo BASELINE do projeto: simples, rapido e diretamente
compativel com o reconhecimento por microfone (recognize.py), ja que usa
as mesmas 26 features do test_mic.py.

Artefatos salvos em models/:  mlp.keras  +  scaler.pkl
"""

import joblib
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler

from audio_features import GENRES, MODELS_DIR
from data_utils import load_dataset, split_by_song

EPOCHS = 300
BATCH_SIZE = 64


def build_model(input_dim: int, n_classes: int) -> keras.Model:
    """MLP com BatchNorm + Dropout para regularizacao."""
    model = keras.Sequential([
        keras.layers.Input(shape=(input_dim,)),
        keras.layers.Dense(256, activation="relu"),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(n_classes, activation="softmax"),
    ], name="mlp_gtzan")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    tf.random.set_seed(42)

    X_mlp, _, y, track_ids = load_dataset()
    tr, va, te = split_by_song(track_ids, y)
    print(f"Split por musica -> treino {len(tr)} | val {len(va)} | teste {len(te)}")

    # Padroniza as features (media 0, desvio 1). O scaler e ajustado SO no
    # treino e reaplicado nos demais conjuntos e na inferencia.
    scaler = StandardScaler().fit(X_mlp[tr])
    Xtr = scaler.transform(X_mlp[tr])
    Xva = scaler.transform(X_mlp[va])
    Xte = scaler.transform(X_mlp[te])

    model = build_model(X_mlp.shape[1], len(GENRES))
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=30, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=12, min_lr=1e-5),
    ]

    model.fit(
        Xtr, y[tr],
        validation_data=(Xva, y[va]),
        epochs=EPOCHS, batch_size=BATCH_SIZE,
        callbacks=callbacks, verbose=2,
    )

    loss, acc = model.evaluate(Xte, y[te], verbose=0)
    print(f"\n[MLP] Acuracia no conjunto de teste: {acc:.4f}  (loss {loss:.4f})")

    MODELS_DIR.mkdir(exist_ok=True)
    model.save(MODELS_DIR / "mlp.keras")
    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
    print(f"Modelo salvo em {MODELS_DIR / 'mlp.keras'}")
    print(f"Scaler salvo em {MODELS_DIR / 'scaler.pkl'}")


if __name__ == "__main__":
    main()
