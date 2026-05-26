"""
compare.py
----------
Avalia MLP e CNN no MESMO conjunto de teste e gera a comparacao:

  - relatorio de metricas por genero (precision / recall / f1)
  - acuracia por segmento e por musica (votacao das probabilidades)
  - matrizes de confusao lado a lado em models/comparison.png

Rode depois de treinar os dois modelos:  python compare.py
"""

import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")  # backend sem janela — apenas salva o arquivo
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from audio_features import GENRES, MODELS_DIR
from data_utils import load_dataset, split_by_song


def song_level(probs: np.ndarray, y: np.ndarray, track_ids: np.ndarray):
    """Agrega as probabilidades dos segmentos de cada musica e vota a classe."""
    songs = np.unique(track_ids)
    y_true, y_pred = [], []
    for s in songs:
        m = track_ids == s
        y_true.append(y[m][0])
        y_pred.append(probs[m].mean(axis=0).argmax())
    return np.array(y_true), np.array(y_pred)


def plot_confusion(ax, y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred, normalize="true")
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_title(title)
    ax.set_xticks(range(len(GENRES)))
    ax.set_yticks(range(len(GENRES)))
    ax.set_xticklabels(GENRES, rotation=45, ha="right")
    ax.set_yticklabels(GENRES)
    ax.set_xlabel("Previsto")
    ax.set_ylabel("Real")
    for i in range(len(GENRES)):
        for j in range(len(GENRES)):
            ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center",
                    color="white" if cm[i, j] > 0.5 else "black", fontsize=7)
    return im


def evaluate(name, probs, y, track_ids):
    """Imprime metricas de um modelo e devolve dados para plotar."""
    seg_pred = probs.argmax(axis=1)
    seg_acc = accuracy_score(y, seg_pred)
    yt_song, yp_song = song_level(probs, y, track_ids)
    song_acc = accuracy_score(yt_song, yp_song)

    print(f"\n{'=' * 60}\n  {name}\n{'=' * 60}")
    print(f"  Acuracia por segmento : {seg_acc:.4f}")
    print(f"  Acuracia por musica   : {song_acc:.4f}  (votacao dos segmentos)")
    print("\n  Relatorio por genero (nivel musica):")
    print(classification_report(yt_song, yp_song, target_names=GENRES, digits=3))
    return seg_acc, song_acc, yt_song, yp_song


def main():
    X_mlp, X_cnn, y, track_ids = load_dataset()
    _, _, te = split_by_song(track_ids, y)
    yte, tte = y[te], track_ids[te]

    results = {}

    # --- MLP ---
    mlp_path = MODELS_DIR / "mlp.keras"
    if mlp_path.exists():
        scaler = joblib.load(MODELS_DIR / "scaler.pkl")
        mlp = keras.models.load_model(mlp_path)
        probs = mlp.predict(scaler.transform(X_mlp[te]), verbose=0)
        results["MLP (fingerprint 26-dim)"] = evaluate(
            "MLP - fingerprint 26-dim", probs, yte, tte)
    else:
        print("! mlp.keras nao encontrado — rode train_mlp.py")

    # --- CNN ---
    cnn_path = MODELS_DIR / "cnn.keras"
    if cnn_path.exists():
        norm = np.load(MODELS_DIR / "cnn_norm.npz")
        cnn = keras.models.load_model(cnn_path)
        Xn = ((X_cnn[te] - norm["mean"]) / norm["std"])[..., np.newaxis]
        probs = cnn.predict(Xn, verbose=0)
        results["CNN (mel-spectrograma)"] = evaluate(
            "CNN - mel-spectrograma", probs, yte, tte)
    else:
        print("! cnn.keras nao encontrado — rode train_cnn.py")

    if not results:
        return

    # --- Resumo + matrizes de confusao ---
    print(f"\n{'=' * 60}\n  RESUMO\n{'=' * 60}")
    print(f"  {'Modelo':<32}{'Segmento':>12}{'Musica':>12}")
    for name, (seg_acc, song_acc, _, _) in results.items():
        print(f"  {name:<32}{seg_acc:>11.3f}{song_acc:>12.3f}")

    fig, axes = plt.subplots(1, len(results), figsize=(7 * len(results), 6))
    if len(results) == 1:
        axes = [axes]
    for ax, (name, (_, song_acc, yt, yp)) in zip(axes, results.items()):
        im = plot_confusion(ax, yt, yp, f"{name}\nacuracia musica = {song_acc:.3f}")
    fig.colorbar(im, ax=axes, fraction=0.025)
    out = MODELS_DIR / "comparison.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\n  Matrizes de confusao salvas em: {out}")


if __name__ == "__main__":
    main()
