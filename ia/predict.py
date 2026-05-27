"""
predict.py — Testa a RNA treinada com um áudio real.

Uso:
    python predict.py caminho/para/musica.wav
    python predict.py caminho/para/musica.mp3   (qualquer formato suportado pelo librosa)
"""

import sys
import pickle
import numpy as np
import keras
import librosa

# Importa a mesma função de extração usada no treino
# (garante que o vetor de features seja idêntico)
from process_data import extract_features


# ─────────────────────────────────────────────
#  CONFIGURAÇÃO — ajuste se mudou os caminhos
# ─────────────────────────────────────────────
MODELO_PATH  = "./tunes-rna.keras"
SCALER_PATH  = "./scaler.pkl"
ENCODER_PATH = "./label_encoder.pkl"


# ─────────────────────────────────────────────
#  EMOJIS POR GÊNERO (visual)
# ─────────────────────────────────────────────
GENRE_EMOJI = {
    "blues":     "🎸",
    "classical": "🎻",
    "country":   "🤠",
    "disco":     "🕺",
    "hiphop":    "🎤",
    "jazz":      "🎷",
    "metal":     "🤘",
    "pop":       "🎵",
    "reggae":    "🌴",
    "rock":      "🎸",
}


def carregar_modelos():
    """Carrega modelo, scaler e label encoder do disco."""
    try:
        model   = keras.models.load_model(MODELO_PATH)
        with open(SCALER_PATH,  "rb") as f: scaler  = pickle.load(f)
        with open(ENCODER_PATH, "rb") as f: encoder = pickle.load(f)
        return model, scaler, encoder
    except FileNotFoundError as e:
        print(f"\n❌ Arquivo não encontrado: {e}")
        print("   Execute processamento.py e modely.py antes de usar predict.py.")
        sys.exit(1)


def prever_genero(audio_path: str, top_k: int = 3):
    """
    Carrega o áudio, extrai features e retorna as top_k predições.

    Returns:
        lista de (gênero, probabilidade) ordenada decrescentemente
    """
    print(f"\n🎵 Carregando áudio: {audio_path}")

    # ── Features ─────────────────────────────────
    try:
        features = extract_features(audio_path, duration=60)
    except Exception as e:
        print(f"❌ Erro ao processar o áudio: {e}")
        sys.exit(1)

    print(f"✔  {len(features)} features extraídas")

    # ── Pré-processamento ─────────────────────────
    model, scaler, encoder = carregar_modelos()

    x = scaler.transform(features.reshape(1, -1))  # (1, n_features)

    # ── Predição ──────────────────────────────────
    probs = model.predict(x, verbose=0)[0]          # array de probabilidades

    # Ordena decrescentemente e pega top_k
    indices_ordenados = np.argsort(probs)[::-1][:top_k]
    resultados = [
        (encoder.classes_[i], float(probs[i]))
        for i in indices_ordenados
    ]

    return resultados


def exibir_resultado(resultados, audio_path: str):
    """Imprime o resultado de forma legível no terminal."""
    genero_pred, conf_pred = resultados[0]
    emoji = GENRE_EMOJI.get(genero_pred, "🎼")

    print("\n" + "═"*50)
    print("          RESULTADO DA PREDIÇÃO")
    print("═"*50)
    print(f"  Arquivo : {audio_path.split('/')[-1]}")
    print(f"  Gênero  : {emoji}  {genero_pred.upper()}")
    print(f"  Confiança: {conf_pred*100:.1f}%")
    print("─"*50)
    print("  Ranking completo:")
    for i, (genero, prob) in enumerate(resultados, 1):
        barra = "█" * int(prob * 30)
        print(f"  {i}. {genero:<12} {barra:<30} {prob*100:5.1f}%")
    print("═"*50)

    # Aviso se a confiança for baixa
    if conf_pred < 0.40:
        print("\n  ⚠️  Confiança baixa — o áudio pode ser atípico")
        print("     ou de um gênero pouco representado no dataset.\n")


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("\nUso: python predict.py <caminho_do_audio>")
        print("Exemplo: python predict.py musicas/test_rock.wav\n")
        sys.exit(1)

    audio_path = sys.argv[1]

    resultados = prever_genero(audio_path, top_k=3)
    exibir_resultado(resultados, audio_path)