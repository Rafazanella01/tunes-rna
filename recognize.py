"""
recognize.py
------------
Reconhecimento de genero musical em tempo real.

Grava audio do microfone, divide em segmentos de 3s e classifica com os
modelos treinados. Se ambos existirem, MLP e CNN votam juntos (media das
probabilidades) para uma previsao final mais estavel.

O audio gravado e salvo em recordings/ (ignorada pelo git) para depuracao.

Uso:  python recognize.py
"""

import sys
import datetime

import wave

import numpy as np
import pyaudio

from audio_features import (
    GENRES, SAMPLE_RATE, MODELS_DIR, SEGMENT_SAMPLES,
    PROJECT_ROOT, segment_audio, extract_fingerprint, extract_melspectrogram,
)

DURATION = 9   # segundos gravados (=> 3 segmentos de 3s)
CHUNK = 1024

RECORDINGS_DIR = PROJECT_ROOT / "recordings"


def record_audio(duration: int, sample_rate: int) -> np.ndarray:
    """Grava audio do microfone e retorna array float32 normalizado [-1, 1]."""
    pa = pyaudio.PyAudio()
    print(f"Gravando por {duration}s... toque uma musica!")
    stream = pa.open(format=pyaudio.paInt16, channels=1, rate=sample_rate,
                     input=True, frames_per_buffer=CHUNK)

    frames = []
    num_chunks = int((sample_rate / CHUNK) * duration)
    for i in range(num_chunks):
        frames.append(stream.read(CHUNK, exception_on_overflow=False))
        progress = int((i + 1) / num_chunks * 20)
        print(f"\r  [{'#' * progress}{'-' * (20 - progress)}]", end="", flush=True)
    print("  ok")

    stream.stop_stream()
    stream.close()
    pa.terminate()

    raw = b"".join(frames)

    # Salva o audio em recordings/ para depuracao (pasta ignorada pelo git)
    RECORDINGS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    wav_path = RECORDINGS_DIR / f"rec_{timestamp}.wav"
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)          # int16 = 2 bytes
        wf.setframerate(sample_rate)
        wf.writeframes(raw)         # raw ja e bytes int16
    print(f"  Audio salvo em: {wav_path}")

    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return audio


def load_models():
    """Carrega os modelos disponiveis em models/. Importa TF so se necessario."""
    from tensorflow import keras
    import joblib

    models = {}
    if (MODELS_DIR / "mlp.keras").exists():
        models["mlp"] = (
            keras.models.load_model(MODELS_DIR / "mlp.keras"),
            joblib.load(MODELS_DIR / "scaler.pkl"),
        )
    if (MODELS_DIR / "cnn.keras").exists():
        models["cnn"] = (
            keras.models.load_model(MODELS_DIR / "cnn.keras"),
            np.load(MODELS_DIR / "cnn_norm.npz"),
        )
    if not models:
        sys.exit("Nenhum modelo encontrado. Rode train_mlp.py e/ou train_cnn.py.")
    return models


def predict(audio: np.ndarray, models: dict) -> np.ndarray:
    """Devolve o vetor medio de probabilidades sobre todos os segmentos/modelos."""
    segments = segment_audio(audio) or [audio[:SEGMENT_SAMPLES]]
    all_probs = []

    for seg in segments:
        if "mlp" in models:
            model, scaler = models["mlp"]
            fp = scaler.transform(extract_fingerprint(seg)[np.newaxis, :])
            all_probs.append(model.predict(fp, verbose=0)[0])
        if "cnn" in models:
            model, norm = models["cnn"]
            mel = extract_melspectrogram(seg)
            mel = ((mel - norm["mean"]) / norm["std"])[np.newaxis, ..., np.newaxis]
            all_probs.append(model.predict(mel, verbose=0)[0])

    return np.mean(all_probs, axis=0)


def main():
    print("=" * 55)
    print("  Tunes-RNA - Reconhecimento de Genero Musical")
    print("=" * 55)

    models = load_models()
    print(f"Modelos carregados: {', '.join(m.upper() for m in models)}\n")

    audio = record_audio(DURATION, SAMPLE_RATE)
    probs = predict(audio, models)

    ranking = np.argsort(probs)[::-1]
    print("\n" + "=" * 55)
    print(f"  GENERO: {GENRES[ranking[0]].upper()}  ({probs[ranking[0]] * 100:.1f}%)")
    print("=" * 55)
    print("  Ranking completo:")
    for r in ranking:
        bar = "#" * int(probs[r] * 30)
        print(f"  {GENRES[r]:<10} {probs[r] * 100:5.1f}%  {bar}")


if __name__ == "__main__":
    main()
