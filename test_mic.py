"""
test_mic.py
-----------
Script de teste: captura audio do microfone por alguns segundos,
extrai features de áudio (MFCC + Chroma) usando librosa como fingerprint
e exibe o resultado no terminal.

Abordagem: em vez de usar o binário nativo Chromaprint/fpcalc,
usamos librosa para extrair características acústicas — que serão
a entrada da nossa RNA posteriormente.
"""

import pyaudio
import numpy as np

from audio_features import extract_fingerprint

# --- Configurações de gravação ---
SAMPLE_RATE = 22050   # Hz (padrão do librosa)
CHANNELS = 1          # Mono
DURATION = 5          # Segundos a gravar
CHUNK = 1024          # Tamanho do buffer de leitura


def record_audio(duration: int, sample_rate: int) -> np.ndarray:
    """Grava audio do microfone e retorna como array numpy float32."""
    pa = pyaudio.PyAudio()

    print(f"🎤 Gravando por {duration} segundos... Fale ou toque algo!")
    stream = pa.open(
        format=pyaudio.paInt16,
        channels=CHANNELS,
        rate=sample_rate,
        input=True,
        frames_per_buffer=CHUNK,
    )

    frames = []
    num_chunks = int((sample_rate / CHUNK) * duration)
    for i in range(num_chunks):
        data = stream.read(CHUNK, exception_on_overflow=False)
        frames.append(data)
        progress = int((i + 1) / num_chunks * 20)
        print(f"\r  [{'█' * progress}{'░' * (20 - progress)}] {int((i+1)/num_chunks*100)}%", end="", flush=True)

    print("\n✅ Gravação finalizada!")

    stream.stop_stream()
    stream.close()
    pa.terminate()

    # Converte bytes PCM int16 → float32 normalizado [-1.0, 1.0]
    raw = b"".join(frames)
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return audio


def generate_fingerprint(audio: np.ndarray, sample_rate: int) -> dict:
    """
    Gera o fingerprint 26-dim do áudio (13 MFCC + 12 Chroma + 1 Centroid).

    A extração foi movida para o módulo `audio_features`, compartilhado com
    o treino (build_dataset.py) e o reconhecimento (recognize.py) — assim
    treino e inferência usam exatamente o mesmo pré-processamento.
    """
    vector = extract_fingerprint(audio, sample_rate)
    return {
        "mfcc": vector[:13],
        "chroma": vector[13:25],
        "spectral_centroid": float(vector[25]),
        "vector": vector,
    }


def main():
    print("=" * 55)
    print("  🎵 Tunes-RNA - Teste de Fingerprint Acústico")
    print("=" * 55)

    # 1. Grava o áudio
    audio = record_audio(DURATION, SAMPLE_RATE)
    print(f"  → Array de áudio: shape={audio.shape}, dtype={audio.dtype}")

    # 2. Gera o fingerprint
    print("\n🔍 Extraindo features acústicas (MFCC + Chroma + Centroid)...")
    fp = generate_fingerprint(audio, SAMPLE_RATE)

    # 3. Exibe resultado
    print("\n" + "=" * 55)
    print("  📄 Fingerprint Gerado:")
    print("=" * 55)
    print(f"  MFCC (13 coef.)    : {np.round(fp['mfcc'], 2)}")
    print(f"  Chroma (12 notas)  : {np.round(fp['chroma'], 3)}")
    print(f"  Centroid espectral : {fp['spectral_centroid']:.2f} Hz")
    print(f"\n  🔢 Vetor final ({len(fp['vector'])} dimensões):")
    print(f"  {np.round(fp['vector'], 4)}")
    print("\n✨ Sucesso! Este vetor é o fingerprint que alimentará a RNA.")


if __name__ == "__main__":
    main()
