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
import librosa

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
    Gera o fingerprint a partir das features acústicas do áudio.
    
    Features extraídas:
    - MFCC (Mel-Frequency Cepstral Coefficients): captura o timbre/textura sonora
    - Chroma: captura as notas musicais presentes
    - Spectral Centroid: representa o "brilho" do som
    
    O vetor resultante é o que alimentará a RNA.
    """
    # 1. MFCCs — 13 coeficientes, média ao longo do tempo
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)  # shape: (13,)

    # 2. Chroma STFT — 12 classes de notas (C, C#, D, ...), média ao longo do tempo
    chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
    chroma_mean = np.mean(chroma, axis=1)  # shape: (12,)

    # 3. Centroid espectral — média e desvio padrão
    centroid = librosa.feature.spectral_centroid(y=audio, sr=sample_rate)
    centroid_mean = float(np.mean(centroid))

    # Vetor final do fingerprint: 13 + 12 + 1 = 26 valores
    fingerprint_vector = np.concatenate([mfcc_mean, chroma_mean, [centroid_mean]])

    return {
        "mfcc": mfcc_mean,
        "chroma": chroma_mean,
        "spectral_centroid": centroid_mean,
        "vector": fingerprint_vector,
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
