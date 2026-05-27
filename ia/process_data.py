import os
import librosa
import numpy as np

# ─────────────────────────────────────────────
#  EXTRAÇÃO DE FEATURES
# ─────────────────────────────────────────────

def extract_features(file_path, sr=22050, n_mfcc=40, duration=30):
    """
    Extrai um vetor de dados rico para classificação de gênero musical.

    Dados extraídos:
      - MFCC (40 coef)  + delta + delta-delta   → mean+std = 240
      - Chroma STFT (12)                        → mean+std =  24
      - Spectral Contrast (7 bandas)            → mean+std =  14
      - Spectral Centroid                       → mean+std =   2
      - Spectral Bandwidth                      → mean+std =   2
      - Spectral Rolloff                        → mean+std =   2
      - Zero Crossing Rate                      → mean+std =   2
      - RMS Energy                              → mean+std =   2
      - Tonnetz (6)                             → mean+std =  12
      - Tempo (BPM)                             →          =   1
    ─────────────────────────────────────────────────────────────
    Total: ~301 dados
    """

    y, _ = librosa.load(file_path, sr=sr, duration=duration)

    # Garante tamanho fixo: preenche silêncio se o áudio for curto
    target_len = int(sr * duration)
    y = librosa.util.fix_length(y, size=target_len)

    features = []

    # ── 1. MFCC (40 coeficientes) + delta + delta² ──
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    delta_mfcc  = librosa.feature.delta(mfcc)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)

    for mat in [mfcc, delta_mfcc, delta2_mfcc]:
        features.extend(np.mean(mat, axis=1))
        features.extend(np.std(mat,  axis=1))

    # ── 2. Chroma STFT ──────────────────────────────
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    features.extend(np.mean(chroma, axis=1))
    features.extend(np.std(chroma,  axis=1))

    # ── 3. Spectral Contrast ────────────────────────
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    features.extend(np.mean(contrast, axis=1))
    features.extend(np.std(contrast,  axis=1))

    # ── 4. Centroide Espectral ──────────────────────
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    features.append(np.mean(centroid))
    features.append(np.std(centroid))

    # ── 5. Largura de Banda Espectral ───────────────
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    features.append(np.mean(bandwidth))
    features.append(np.std(bandwidth))

    # ── 6. Rolloff Espectral ────────────────────────
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    features.append(np.mean(rolloff))
    features.append(np.std(rolloff))

    # ── 7. Zero Crossing Rate ───────────────────────
    zcr = librosa.feature.zero_crossing_rate(y)
    features.append(np.mean(zcr))
    features.append(np.std(zcr))

    # ── 8. RMS Energy ───────────────────────────────
    rms = librosa.feature.rms(y=y)
    features.append(np.mean(rms))
    features.append(np.std(rms))

    # ── 9. Tonnetz ──────────────────────────────────
    # Captura relações harmônicas e de afinação
    y_harm = librosa.effects.harmonic(y)
    tonnetz = librosa.feature.tonnetz(y=y_harm, sr=sr)
    features.extend(np.mean(tonnetz, axis=1))
    features.extend(np.std(tonnetz,  axis=1))

    # ── 10. Tempo (BPM) ─────────────────────────────
    # librosa >= 0.10 retorna tempo como array (1,) em vez de escalar
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    features.append(float(np.atleast_1d(tempo)[0]))

    return np.array(features, dtype=np.float32)


def process_all_dataset(base_dir):
    print("🎵 Iniciando extração de features do dataset...\n")

    x_features = []
    y_genres    = []

    generos = sorted([
        g for g in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, g))
    ])

    for genre in generos:
        genre_path = os.path.join(base_dir, genre)
        arquivos   = [f for f in os.listdir(genre_path) if f.endswith('.wav')]

        print(f"📁 Gênero: {genre:<12} ({len(arquivos)} arquivos)")

        for file in arquivos:
            caminho_audio = os.path.join(genre_path, file)
            try:
                feat = extract_features(caminho_audio, duration=30)
                x_features.append(feat)
                y_genres.append(genre)
                print(f"   ✔ {file}  →  {len(feat)} features")

            except Exception as e:
                print(f"   ✘ Erro em {file}: {e}")

        print()

    return np.array(x_features), np.array(y_genres)


if __name__ == "__main__":
    CAMINHO_DATASET = "./datasets/gtzan-data/genres_original"

    x, y = process_all_dataset(CAMINHO_DATASET)

    print(f"\n📐 Shape das features: {x.shape}")
    print(f"🏷️  Gêneros únicos:     {np.unique(y)}")

    np.savez_compressed("./datasets/dados_treino.npz", mfcc=x, genres=y)
    print("\n✅ 'dados_treino.npz' gerado com sucesso!")