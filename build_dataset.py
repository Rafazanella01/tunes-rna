"""
build_dataset.py
----------------
Le os 1.000 arquivos .wav do GTZAN, divide cada clipe de 30s em segmentos
de 3s e extrai DUAS representacoes por segmento:

  - fingerprint 26-dim  -> entrada do MLP   (train_mlp.py)
  - mel-spectrograma    -> entrada da CNN   (train_cnn.py)

Tudo e salvo em cache em `models/dataset.npz`, junto do `track_id` de cada
segmento (qual musica originou ele) para permitir split sem data leakage.

Rode uma vez:  python build_dataset.py
"""

import time

import numpy as np
import librosa

from audio_features import (
    GENRES, SAMPLE_RATE, MODELS_DIR,
    find_genres_dir, segment_audio,
    extract_fingerprint, extract_melspectrogram,
)


def main():
    genres_dir = find_genres_dir()
    print(f"Dataset:  {genres_dir}")
    print("Extraindo features (pode levar alguns minutos)...\n")

    X_mlp, X_cnn, y, track_ids = [], [], [], []
    track_counter = 0
    skipped = 0
    t0 = time.time()

    for label, genre in enumerate(GENRES):
        genre_dir = genres_dir / genre
        wavs = sorted(genre_dir.glob("*.wav"))
        seg_count = 0

        for wav in wavs:
            try:
                audio, _ = librosa.load(wav, sr=SAMPLE_RATE, mono=True)
            except Exception as exc:
                # jazz.00054.wav do GTZAN e conhecido por estar corrompido.
                print(f"  ! ignorado {wav.name}: {exc}")
                skipped += 1
                continue

            for segment in segment_audio(audio):
                X_mlp.append(extract_fingerprint(segment, SAMPLE_RATE))
                X_cnn.append(extract_melspectrogram(segment, SAMPLE_RATE))
                y.append(label)
                track_ids.append(track_counter)
                seg_count += 1

            track_counter += 1

        print(f"  {genre:<10} {len(wavs):>3} clipes -> {seg_count:>4} segmentos")

    X_mlp = np.stack(X_mlp).astype(np.float32)
    X_cnn = np.stack(X_cnn).astype(np.float32)
    y = np.array(y, dtype=np.int64)
    track_ids = np.array(track_ids, dtype=np.int64)

    MODELS_DIR.mkdir(exist_ok=True)
    out = MODELS_DIR / "dataset.npz"
    np.savez_compressed(out, X_mlp=X_mlp, X_cnn=X_cnn, y=y, track_ids=track_ids)

    dt = time.time() - t0
    print(f"\nConcluido em {dt:.0f}s.  {len(y)} segmentos, {skipped} arquivos ignorados.")
    print(f"  X_mlp: {X_mlp.shape}   X_cnn: {X_cnn.shape}")
    print(f"  Cache salvo em: {out}  ({out.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
