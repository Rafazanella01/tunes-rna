"""
audio_features.py
-----------------
Modulo compartilhado de extracao de features de audio.

Usado tanto no treino (build_dataset.py) quanto no reconhecimento em tempo
real (recognize.py). Centralizar aqui garante que treino e inferencia usem
EXATAMENTE o mesmo pre-processamento — pre-requisito para o modelo funcionar
fora do dataset.
"""

from pathlib import Path

import numpy as np
import librosa

# --- Configuracoes globais de audio ---
SAMPLE_RATE = 22050        # Hz (padrao do librosa e do GTZAN)
SEGMENT_DURATION = 3.0     # segundos por segmento de treino
N_MFCC = 13                # coeficientes MFCC
N_MELS = 128               # bandas mel do espectrograma (entrada da CNN)
HOP_LENGTH = 512           # passo da janela STFT

# 10 generos do GTZAN (ordem fixa = indice da classe)
GENRES = [
    "blues", "classical", "country", "disco", "hiphop",
    "jazz", "metal", "pop", "reggae", "rock",
]

# Amostras por segmento e numero fixo de frames do mel-spectrograma.
SEGMENT_SAMPLES = int(SEGMENT_DURATION * SAMPLE_RATE)
MEL_FRAMES = 1 + SEGMENT_SAMPLES // HOP_LENGTH

# --- Caminhos do projeto ---
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "Data"        # ignorada pelo git
MODELS_DIR = PROJECT_ROOT / "models"    # artefatos de treino (ignorada pelo git)


def find_genres_dir() -> Path:
    """Localiza a pasta `genres_original/` baixada pelo kagglehub dentro de Data/."""
    matches = list(DATA_DIR.rglob("genres_original"))
    if not matches:
        raise FileNotFoundError(
            "Pasta 'genres_original' nao encontrada em Data/.\n"
            "Rode primeiro:  python download_dataset.py"
        )
    return matches[0]


def segment_audio(audio: np.ndarray, segment_samples: int = SEGMENT_SAMPLES) -> list:
    """Divide um array de audio em segmentos de tamanho fixo (sobra e descartada)."""
    n = len(audio) // segment_samples
    return [audio[i * segment_samples:(i + 1) * segment_samples] for i in range(n)]


def extract_fingerprint(audio: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Fingerprint 26-dim usado pelo MLP: 13 MFCC + 12 Chroma + 1 Centroid.

    Cada feature e a media temporal — o vetor independe da duracao do audio,
    entao serve tanto para segmentos de 3s do treino quanto para o microfone.
    """
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)
    mfcc_mean = np.mean(mfcc, axis=1)                       # (13,)

    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)                   # (12,)

    centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
    centroid_mean = float(np.mean(centroid))                # (1,)

    vector = np.concatenate([mfcc_mean, chroma_mean, [centroid_mean]])
    return vector.astype(np.float32)


def extract_melspectrogram(audio: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Mel-spectrograma em dB usado pela CNN. Shape fixo (N_MELS, MEL_FRAMES).

    Funciona como uma "imagem" do som: eixo Y = frequencia (mel),
    eixo X = tempo, valor = energia em decibeis.
    """
    mel = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_mels=N_MELS, hop_length=HOP_LENGTH
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # Garante numero fixo de frames (pad com silencio ou corta).
    if mel_db.shape[1] < MEL_FRAMES:
        pad = MEL_FRAMES - mel_db.shape[1]
        mel_db = np.pad(mel_db, ((0, 0), (0, pad)),
                        mode="constant", constant_values=mel_db.min())
    else:
        mel_db = mel_db[:, :MEL_FRAMES]

    return mel_db.astype(np.float32)
