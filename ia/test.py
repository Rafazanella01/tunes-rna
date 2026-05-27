import datasets.kaggle as kaggle
import pathlib
import librosa
import numpy as np
import matplotlib.pyplot as plt

base_path = pathlib.Path()
#kaggle.gtzan(base_path / "datasets" / "gtzan-data")
#kaggle.spotify3sec(base_path / "datasets" / "spt3sec-data")

def extract_features(file_path, sr=22050, n_mfcc=13, max_len=128, duration=15):
    # Carrega o arquivo definindo a duração
    y, _ = librosa.load(file_path, sr=sr, duration=duration)
    # Caso não tenha a duração de áudio, preenche com silêncio
    y = librosa.util.fix_length(y, size=max_len*sr//duration)
    # Extrai o MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
    return np.mean(mfcc.T, axis=0), librosa.power_to_db(mel_spec)

#features, spec = extract_features("/datasets/gtzan-data/Data/genres_original/rock/rock.00000.wav")
features, spec = extract_features("./datasets/gtzan-data/Data/genres_original/reggae/reggae.00000.wav")

print(f"mfcc: {features}\n")
print(f"spectogram: {spec}")

plt.figure(figsize=(15, 4))
librosa.display.specshow(spec, 
                         x_axis='time', 
                         y_axis='mel', 
                         sr=22050, 
                         cmap='viridis')

plt.colorbar(format='%+2.0f dB')

plt.title('Espectrograma de Mel - Visualização Direta')
plt.tight_layout()
plt.show()