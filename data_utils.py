"""
data_utils.py
-------------
Carregamento do dataset em cache e divisao treino/validacao/teste.

Ponto critico: a divisao e feita POR MUSICA, nao por segmento. Como cada
clipe de 30s vira ~10 segmentos de 3s, se segmentos da mesma musica caissem
em conjuntos diferentes o modelo "decoraria" a musica e a acuracia ficaria
inflada (data leakage). Aqui os segmentos de uma musica ficam sempre juntos.
"""

import numpy as np
from sklearn.model_selection import GroupShuffleSplit

from audio_features import MODELS_DIR

DATASET_PATH = MODELS_DIR / "dataset.npz"
SEED = 42  # fixo => MLP e CNN treinam e sao avaliados no MESMO split


def load_dataset():
    """Retorna X_mlp (N,26), X_cnn (N,128,130), y (N,), track_ids (N,)."""
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"{DATASET_PATH} nao existe.\nRode primeiro:  python build_dataset.py"
        )
    d = np.load(DATASET_PATH, allow_pickle=True)
    return d["X_mlp"], d["X_cnn"], d["y"], d["track_ids"]


def split_by_song(track_ids: np.ndarray, y: np.ndarray, seed: int = SEED):
    """
    Indices de treino/validacao/teste agrupados por musica (~70/15/15).

    Usa GroupShuffleSplit com `groups=track_ids`: nenhum segmento de uma
    mesma musica aparece em dois conjuntos diferentes.
    """
    idx = np.arange(len(y))

    # 1o corte: 70% treino, 30% restante.
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
    train_idx, temp_idx = next(gss1.split(idx, y, groups=track_ids))

    # 2o corte: divide o restante em metade validacao, metade teste.
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=seed)
    val_rel, test_rel = next(
        gss2.split(temp_idx, y[temp_idx], groups=track_ids[temp_idx])
    )
    val_idx = temp_idx[val_rel]
    test_idx = temp_idx[test_rel]

    return train_idx, val_idx, test_idx
