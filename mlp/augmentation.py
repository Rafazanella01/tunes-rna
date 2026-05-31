"""
augmentation.py — Técnicas de Data Augmentation para áudio.

Cada função recebe um sinal de áudio (y, sr) e retorna
uma lista de variações do mesmo sinal — sem alterar o gênero.
"""

import numpy as np
import librosa


def time_stretch(y, rates=(0.85, 1.15)):
    """
    Estica ou comprime o áudio no tempo sem mudar o tom.
    rates: tupla com (mais_lento, mais_rapido)
    """
    return [
        librosa.effects.time_stretch(y, rate=rates[0]),  # mais lento
        librosa.effects.time_stretch(y, rate=rates[1]),  # mais rápido
    ]


def pitch_shift(y, sr, steps=(-2, 2)):
    """
    Sobe ou desce o tom sem mudar a velocidade.
    steps: semitons pra baixo e pra cima
    """
    return [
        librosa.effects.pitch_shift(y, sr=sr, n_steps=steps[0]),  # tom mais baixo
        librosa.effects.pitch_shift(y, sr=sr, n_steps=steps[1]),  # tom mais alto
    ]


def add_noise(y, noise_factor=0.005):
    """
    Adiciona ruído gaussiano leve — simula gravações de qualidade variada.
    """
    noise = np.random.normal(0, noise_factor, len(y))
    return [np.clip(y + noise, -1.0, 1.0)]


def augmentar(y, sr):
    """
    Aplica todas as técnicas e retorna lista com todas as variações.

    Entrada : sinal original (1 amostra)
    Saída   : array de sinais aumentados (5 amostras)

    Total com original = 6x o dataset original.
    """
    variacoes = []
    variacoes.extend(time_stretch(y))
    variacoes.extend(pitch_shift(y, sr))
    variacoes.extend(add_noise(y))
    return variacoes