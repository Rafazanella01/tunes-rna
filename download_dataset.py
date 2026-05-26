"""Baixa o dataset GTZAN para a pasta Data/ (ignorada pelo git)."""

import os
from pathlib import Path

# Pasta Data/ na raiz do projeto — esta pasta esta no .gitignore
DATA_DIR = Path(__file__).resolve().parent / "Data"
DATA_DIR.mkdir(exist_ok=True)

# Faz o kagglehub usar Data/ como cache em vez de ~/.cache/kagglehub
os.environ["KAGGLEHUB_CACHE"] = str(DATA_DIR)

import kagglehub

# Download latest version
path = kagglehub.dataset_download("andradaolteanu/gtzan-dataset-music-genre-classification")

print("Path to dataset files:", path)
