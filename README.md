# 🎵 Tunes-RNA — Classificação de Gênero Musical com RNA

**Trabalho 2 - Cadeira de Inteligência Artificial (UNISC)**

Sistema de **reconhecimento e classificação de gênero musical em tempo real**, capturando áudio pelo microfone e utilizando uma **Rede Neural Artificial (RNA)** para identificar o gênero da música tocada.

---

## 🧠 Como Funciona

O sistema é dividido em duas fases:

### Fase 1 — Treino (Offline)
1. Utilizamos o dataset **GTZAN** (1.000 clipes de áudio, 10 gêneros musicais)
2. Para cada clipe, extraímos **features acústicas** com a biblioteca `librosa`:
   - **MFCC** (13 coeficientes) — representa o timbre e textura sonora
   - **Chroma** (12 valores) — representa as notas musicais presentes
   - **Centroid Espectral** (1 valor) — representa o "brilho" do som
3. O vetor de **26 dimensões** resultante é o **fingerprint** da música
4. Uma **RNA MLP (Multi-Layer Perceptron)** é treinada para classificar esses vetores nos 10 gêneros

### Fase 2 — Reconhecimento (Online)
1. O microfone grava um trecho de áudio (5 segundos)
2. O mesmo processo de extração de features é aplicado ao áudio gravado
3. O vetor de 26 dimensões é passado para a **RNA já treinada**
4. A RNA retorna o **gênero musical identificado**

```
🎤 Microfone
    ↓
librosa (extrai MFCC + Chroma + Centroid)
    ↓
Vetor de 26 dimensões (fingerprint)
    ↓
🧠 RNA MLP (treinada no GTZAN)
    ↓
🎸 "rock" | 🎷 "jazz" | 🎹 "classical" | ...
```

---

## 🎼 Dataset — GTZAN

| Propriedade | Detalhe |
|---|---|
| **Fonte** | [Kaggle — GTZAN Dataset](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification) |
| **Total de clipes** | 1.000 arquivos `.wav` (30 segundos cada) |
| **Gêneros** | blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock |
| **Clipes por gênero** | 100 |
| **Features pré-extraídas** | `features_3_sec.csv` e `features_30_sec.csv` |

---

## 🛠️ Tecnologias e Bibliotecas

- **Python 3.x**
- **librosa** — Extração de features acústicas (MFCC, Chroma, Spectral Centroid)
- **numpy** — Manipulação de arrays numéricos
- **pyaudio** — Captura de áudio pelo microfone
- **scikit-learn / TensorFlow** — Treinamento e inferência da RNA
- **soundfile** — Leitura/escrita de arquivos de áudio

---

## 🚀 Como Executar

### 1. Pré-requisitos
```bash
# Clonar o repositório
git clone https://github.com/Rafazanella01/tunes-rna.git
cd tunes-rna

# Criar e ativar o ambiente virtual
py -m venv venv
.\venv\Scripts\activate  # Windows
```

### 2. Instalar dependências
```bash
pip install -r requirements.txt
```

### 3. Baixar o dataset
Acesse o link do GTZAN no Kaggle e extraia na pasta `data/` dentro do projeto.

### 4. Testar a captura do microfone
```bash
python test_mic.py
```

### 5. Treinar a RNA *(em breve)*
```bash
python train.py
```

### 6. Reconhecer gênero em tempo real *(em breve)*
```bash
python recognize.py
```

---

## 📁 Estrutura do Projeto

```
tunes-rna/
├── data/                  # Dataset GTZAN (não versionado)
├── venv/                  # Ambiente virtual Python (não versionado)
├── test_mic.py            # Teste de captura e extração de features
├── train.py               # Treino da RNA (em breve)
├── recognize.py           # Reconhecimento em tempo real (em breve)
├── requirements.txt       # Dependências do projeto
└── README.md
```