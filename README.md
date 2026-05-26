# Tunes-RNA — Classificação de Gênero Musical com RNA

**Trabalho 2 — Cadeira de Inteligência Artificial (UNISC)**

Sistema de **reconhecimento e classificação de gênero musical em tempo real**, capturando áudio pelo microfone e utilizando **Redes Neurais Artificiais (RNA)** para identificar o gênero da música tocada.

---

## Como Funciona

O sistema é dividido em duas fases:

### Fase 1 — Treino (Offline)

1. Utilizamos o dataset **GTZAN** (1.000 clipes de 30s, 10 gêneros musicais)
2. Cada clipe é dividido em **~10 segmentos de 3 segundos** → ~10.000 amostras de treino
3. Para cada segmento, extraímos **duas representações** diferentes:
   - **Fingerprint 26-dim** — média temporal de: 13 MFCC + 12 Chroma + 1 Centroid Espectral
   - **Mel-espectrograma 128×130** — "foto" do som em escala de frequência mel, em dB
4. Dois modelos são treinados e avaliados no **mesmo split** de teste:
   - **MLP** — rede densa sobre o fingerprint 26-dim (baseline rápido)
   - **CNN** — rede convolucional sobre o mel-espectrograma (aprende padrões tempo-frequência)
5. O split é feito **por música**, não por segmento, para evitar *data leakage*

### Fase 2 — Reconhecimento (Online)

1. O microfone grava **9 segundos** de áudio
2. O áudio é dividido em **3 segmentos de 3s** (mesma segmentação do treino)
3. Para cada segmento, extraímos fingerprint (para o MLP) e mel-espectrograma (para a CNN)
4. Cada modelo retorna um vetor de probabilidades por gênero
5. As probabilidades de todos os segmentos e modelos são **somadas e normalizadas (ensemble)**
6. O gênero com maior probabilidade acumulada é exibido

```
Fase 1 — Treino
───────────────────────────────────────────────────────────────
GTZAN (1.000 × 30s)
    │
    ├─ build_dataset.py  →  segmenta em 3s, extrai features, salva em .npz
    │
    ├─ train_mlp.py  →  MLP (fingerprint 26-dim)  →  mlp.keras + scaler.pkl
    └─ train_cnn.py  →  CNN (mel-espectrograma)   →  cnn.keras + cnn_norm.npz
    │
    └─ compare.py  →  avalia ambos no mesmo teste, gera comparison.png


Fase 2 — Reconhecimento em Tempo Real
───────────────────────────────────────────────────────────────
Microfone (9s)
    │
    ├─ segmenta em 3 clipes de 3s
    │
    ├─ MLP:  fingerprint 26-dim  →  vetor de probabilidades por gênero
    └─ CNN:  mel-espectrograma   →  vetor de probabilidades por gênero
    │
    └─ média das probabilidades (ensemble)  →  gênero final
```

---

## Dataset — GTZAN

| Propriedade | Detalhe |
|---|---|
| **Fonte** | Kaggle — GTZAN Dataset (andradaolteanu) |
| **Total de clipes** | 1.000 arquivos `.wav` (30 segundos cada) |
| **Gêneros** | blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock |
| **Clipes por gênero** | 100 |
| **Nota** | `jazz.00054.wav` é corrompido no dataset original — ignorado automaticamente |

---

## Tecnologias

| Componente | Biblioteca | Papel |
|---|---|---|
| Extração de features | **librosa** | MFCC, Chroma, Centroid, Mel-espectrograma |
| Captura de áudio | **pyaudio** | Gravar do microfone em tempo real |
| Modelos neurais | **TensorFlow / Keras** | MLP e CNN |
| Normalização / split | **scikit-learn** | StandardScaler, GroupShuffleSplit |
| Serialização | **numpy**, **joblib** | Cache .npz, scaler .pkl |

---

## Arquitetura dos Modelos

### MLP (baseline)

```
Input (26,)         ← fingerprint: 13 MFCC + 12 Chroma + 1 Centroid
Dense(256) + BN + Dropout(0.3)
Dense(128) + BN + Dropout(0.3)
Dense(64) + Dropout(0.2)
Dense(10, softmax)  ← 10 gêneros
```

Treinamento: 300 épocas máx., Adam lr=1e-3, EarlyStopping(patience=30), ReduceLROnPlateau

### CNN

```
Input (128, 130, 1)  ← mel-espectrograma normalizado, canal único
Conv2D(32) + BN + MaxPool
Conv2D(64) + BN + MaxPool
Conv2D(128) + BN + MaxPool
Conv2D(128) + BN
GlobalAveragePooling2D
Dense(128) + Dropout(0.4)
Dense(10, softmax)   ← 10 gêneros
```

Treinamento: 120 épocas máx., Adam lr=1e-3, EarlyStopping(patience=20), ReduceLROnPlateau

---

## Estrutura do Projeto

```
tunes-rna/
├── audio_features.py      # Extração de features (compartilhado treino/inferência)
├── data_utils.py          # Carga do cache + split por música (sem data leakage)
├── download_dataset.py    # Baixa o GTZAN via kagglehub
├── build_dataset.py       # Extrai features dos .wav e gera o cache .npz
├── train_mlp.py           # Treino do MLP (fingerprint 26-dim)
├── train_cnn.py           # Treino da CNN (mel-espectrograma)
├── compare.py             # Avaliação comparativa MLP × CNN
├── test_mic.py            # Teste de captura e extração de features
├── recognize.py           # Reconhecimento de gênero em tempo real
├── requirements.txt       # Dependências do projeto
├── gtzan.ipynb            # Notebook exploratório (análise do dataset)
├── Data/                  # Dataset GTZAN (não versionado, ~2 GB)
├── models/                # Cache + modelos treinados (não versionado)
└── venv/                  # Ambiente virtual Python (não versionado)
```

Dependências entre os scripts:

```
download_dataset.py
    └─► build_dataset.py
            ├─► train_mlp.py ─┐
            └─► train_cnn.py ─┴─► compare.py
                                       └─► recognize.py
```

---

## Como Executar

### 1. Pré-requisitos

- Python 3.10+ (recomendado 3.11/3.12)
- Conta no [Kaggle](https://www.kaggle.com/) com API key configurada
- Microfone (para `test_mic.py` e `recognize.py`)

**Configurar credenciais do Kaggle:**

```bash
# Acesse https://www.kaggle.com/settings → API → "Create New Token"
# Isso baixa um arquivo kaggle.json. Coloque-o em:
# Windows:  C:\Users\<SEU_USUARIO>\.kaggle\kaggle.json
# Linux/Mac: ~/.kaggle/kaggle.json
```

### 2. Criar e ativar ambiente virtual

```bash
py -m venv venv
.\venv\Scripts\activate       # Windows PowerShell
# source venv/bin/activate    # Linux/Mac
```

### 3. Instalar dependências

```bash
pip install -r requirements.txt
```

### 4. Baixar o dataset (uma vez, ~2 GB)

```bash
python download_dataset.py
```

Usa `kagglehub` para baixar o GTZAN na pasta `Data/` (não versionada).

### 5. Construir o cache de features (uma vez, ~5–15 min)

```bash
python build_dataset.py
```

Lê os 1.000 `.wav`, divide em segmentos de 3s e extrai fingerprint + mel-espectrograma. Salva em `models/dataset.npz`.

### 6. Treinar os modelos

```bash
python train_mlp.py   # MLP — salva mlp.keras + scaler.pkl
python train_cnn.py   # CNN — salva cnn.keras + cnn_norm.npz
```

Podem ser executados em paralelo (terminais separados). O MLP treina mais rápido (~2–5 min); a CNN pode levar mais (~10–30 min dependendo do hardware).

### 7. Comparar os modelos

```bash
python compare.py
```

Avalia MLP e CNN no mesmo conjunto de teste e gera as matrizes de confusão em `models/comparison.png`.

### 8. Testar captura do microfone (opcional)

```bash
python test_mic.py
```

Grava 5 segundos e exibe o vetor de features extraído. Serve para confirmar que o microfone e a pipeline de extração funcionam antes de rodar o reconhecimento.

### 9. Reconhecer gênero em tempo real

```bash
python recognize.py
```

Grava 9 segundos, classifica com os modelos disponíveis e exibe o ranking de gêneros. Funciona com MLP sozinho, CNN sozinha ou ambos (ensemble automático).

---

## Saída esperada

### `build_dataset.py`

```
Dataset:  .../Data/.../genres_original
Extraindo features (pode levar alguns minutos)...

  blues       100 clipes ->  900 segmentos
  classical   100 clipes ->  900 segmentos
  ...
  rock        100 clipes ->  900 segmentos

Concluido em 480s.  8991 segmentos, 1 arquivos ignorados.
  X_mlp: (8991, 26)   X_cnn: (8991, 128, 130)
  Cache salvo em: .../models/dataset.npz  (450 MB)
```

### `recognize.py`

```
=======================================================
  Tunes-RNA - Reconhecimento de Genero Musical
=======================================================
Modelos carregados: MLP, CNN

Gravando por 9s... toque uma musica!
  [####################]  ok

=======================================================
  GENERO: ROCK  (72.4%)
=======================================================
  Ranking completo:
  rock        72.4%  ######################
  metal       14.1%  ####
  blues        6.3%  #
  country      2.8%
  ...
```

---

## Decisões de Design

| Decisão | Motivo |
|---|---|
| Split por música, não por segmento | Evita *data leakage*: segmentos de uma mesma faixa nunca caem em conjuntos diferentes |
| Módulo `audio_features.py` compartilhado | Garante que treino e inferência usam **exatamente** o mesmo pré-processamento |
| Normalização salva como artefato | `scaler.pkl` e `cnn_norm.npz` permitem aplicar na inferência os parâmetros do treino |
| Ensemble MLP + CNN | Médias de probabilidades reduzem variância e aumentam robustez |
| `jazz.00054.wav` ignorado | Arquivo corrompido no GTZAN original — tratado com `try/except` no `build_dataset.py` |

---

## Troubleshooting

**`PyAudio` não instala no Windows**
```bash
pip install pipwin
pipwin install pyaudio
```

**`No module named 'kagglehub'`**
```bash
pip install kagglehub
```

**`kaggle.json` não encontrado**
Crie a pasta `%USERPROFILE%\.kaggle\` e coloque o arquivo `kaggle.json` baixado das configurações da sua conta Kaggle.

**`jazz.00054.wav`: error reading file**
Normal — o arquivo é corrompido no dataset original. O `build_dataset.py` ignora automaticamente e registra no log.

**Erro de microfone no `recognize.py`**
Execute `python test_mic.py` primeiro para validar que o dispositivo de entrada está acessível.
