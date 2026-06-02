# 🎵 Tunes-RNA — Classificação de Gênero Musical com Rede Neural Artificial

> **Trabalho 2 — Cadeira de Inteligência Artificial | UNISC**

Sistema de **reconhecimento e classificação de gênero musical** a partir de arquivos de áudio, utilizando uma **Rede Neural Artificial MLP (Multi-Layer Perceptron)** treinada com features acústicas extraídas pelo `librosa`.

---

## 📋 Índice

- [Como Funciona](#-como-funciona)
- [Arquitetura da RNA](#-arquitetura-da-rna)
- [Features Acústicas Extraídas](#-features-acústicas-extraídas)
- [Data Augmentation](#-data-augmentation)
- [Dataset — GTZAN](#-dataset--gtzan)
- [Estrutura do Projeto](#-estrutura-do-projeto)
- [Tecnologias e Bibliotecas](#%EF%B8%8F-tecnologias-e-bibliotecas)
- [Como Executar](#-como-executar)
- [Fluxo Completo de Uso](#-fluxo-completo-de-uso)
- [Gêneros Suportados](#-gêneros-suportados)

---

## 🧠 Como Funciona

O sistema é dividido em **três fases**:

### Fase 1 — Processamento do Dataset
1. O dataset **GTZAN** é baixado do Kaggle (1.000 clipes `.wav`, 10 gêneros)
2. Cada clipe de 30 segundos tem **~301 features acústicas** extraídas com `librosa`
3. Opcionalmente, **Data Augmentation** é aplicada, gerando até **6x** mais amostras
4. Os dados são salvos comprimidos em `.npz` para uso no treinamento

### Fase 2 — Treinamento da RNA
1. A RNA MLP é treinada com **K-Fold Cross Validation** (5 folds)
2. O melhor modelo é salvo junto com o `StandardScaler` e o `LabelEncoder`
3. Gráficos de desempenho são gerados automaticamente

### Fase 3 — Predição
1. Um arquivo de áudio (`.mp3`, `.wav`, etc.) é fornecido
2. As mesmas features são extraídas do áudio
3. A RNA retorna o **gênero musical identificado** com ranking de confiança

```
🎵 Arquivo de Áudio (mp3/wav)
        ↓
librosa (extrai ~301 features acústicas)
        ↓
StandardScaler (normalização)
        ↓
🧠 RNA MLP (4 camadas ocultas: 512 → 256 → 128 → 64)
        ↓
Softmax (probabilidades por gênero)
        ↓
🎸 "rock" 87.3%  |  🎤 "hiphop" 7.1%  |  🎵 "pop" 3.2%
```

---

## 🏗️ Arquitetura da RNA

A MLP foi construída com **Keras/TensorFlow** e possui a seguinte arquitetura:

| Camada | Neurônios | Ativação | Regularização |
|--------|-----------|----------|---------------|
| Entrada | ~301 | — | — |
| Oculta 1 | 512 | ReLU | L2 + BatchNorm + Dropout(0.4) |
| Oculta 2 | 256 | ReLU | L2 + BatchNorm + Dropout(0.4) |
| Oculta 3 | 128 | ReLU | L2 + BatchNorm + Dropout(0.4) |
| Oculta 4 | 64 | ReLU | L2 + BatchNorm + Dropout(0.2) |
| Saída | 10 | Softmax | — |

**Configurações de treinamento:**
- **Otimizador:** Adam (lr = 3e-4)
- **Loss:** Categorical Crossentropy
- **Métricas:** Accuracy + F1-Score (macro)
- **Épocas máximas:** 300
- **Batch size:** 32
- **Early Stopping:** patience=25 (monitora `val_loss`)
- **ReduceLROnPlateau:** patience=10, fator=0.5, min_lr=1e-6
- **K-Fold:** 5 folds estratificados (StratifiedKFold)

---

## 🔊 Features Acústicas Extraídas

Cada clipe de áudio gera um vetor de **~301 features**, combinando média e desvio padrão de cada característica ao longo do tempo:

| Feature | Dimensões | Descrição |
|---------|-----------|-----------|
| **MFCC** (40 coef.) | 80 | Timbre e textura sonora |
| **Delta MFCC** | 80 | Variação temporal dos MFCCs |
| **Delta² MFCC** | 80 | Aceleração da variação dos MFCCs |
| **Chroma STFT** (12) | 24 | Notas musicais presentes (Dó, Ré, Mi...) |
| **Spectral Contrast** (7 bandas) | 14 | Picos e vales espectrais |
| **Spectral Centroid** | 2 | "Brilho" do som |
| **Spectral Bandwidth** | 2 | Largura de banda espectral |
| **Spectral Rolloff** | 2 | Frequência de rolloff |
| **Zero Crossing Rate** | 2 | Taxa de cruzamento por zero |
| **RMS Energy** | 2 | Volume médio do áudio |
| **Tonnetz** (6) | 12 | Relações harmônicas tonais |
| **Tempo (BPM)** | 1 | Batidas por minuto |
| **Total** | **~301** | |

---

## 🔀 Data Augmentation

Quando ativada, cada arquivo original gera **5 variações adicionais** (total = 6x o dataset):

| Técnica | Descrição |
|---------|-----------|
| **Time Stretch (lento)** | Comprime o tempo em 85% sem alterar o tom |
| **Time Stretch (rápido)** | Expande o tempo em 115% sem alterar o tom |
| **Pitch Shift (−2 semitons)** | Baixa o tom 2 semitons sem alterar velocidade |
| **Pitch Shift (+2 semitons)** | Sobe o tom 2 semitons sem alterar velocidade |
| **Gaussian Noise** | Adiciona ruído leve (noise_factor=0.005) |

---

## 🎼 Dataset — GTZAN

| Propriedade | Detalhe |
|-------------|---------|
| **Fonte** | [Kaggle — GTZAN Dataset](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification) |
| **Total de clipes** | 1.000 arquivos `.wav` (30 segundos cada) |
| **Gêneros** | blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock |
| **Clipes por gênero** | 100 |
| **Taxa de amostragem** | 22.050 Hz |
| **Com augmentation** | ~6.000 amostras (6x) |

Outros datasets disponíveis via `kaggle.py`:

| Dataset | Função | Kaggle ID |
|---------|--------|-----------|
| GTZAN | `gtzan()` | `andradaolteanu/gtzan-dataset-music-genre-classification` |
| Neural Audio Fingerprint | `neuralaudio()` | `mimbres/neural-audio-fingerprint` |
| Spotify 3-sec Mel Spectrograms | `spotify3sec()` | `mrodriguez2/spotify-3-second-mel-spectrograms` |
| FMA (Free Music Archive) | `fma()` | `iamsparsh/fma-free-music-archive-small-medium` |

---

## 📁 Estrutura do Projeto

```
tunes-rna/
├── main.py                    # Ponto de entrada principal (menu interativo)
├── kaggle.py                  # Download de datasets do Kaggle
├── test_mic.py                # Teste de fingerprint via microfone (standalone)
├── requirements.txt           # Dependências do projeto
│
├── mlp/                       # Módulo principal da RNA
│   ├── __init__.py
│   ├── process_data.py        # Extração de features acústicas (~301 dimensões)
│   ├── augmentation.py        # Data Augmentation (time stretch, pitch shift, noise)
│   ├── modely.py              # Arquitetura MLP, treino K-Fold, visualizações
│   └── predict.py             # Predição de gênero a partir de arquivo de áudio
│
├── songs/                     # Músicas de teste para predição
│   ├── 50cent-indaclub.mp3
│   ├── avengedsevenfold-hailtotheking.mp3
│   ├── eriksatie-gymnopedieno.mp3
│   ├── johncoltrane-giantsteps.mp3
│   └── ... (11 arquivos mp3)
│
├── datasets/                  # Dataset GTZAN (NÃO versionado — .gitignore)
│   ├── gtzan-data/            # Arquivos .wav por gênero
│   └── dados_treino.npz       # Features extraídas (gerado pelo main.py)
│
├── model/                     # Modelo treinado (NÃO versionado — .gitignore)
│   ├── tunes-rna.keras        # Melhor modelo salvo
│   ├── scaler.pkl             # StandardScaler serializado
│   ├── label_encoder.pkl      # LabelEncoder serializado
│   ├── kfold_accuracy.png     # Gráfico de acurácia por fold
│   ├── history_folds.png      # Curvas de treinamento
│   └── matriz_confusao.png    # Matriz de confusão consolidada
│
├── mlp.png                    # Diagrama da arquitetura MLP
├── mlp_tunes.png              # Diagrama visual do projeto
└── README.md
```

> **Nota:** As pastas `datasets/`, `model/` e `venv/` são ignoradas pelo Git (`.gitignore`).

---

## 🛠️ Tecnologias e Bibliotecas

| Biblioteca | Versão | Uso |
|------------|--------|-----|
| **Python** | 3.10+ | Linguagem principal |
| **TensorFlow / Keras** | — | Construção e treinamento da RNA MLP |
| **librosa** | — | Extração de features acústicas (MFCC, Chroma, etc.) |
| **numpy** | — | Manipulação de arrays numéricos |
| **scikit-learn** | — | K-Fold, StandardScaler, LabelEncoder, métricas |
| **matplotlib** | — | Visualização de gráficos de treinamento |
| **seaborn** | — | Matriz de confusão com heatmap |
| **pyaudio** | — | Captura de áudio pelo microfone (test_mic.py) |
| **kagglehub** | — | Download automático de datasets do Kaggle |

---

## 🚀 Como Executar

### Pré-requisitos

- Python 3.10 ou superior
- Conta no [Kaggle](https://kaggle.com) com API token configurado (para download do dataset)
- Microfone (apenas para `test_mic.py`)

### 1. Clonar o repositório

```bash
git clone https://github.com/Rafazanella01/tunes-rna.git
cd tunes-rna
```

### 2. Criar e ativar ambiente virtual

```bash
# Windows
py -m venv venv
.\venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Instalar dependências

```bash
pip install -r requirements.txt
```

> ⚠️ **PyAudio no Windows:** Se houver erro ao instalar `pyaudio`, use:
> ```bash
> pip install pipwin
> pipwin install pyaudio
> ```

### 4. Configurar a API do Kaggle

1. Acesse [kaggle.com/settings](https://www.kaggle.com/settings) → **API** → **Create New Token**
2. Coloque o arquivo `kaggle.json` gerado em:
   - **Windows:** `C:\Users\<seu-usuário>\.kaggle\kaggle.json`
   - **Linux/macOS:** `~/.kaggle/kaggle.json`

---

## 🔄 Fluxo Completo de Uso

Execute o sistema pelo menu principal:

```bash
python main.py
```

```
SELECIONE A OPÇÃO:
      1 - Processar novos datasets
      2 - Treinar modelo
      3 - Executar modelo
--> 
```

---

### Opção 1 — Processar Dataset

Extrai features de todos os arquivos `.wav` do dataset e salva em `.npz`:

```
--> 1
Nome da pasta do dataset: ./datasets/  gtzan-data/genres_original
Usar augmentation para aumentar as informações? (1 - True, 0 - False) -> 1
```

**O que acontece internamente:**
- Percorre cada subpasta (gênero musical) dentro do caminho informado
- Para cada `.wav`, extrai ~301 features com `librosa`
- Se `augmentation=True`, gera 5 variações por arquivo (6x o total)
- Salva tudo em `./datasets/dados_treino.npz`

**Saída esperada:**
```
📁 Gênero: blues        (100 arquivos)
   ✔ blues.00000.wav  →  6 amostras
   ✔ blues.00001.wav  →  6 amostras
   ...
✅ 'dados_treino.npz' gerado com sucesso!
```

> 💡 Para baixar o dataset GTZAN automaticamente, edite `kaggle.py` e execute:
> ```python
> from kaggle import gtzan
> gtzan("./datasets/gtzan-data")
> ```

---

### Opção 2 — Treinar Modelo

Treina a RNA MLP com K-Fold Cross Validation sobre o `.npz` gerado:

```
--> 2
 0: dados_treino.npz

Escolha um arquivo de processamento: 0
```

**O que acontece internamente:**
- Carrega o `.npz` com features e labels
- Aplica `LabelEncoder` e `to_categorical` nas classes
- Executa 5 folds estratificados (StratifiedKFold)
- Em cada fold: normaliza com `StandardScaler`, treina e avalia a MLP
- Salva o melhor modelo (maior `val_accuracy`) em `./model/`
- Gera 3 gráficos automaticamente:
  - `kfold_accuracy.png` — Acurácia por fold + linha da média
  - `history_folds.png` — Curvas de Loss e Accuracy de todos os folds
  - `matriz_confusao.png` — Matriz de confusão consolidada (todos os folds)

**Saída esperada:**
```
📦 Dataset: 6000 amostras, 301 features
🏷️ Classes (10): ['blues', 'classical', 'country', ...]
📐 Folds: 5  →  ~1200 amostras por fold

────────────────────────────────────────────────────────────
  FOLD 1/5
  Treino: 4800 amostras | Validação: 1200 amostras
────────────────────────────────────────────────────────────
...
  ✔ Fold 1 → val_accuracy: 83.50%  |  val_f1: 82.91%

============================================================
              RESULTADO FINAL — K-FOLD
============================================================
  Acurácia por fold: ['83.5%', '85.2%', '82.8%', '84.1%', '86.0%']
  Média  : 84.32%
  Desvio : ±1.14%
  Melhor fold: 5  (86.00%)
============================================================
💾 Melhor modelo salvo: tunes-rna.keras | scaler.pkl | label_encoder.pkl
```

---

### Opção 3 — Executar Predição

Classifica o gênero de uma música da pasta `songs/`:

```
--> 3
 0: 50cent-indaclub.mp3
 1: avengedsevenfold-hailtotheking.mp3
 2: eriksatie-gymnopedieno.mp3
 ...

Escolha um arquivo de áudio: 0
```

**O que acontece internamente:**
- Carrega o modelo `tunes-rna.keras`, `scaler.pkl` e `label_encoder.pkl`
- Extrai ~301 features do áudio selecionado (primeiros 60s)
- Normaliza com o `StandardScaler` salvo
- A RNA retorna probabilidades para os 10 gêneros
- Exibe o **Top 3** com barra de confiança

**Saída esperada:**
```
🎵 Carregando áudio: ./songs/50cent-indaclub.mp3
✔  301 features extraídas

══════════════════════════════════════════════════
          RESULTADO DA PREDIÇÃO
══════════════════════════════════════════════════
  Arquivo : 50cent-indaclub.mp3
  Gênero  : 🎤  HIPHOP
  Confiança: 87.3%
──────────────────────────────────────────────────
  Ranking completo:
  1. hiphop       ██████████████████████████   87.3%
  2. pop          ██                            5.8%
  3. disco        █                             3.2%
══════════════════════════════════════════════════
```

---

### Teste de Fingerprint via Microfone (standalone)

Captura 5 segundos do microfone e exibe o fingerprint acústico gerado:

```bash
python test_mic.py
```

```
🎤 Gravando por 5 segundos... Fale ou toque algo!
  [████████████████████] 100%
✅ Gravação finalizada!

🔍 Extraindo features acústicas (MFCC + Chroma + Centroid)...

══════════════════════════════════════════════════
  📄 Fingerprint Gerado:
══════════════════════════════════════════════════
  MFCC (13 coef.)    : [-312.5   87.3  -10.2 ...]
  Chroma (12 notas)  : [0.423  0.312  0.285 ...]
  Centroid espectral : 2041.58 Hz

  🔢 Vetor final (26 dimensões):
  [-312.5  87.3  ...]

✨ Sucesso! Este vetor é o fingerprint que alimentará a RNA.
```

> **Nota:** O `test_mic.py` usa um vetor simplificado de **26 dimensões** (MFCC 13 + Chroma 12 + Centroid 1). O sistema completo usa **~301 features** para maior precisão.

---

## 🎸 Gêneros Suportados

| Gênero | Emoji | Exemplos na pasta `songs/` |
|--------|-------|---------------------------|
| Blues | 🎸 | gotmymojo-atsuper.mp3 |
| Classical | 🎻 | eriksatie-gymnopedieno.mp3 |
| Country | 🤠 | johndever-countryroads.mp3, lukebryan-countrysongcameon.mp3 |
| Disco | 🕺 | — |
| Hip-Hop | 🎤 | 50cent-indaclub.mp3, matue-cidade2000.mp3 |
| Jazz | 🎷 | johncoltrane-giantsteps.mp3 |
| Metal | 🤘 | avengedsevenfold-hailtotheking.mp3 |
| Pop | 🎵 | zaralarson.mp3, jackson5.mp3 |
| Reggae | 🌴 | — |
| Rock | 🎸 | thecure-boysdontcry.mp3 |

---

## 📊 Arquivos Gerados

| Arquivo | Localização | Descrição |
|---------|-------------|-----------|
| `dados_treino.npz` | `./datasets/` | Features + labels comprimidos |
| `tunes-rna.keras` | `./model/` | Melhor modelo treinado |
| `scaler.pkl` | `./model/` | StandardScaler para normalização |
| `label_encoder.pkl` | `./model/` | Encoder de gêneros musicais |
| `kfold_accuracy.png` | `./model/` | Gráfico de acurácia por fold |
| `history_folds.png` | `./model/` | Curvas de treino de todos os folds |
| `matriz_confusao.png` | `./model/` | Matriz de confusão consolidada |

> ⚠️ Todos esses arquivos estão no `.gitignore` e não são versionados.

---

## 🐛 Problemas Comuns

| Problema | Solução |
|----------|---------|
| `ModuleNotFoundError: pyaudio` | `pipwin install pyaudio` (Windows) |
| `FileNotFoundError: tunes-rna.keras` | Execute as opções 1 e 2 antes da 3 |
| `kaggle.json not found` | Configure a API do Kaggle corretamente |
| Erro ao carregar `.wav` corrompido | O `process_data.py` captura exceções e continua |
| Confiança baixa (< 50%) | Áudio atípico ou gênero pouco representado no dataset |

---

## 👨‍💻 Autor

**Rafael Zanella** — [@Rafazanella01](https://github.com/Rafazanella01)

Projeto desenvolvido para a cadeira de **Inteligência Artificial** da **UNISC (Universidade de Santa Cruz do Sul)**.