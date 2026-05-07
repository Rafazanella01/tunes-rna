# Tunes RNA - Reconhecimento de Áudio

**Trabalho 2 - Cadeira de Inteligência Artificial (UNISC)**

Este projeto consiste em um sistema de reconhecimento de áudio inspirado no funcionamento do Shazam, utilizando **Redes Neurais Artificiais (RNA)** em conjunto com processamento de sinais clássico.

## 🧠 A Ideia do Projeto

O objetivo principal é identificar músicas ou trechos de áudio comparando-os com uma base de dados local. Para atingir esse objetivo, o projeto combina duas tecnologias principais:

1. **Extração de Características (Fingerprinting Clássico):**
   Utilizaremos a biblioteca **Chromaprint** para processar o arquivo de áudio bruto e extrair um "fingerprint" (uma assinatura digital compacta baseada em frequências). O Chromaprint transforma o som em dados estruturados.

2. **Reconhecimento com RNA:**
   Em vez de utilizar buscas simples em banco de dados, os fingerprints gerados servirão como entrada para uma **Rede Neural Artificial**. A RNA será treinada/utilizada para analisar esses dados, aprender os padrões e realizar a classificação ou correspondência com o acervo musical armazenado.

## 🛠️ Tecnologias e Bibliotecas

- **Python 3.x**
- **Chromaprint (`chromaprint`)**: Responsável por gerar o fingerprint do áudio.
- *(Outras bibliotecas de IA/RNA serão adicionadas em breve, como TensorFlow/Keras ou PyTorch)*

## 🚀 Como Executar

1. Clone o repositório.
2. Ative o ambiente virtual: `.\venv\Scripts\activate` (Windows)
3. Instale as dependências: `pip install -r requirements.txt`