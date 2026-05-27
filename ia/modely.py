import numpy as np
#import tensorflow as tf
import keras
from keras.layers import Dense, Dropout, BatchNormalization
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.regularizers import l2

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

import pickle
import matplotlib.pyplot as plt
import seaborn as sns

def build_model(num_attributes, num_classes, dropout_rate=0.4, l2_reg=1e-4):
    """
    MLP com BatchNormalization e regularização L2.
 
    Mudanças em relação ao original:
      - BatchNorm após cada Dense → treino mais estável
      - L2 regularization         → penaliza pesos grandes (menos overfitting)
      - Dropout levemente maior   → mais generalização
      - Camada extra 256 neurônios→ mais capacidade para ~300 features
    """
    model = keras.Sequential([
        # ── Entrada ────────────────────────────────
        Dense(512, kernel_regularizer=l2(l2_reg), input_shape=(num_attributes,)),
        BatchNormalization(),
        keras.layers.Activation('relu'),
        Dropout(dropout_rate),
 
        # ── Camadas Ocultas ────────────────────────
        Dense(256, kernel_regularizer=l2(l2_reg)),
        BatchNormalization(),
        keras.layers.Activation('relu'),
        Dropout(dropout_rate),
 
        Dense(128, kernel_regularizer=l2(l2_reg)),
        BatchNormalization(),
        keras.layers.Activation('relu'),
        Dropout(dropout_rate - 0.1),
 
        Dense(64, kernel_regularizer=l2(l2_reg)),
        BatchNormalization(),
        keras.layers.Activation('relu'),
        Dropout(0.2),
 
        # ── Saída ──────────────────────────────────
        Dense(num_classes, activation='softmax')
    ])
 
    model.compile(
        optimizer=Adam(learning_rate=3e-4),       # LR inicial levemente maior
        loss='categorical_crossentropy',
        metrics=[
            'accuracy',
            keras.metrics.F1Score(average="macro")
        ]
    )
 
    return model

def get_callbacks():
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=25,                  # Mais paciência pois o LR vai cair
        restore_best_weights=True,
        verbose=1
    )
 
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,                   # Divide o LR por 2 quando estagna
        patience=10,
        min_lr=1e-6,
        verbose=1
    )
 
    return [early_stop, reduce_lr]

def _plot_history(hist):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Histórico de Treinamento", fontsize=14)
 
    # Acurácia
    axes[0].plot(hist.history['accuracy'],     label='Treino')
    axes[0].plot(hist.history['val_accuracy'], label='Validação')
    axes[0].set_title("Acurácia")
    axes[0].set_xlabel("Época")
    axes[0].set_ylabel("Acurácia")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
 
    # Loss
    axes[1].plot(hist.history['loss'],     label='Treino')
    axes[1].plot(hist.history['val_loss'], label='Validação')
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Época")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
 
    plt.tight_layout()
    plt.savefig("./historico_treino.png", dpi=120, bbox_inches='tight')
    print("📊 Gráfico salvo: historico_treino.png")
    plt.show()
 
def _plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)  # Normalizada
 
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Matriz de Confusão", fontsize=14)
 
    for ax, data, fmt, title in zip(
        axes,
        [cm, cm_norm],
        ['d', '.2f'],
        ['Contagem Absoluta', 'Proporção (normalizada)']
    ):
        sns.heatmap(
            data,
            annot=True, fmt=fmt,
            xticklabels=class_names, yticklabels=class_names,
            cmap='Blues', ax=ax
        )
        ax.set_title(title)
        ax.set_xlabel("Previsto")
        ax.set_ylabel("Real")
 
    plt.tight_layout()
    plt.savefig("./matriz_confusao.png", dpi=120, bbox_inches='tight')
    print("📊 Gráfico salvo: matriz_confusao.png")
    plt.show()

def treinar_rede_neural(file_dataset, camadas_ocultas=None, taxa_aprendizado=None, max_iteracoes=None):
    # Carrega os vetores numéricos e o espectrograma real extraído da sua função
    data = np.load(file_dataset)
    mfcc_data, genres_raw = data['mfcc'], data['genres']
    
    # Transforma os textos dos gêneros em números identificáveis pela rede
    encoder = LabelEncoder()
    genres_int = encoder.fit_transform(genres_raw)
    encoded_genres_cat = to_categorical(genres_int)

    num_classes     = encoded_genres_cat.shape[1] # Qnt de classes do MLP
    num_attributes  = mfcc_data.shape[1] # Qnt de coeficientes do MFCC

    # Separação clássica: 80% treino / 20% teste
    mfcc_training, mfcc_test, genres_training, genres_test = train_test_split(
        mfcc_data, 
        encoded_genres_cat, 
        test_size=0.2, 
        random_state=42,
        stratify=genres_int
    )
    
    # Escalonamento para a MLP convergir sem estourar gradiente -> deixa os valores mais padronizados...
    scaler = StandardScaler()
    mfcc_training_scaled = scaler.fit_transform(mfcc_training) # Aprende e tranforma o dados
    mfcc_test_scaled     = scaler.transform(mfcc_test) # Apenas transforma os dados

    model = build_model(num_attributes, num_classes)
    model.summary()

    print("🧠 Treinando a Rede Neural com TensorFlow...")

    hist = model.fit(
        mfcc_training_scaled, genres_training, 
        validation_data=(mfcc_test_scaled, genres_test),
        epochs=300, # Número de iterações
        batch_size=32,
        callbacks=get_callbacks(),
        verbose=1 # Mostra o erro sumindo no terminal linha por linha
    )

    y_pred_prob = model.predict(mfcc_test_scaled)
    y_pred      = np.argmax(y_pred_prob, axis=1)
    y_true      = np.argmax(genres_test,      axis=1)
 
    print("\n" + "="*60)
    print("                RESULTADOS FINAIS")
    print("="*60)
    print(classification_report(
        y_true, y_pred,
        target_names=encoder.classes_
    ))
 
    val_acc = hist.history['val_accuracy']
    print(f"Melhor val_accuracy:  {max(val_acc)*100:.2f}%")
    print(f"Última val_accuracy:  {val_acc[-1]*100:.2f}%")
    print(f"Épocas executadas:    {len(hist.history['loss'])}")
    print("="*60)
 
    # ── Salva modelo e scaler ──────────────────
    model.save("./tunes-rna.keras")
    with open("./scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open("./label_encoder.pkl", "wb") as f:
        pickle.dump(encoder, f)
 
    print("\n💾 Arquivos salvos: tunes-rna.keras | scaler.pkl | label_encoder.pkl")
 
    # ── Gráficos ───────────────────────────────
    _plot_history(hist)
    _plot_confusion_matrix(y_true, y_pred, encoder.classes_)
 
    return model, hist