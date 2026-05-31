import numpy as np
import keras
from keras.layers import Dense, Dropout, BatchNormalization
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.regularizers import l2

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

import pickle
import matplotlib.pyplot as plt
import seaborn as sns


# ─────────────────────────────────────────────
#  ARQUITETURA DA MLP
# ─────────────────────────────────────────────

def build_model(num_attributes, num_classes, dropout_rate=0.4, l2_reg=1e-4):
    model = keras.Sequential([
        Dense(512, kernel_regularizer=l2(l2_reg), input_shape=(num_attributes,)),
        BatchNormalization(),
        keras.layers.Activation('relu'),
        Dropout(dropout_rate),

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

        Dense(num_classes, activation='softmax')
    ])

    # BatchNormalization: normaliza os numéros entre cada camada, para evitar valores muito fora.
    # DropOut: Durante o treino, desliga aleatoriamento a quantidade passada para todos neurônios "aprenderem".
    # Kernel_Regularizer: Penaliza pesos muito grandes, evita de um só neurônio dominar.
    # Activation ReLU: Função que decide se o neurônio ativa ou não. 0>n -> 0 não ativa. 0<n ativa.
    # Softmax tranforma os valores em porcentagens. 

    model.compile(
        optimizer=Adam(learning_rate=3e-4), # 0,0003
        loss='categorical_crossentropy',
        metrics=['accuracy', keras.metrics.F1Score(average="macro")]
    )

    # categorical_crossentropy: calc quão errado está... pune mais quando erra com certeza
    # accuracy: % de acertos simples.
    # F1Score: media de precision e recall. 

    return model


def get_callbacks():
    # earlystopping: para de treinar quando não tem avanço.
    # reducelronplateau: reduz o learning rate quando não tem avanço. 
    return [
        EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6, verbose=0)
    ]


# ─────────────────────────────────────────────
#  TREINAMENTO COM K-FOLD
# ─────────────────────────────────────────────

def train_rna(file_dataset, n_folds=5, save_path="model", plot=True):
    # Carrega dados
    data = np.load(file_dataset)
    mfcc, genres_raw = data['mfcc'], data['genres']

    print(f"📦 Dataset: {mfcc.shape[0]} amostras, {mfcc.shape[1]} features")
    # .shape[0] -> linhas
    # .shape[1] -> colunas
    
    # Encoding 
    encoder    = LabelEncoder()
    labels_int = encoder.fit_transform(genres_raw)   # ex: [0,1,2,...] (inteiros)
    labels_cat = to_categorical(labels_int)          # transforma para binário (000, 001, 010...)
    num_classes    = labels_cat.shape[1]
    num_attributes = mfcc.shape[1]

    print(f"🏷️ Classes ({num_classes}): {list(encoder.classes_)}")
    print(f"📐 Folds: {n_folds}  →  ~{len(mfcc)//n_folds} amostras por fold\n")

    # StratifiedKFold garante proporção igual de cada gênero em todos os folds
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    historicos      = []   # histórico de cada fold
    accuracy        = []   # val_accuracy de cada fold
    f1s             = []   # val_f1 de cada fold

    y_true_total = []
    y_pred_total = []

    melhor_acc    = 0.0
    melhor_modelo = None
    melhor_scaler = None

    # Loop KFold
    for fold, (idx_treino, idx_val) in enumerate(skf.split(mfcc, labels_int), start=1):

        print("─" * 60)
        print(f"  FOLD {fold}/{n_folds}")
        print(f"  Treino: {len(idx_treino)} amostras | Validação: {len(idx_val)} amostras")
        print("─" * 60)

        # Separa treino e validação deste fold
        x_treino, x_val = mfcc[idx_treino], mfcc[idx_val]
        y_treino, y_val = labels_cat[idx_treino], labels_cat[idx_val]

        # Treinado APENAS nos dados de treino do fold
        scaler = StandardScaler()
        x_treino_s = scaler.fit_transform(x_treino)
        x_val_s    = scaler.transform(x_val)

        # Modelo novo a cada fold
        model = build_model(num_attributes, num_classes)

        hist = model.fit(
            x_treino_s, y_treino,
            validation_data=(x_val_s, y_val),
            epochs=300,
            batch_size=32,
            callbacks=get_callbacks(),
            verbose=1
        )

        val_acc = max(hist.history['val_accuracy'])
        val_f1  = max(hist.history['val_f1_score']) if 'val_f1_score' in hist.history else 0.0

        accuracy.append(val_acc)
        f1s.append(val_f1)
        historicos.append(hist)

        # Acumula predições para a matriz de confusão global
        y_pred_prob  = model.predict(x_val_s, verbose=0)
        y_pred_fold  = np.argmax(y_pred_prob, axis=1)
        y_true_fold  = np.argmax(y_val,       axis=1)
        y_true_total.extend(y_true_fold)
        y_pred_total.extend(y_pred_fold)

        print(f"\n  ✔ Fold {fold} → val_accuracy: {val_acc*100:.2f}%  |  val_f1: {val_f1*100:.2f}%\n")

        # Melhor modelo entre os folds
        if val_acc > melhor_acc:
            melhor_acc    = val_acc
            melhor_modelo = model
            melhor_scaler = scaler

    # Resultado
    print("\n" + "=" * 60)
    print("              RESULTADO FINAL — K-FOLD")
    print("=" * 60)
    print(f"  Acurácia por fold: {[f'{a*100:.1f}%' for a in accuracy]}")
    print(f"  Média  : {np.mean(accuracy)*100:.2f}%")
    print(f"  Desvio : ±{np.std(accuracy)*100:.2f}%")
    print(f"  Melhor fold: {np.argmax(accuracy)+1}  ({melhor_acc*100:.2f}%)")
    print("=" * 60)

    print("\n📋 Relatório consolidado (todas as predições dos folds):")
    print(classification_report(y_true_total, y_pred_total, target_names=encoder.classes_))

    # Salva o modelo
    melhor_modelo.save(f"./{save_path}/tunes-rna.keras")
    with open(f"./{save_path}/scaler.pkl",        "wb") as f: pickle.dump(melhor_scaler, f)
    with open(f"./{save_path}/label_encoder.pkl", "wb") as f: pickle.dump(encoder,       f)

    print("💾 Melhor modelo salvo: tunes-rna.keras | scaler.pkl | label_encoder.pkl")

    if plot:
        _plot_kfold_accuracy(accuracy)
        _plot_histories(historicos)
        _plot_confusion_matrix(y_true_total, y_pred_total, encoder.classes_)

    return melhor_modelo


# ─────────────────────────────────────────────
#  VISUALIZAÇÕES
# ─────────────────────────────────────────────

def _plot_kfold_accuracy(accuracy):
    """Barra com a acurácia de cada fold + linha da média."""
    fig, ax = plt.subplots(figsize=(8, 4))
    folds = [f"Fold {i+1}" for i in range(len(accuracy))]
    bars  = ax.bar(folds, [a*100 for a in accuracy], color='steelblue', alpha=0.8)
    media = np.mean(accuracy) * 100
    ax.axhline(media, color='tomato', linestyle='--', linewidth=1.5, label=f'Média: {media:.1f}%')
    ax.set_ylabel("Val Accuracy (%)")
    ax.set_title("Acurácia por Fold — K-Fold Cross Validation")
    ax.set_ylim(0, 100)
    ax.legend()
    for bar, acc in zip(bars, accuracy):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{acc*100:.1f}%', ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    plt.savefig("./model/kfold_accuracy.png", dpi=120, bbox_inches='tight')
    print("📊 kfold_accuracy.png")
    plt.show()

def _plot_histories(history):
    """Curvas de loss de todos os folds sobrepostas."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Histórico de Treino — Todos os Folds", fontsize=13)

    for i, hist in enumerate(history):
        label = f"Fold {i+1}"
        axes[0].plot(hist.history['val_accuracy'], label=label, alpha=0.75)
        axes[1].plot(hist.history['val_loss'],     label=label, alpha=0.75)

    axes[0].set_title("Val Accuracy"); axes[0].set_xlabel("Época"); axes[0].legend()
    axes[1].set_title("Val Loss");     axes[1].set_xlabel("Época"); axes[1].legend()
    for ax in axes:
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("./model/history_folds.png", dpi=120, bbox_inches='tight')
    print("📊 history_folds.png")
    plt.show()

def _plot_confusion_matrix(y_true, y_pred, class_names):
    cm      = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Matriz de Confusão — Consolidada (todos os folds)", fontsize=13)

    for ax, data, fmt, title in zip(
        axes, [cm, cm_norm], ['d', '.2f'],
        ['Contagem Absoluta', 'Proporção (normalizada)']
    ):
        sns.heatmap(data, annot=True, fmt=fmt,
                    xticklabels=class_names, yticklabels=class_names,
                    cmap='Blues', ax=ax)
        ax.set_title(title); ax.set_xlabel("Previsto"); ax.set_ylabel("Real")

    plt.tight_layout()
    plt.savefig("./model/matriz_confusao.png", dpi=120, bbox_inches='tight')
    print("📊 matriz_confusao.png")
    plt.show()

if __name__ == "__main__":
    train_rna("./datasets/train_data.npz", n_folds=5)