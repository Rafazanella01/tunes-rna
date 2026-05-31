import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['ABSL_LOGGING_MIN_LOG_LEVEL'] = '3'
os.environ["TF_ENABLE_ONEDNN_OPTS"]="0"

import tensorflow as tf
import logging
tf.get_logger().setLevel(logging.ERROR)

from mlp import modely, predict, process_data

print("""
SELECIONE A OPÇÃO:
      1 - Treinar modelo
      2 - Executar modelo
      3 - Processar novos datasets\n--> """,end="")

try:
    choose = int(input(""))
    print("-"*60)

    # Treinar modelo
    if choose == 1:
        npz   = [n for n in os.listdir("datasets") if n.endswith(".npz")]
        for i, f in enumerate(npz):
            print(f" {i}: {f}")

        file_index = int(input("\nEscolha um arquivo de processamento: "))

        modely.train_rna(f"./datasets/{npz[file_index]}")

    # Executar modelo
    elif choose == 2:
        fsongs   = [f for f in os.listdir("songs")]
        for i, s in enumerate(fsongs):
            print(f" {i}: {s}")

        file_index = int(input("\nEscolha um arquivo de áudio: "))

        if file_index <= i+1:
            predict.main(f"./songs/{fsongs[file_index]}")
    
    # Processar datasets
    elif choose == 3:
        input_path = input("Nome da pasta do dataset: ./datasets/")
        base_dir_dataset = f"./datasets/{input_path}" if input_path else None 
        aug = int(input("Usar augmentation para aumentar as informações? (1 - True, 0 - False) -> "))
        
        process_data.main(base_dir_dataset, augmentation=aug)

    else:
        raise Exception
    
except Exception as e:
    print(f"Escolha inexistente! {e}")