import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['ABSL_LOGGING_MIN_LOG_LEVEL'] = '3'
os.environ["TF_ENABLE_ONEDNN_OPTS"]="0"

import tensorflow as tf
import logging
tf.get_logger().setLevel(logging.ERROR)

import modely
import predict
import process_data

print("""
SELECIONE A OPÇÃO:
      1 - Treinar modelo
      2 - Executar modelo
      3 - Processar novos datasets\n--> """,end="")

try:
    choose = int(input(""))
    print("-"*60)

    if choose == 1:
        modely.treinar_rede_neural("./datasets/dados_treino.npz")

    elif choose == 2:
        fsongs   = [f for f in os.listdir("songs")]
        for i, s in enumerate(fsongs):
            print(f" {i}: {s}")

        file_index = int(input("\nEscolha um arquivo de áudio: "))

        if file_index <= i+1:
            predict.main(f"./songs/{fsongs[file_index]}")
    
    elif choose == 3:
        base_dir_dataset = "./datasets/" + input("Nome da pasta do dataset: ./datasets/")
        process_data.process_all_dataset(base_dir_dataset)

    else:
        raise Exception
except Exception as e:
    print(f"Escolha inexistente! {e}")