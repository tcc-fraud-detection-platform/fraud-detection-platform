import random
import os
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.predict import Predict
import pandas as pd

PATH = r"C:/Users/lastg/Downloads/for-original/for-original/testing/real/"
def process_audio(file_path, label):
    try:
        prediction, _ = Predict.run_predict(PATH + file_path)  # Executa a função run_predict
    except Exception as e:
        # Em caso de erro, salve os dados de teste e levante a exceção novamente
        save_error_data(file_path, label)
        raise e
    return prediction, label

def save_error_data(file_path, label):
    with open('error_log.txt', 'a') as file:
        file.write(f"Error processing audio: {file_path}, Label: {label}\n")

if __name__ == "__main__":
    start_time = time.time() 

    audio_files = os.listdir(PATH)  # Lista todos os arquivos de áudio na pasta com a label 'spoof'
    print(audio_files)

    results = []  # Lista para armazenar os resultados de processamento

    for file_path in audio_files:
        label = 'bona-fide'  # Define a label para cada arquivo como 'spoof'
        result = process_audio(file_path, label)
        results.append(result)

    y_true = [result[1] for result in results]  # Extrai as labels verdadeiras
    y_pred = [result[0] for result in results] 

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, pos_label='spoof')
    recall = recall_score(y_true, y_pred, pos_label='spoof')
    f1 = f1_score(y_true, y_pred, pos_label='spoof')


    # Calcula a porcentagem de acertos comparando as previsões com os rótulos reais
    correct_predictions = sum(1 for result in results if result[0] == result[1])
    total_files = len(results)

    end_time = time.time()  # Captura o tempo de término da execução
    execution_time = end_time - start_time  # Calcula o tempo de execução
    print(f"Tempo de execução: {execution_time:.2f} segundos")
    print(f"Acurácia: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1-score: {f1:.2f}")

    print(f"correct_predictions: {correct_predictions}")
    print(f"total_files: {total_files}")

    with open('test_results.txt', 'w') as file:
        file.write(f"Acurácia: {accuracy:.2f}%\n")
        file.write(f"Precision: {precision:.2f}\n")
        file.write(f"Recall: {recall:.2f}\n")
        file.write(f"F1-score: {f1:.2f}\n")
        file.write(f"Tempo de execução: {execution_time:.2f} segundos\n")
        file.write(f"correct_predictions: {correct_predictions}\n")
        file.write(f"total_files: {total_files}\n")