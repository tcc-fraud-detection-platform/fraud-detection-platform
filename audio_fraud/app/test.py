

import psycopg2
import random
from src.predict import Predict
import pandas as pd

PATH = r"F:/Projetos/tcc/audio/release_in_the_wild/release_in_the_wild/"
def process_audio(file_path, label):
    prediction, _ = Predict.run_predict(PATH + file_path)  # Executa a função run_predict
    return prediction, label

if __name__ == "__main__":
    # Carrega o arquivo meta.csv
    meta_data = pd.read_csv(PATH + 'meta.csv', nrows=2)

    # Aplica a função process_audio a cada linha do DataFrame
    results = meta_data.apply(lambda row: process_audio(row['file'], row['label']), axis=1)

    # Calcula a porcentagem de acertos comparando as previsões com os rótulos reais
    correct_predictions = sum(1 for result in results if result[0] == result[1])
    total_files = len(results)
    accuracy = correct_predictions / total_files * 100

    print(f"Acurácia: {accuracy:.2f}%")