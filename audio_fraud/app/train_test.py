
import psycopg2
import random
from src.predict import Predict
import pandas as pd

PATH = r"F:/Projetos/tcc/audio/release_in_the_wild/release_in_the_wild/"
def process_audio(file_path, label):
    prediction, _ = Predict.run_predict(PATH + file_path)  # Executa a função run_predict
    return prediction, label

if __name__ == "__main__":
    pass