import pickle
import torch
import transformers as ppb
import psycopg2
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from decouple import config

DB_NAME = config('DB_NAME')
DB_USER = config('DB_USER')
DB_PASSWORD = config('DB_PASSWORD')
DB_HOST = config('DB_HOST')
DB_PORT = config('DB_PORT')
# import re

# Função para pré-processar os dados de entrada para previsão
def preprocess_with_bert_for_prediction(text):
    tokenized = tokenizer.encode(text, add_special_tokens=True)
    if len(tokenized) > max_seq_length:
        tokenized = tokenized[:max_seq_length]
    padded = tokenized + [0] * (max_seq_length - len(tokenized))
    input_ids = torch.tensor([padded]).to(device)
    return input_ids

# Função para fazer previsões com o modelo BERT
def predict_with_bert(text):
    input_ids = preprocess_with_bert_for_prediction(text)
    with torch.no_grad():
        outputs = model(input_ids.to(device))
    last_hidden_states = outputs.last_hidden_state
    embeddings = last_hidden_states.cpu().numpy().squeeze()
    prediction = classifier.predict(embeddings.reshape(1, -1))
    probabilities = classifier.predict_proba(embeddings.reshape(1, -1))
    predicted_label = prediction[0]
    confidence = probabilities.max()
    if predicted_label == 0:
        predicted_text = "Fake News"
    elif predicted_label == 1:
        predicted_text = "Notícia verdadeira"

    print(f"Previsão: {predicted_text}, Confiança: {confidence*100:.2f}%")
    return predicted_label, confidence

##connection
conn = psycopg2.connect(dbname=DB_NAME,
                        user=DB_USER,
                        password=DB_PASSWORD,
                        host=DB_HOST,
                        port=DB_PORT)
cur = conn.cursor()
select_query = """
    SELECT 
        * 
    FROM
        public.tcc_2
    WHERE
            was_processed is false 
        AND
            ai_flow = 'fake_news_analysis'
"""
cur.execute(select_query)
rows = cur.fetchall()
for row in rows:

    memory_view = row[1]  # Accessing the memory view object
    byte_data = memory_view.tobytes()  # Convert memory view to bytes  
    # Decode if necessary, e.g., to a UTF-8 string
    news = byte_data.decode('utf-8')

    # Carregar o modelo treinado
    model_file = './models/WELFake_ensemble_model.sav'
    classifier = pickle.load(open(model_file, 'rb'))

    tokenizer = ppb.BertTokenizer.from_pretrained('./bert-base-uncased-tokenizer')
    model = ppb.BertModel.from_pretrained('./bert-base-uncased-model')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)

    max_seq_length = 128

    # get from DB
    prediction_label, confidence = predict_with_bert(news)
    print('<<{"result": '+ str(prediction_label) + ', "score": '+ str(confidence) + '}>>')
    update_query = f"""
    UPDATE 
        public.tcc_2 
    SET 
        was_processed = true,
        score = {confidence},
        is_real = {prediction_label}
    WHERE     
            was_processed is false 
        AND
            ai_flow = 'fake_news_analysis'
        AND 
            operation_id =  {row[3]}
            ;
    """
    cur.execute(update_query)
    conn.commit()