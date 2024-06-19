import numpy as np
import pandas as pd
import pickle
import torch
import transformers as ppb  # pytorch transformers
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Carregar os dados de treinamento do arquivo CSV
train_filename = './welfake_dataset/WELFake_Dataset.csv'
train_news = pd.read_csv(train_filename)

# Verificar se CUDA está disponível e definir o dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Carregar o modelo pré-treinado BERT
model_class, tokenizer_class, pretrained_weights = (ppb.BertModel, ppb.BertTokenizer, 'bert-base-uncased')
tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
model = model_class.from_pretrained(pretrained_weights)
model.to(device) 

# Definir o tamanho máximo da sequência
max_seq_length = 128

# Função para pré-processar os dados de texto usando BERT
def preprocess_with_bert(text):
    tokenized = tokenizer.encode(text, add_special_tokens=True)
    if len(tokenized) > max_seq_length:
        tokenized = tokenized[:max_seq_length]
    padded = tokenized + [0] * (max_seq_length - len(tokenized))
    input_ids = torch.tensor([padded]).to(device)
    return input_ids

train_news = train_news.dropna(subset=['Statement']).reset_index(drop=True)
train_news['Statement'] = train_news['Statement'].apply(lambda x: x[:max_seq_length] if len(x) > max_seq_length else x)

# Aplicar a função de pré-processamento aos dados de treinamento
train_inputs = train_news['Statement'].apply(preprocess_with_bert)

train_embeddings = []
for i, input_ids in enumerate(train_inputs):
    logger.info(f"Processing training example {i+1}/{len(train_inputs)}")
    with torch.no_grad():
        outputs = model(input_ids)
    last_hidden_states = outputs.last_hidden_state
    train_embeddings.append(last_hidden_states.cpu().numpy())

# Converter embeddings para numpy arrays e achatar para 2Df
X_train = np.vstack(train_embeddings).reshape(len(train_embeddings), -1)

# Preparar os rótulos
y_train = train_news['Label']

X_train = X_train[:1000]
y_train = y_train[:1000]

# Dividir o conjunto de treinamento em três subconjuntos aleatórios
subset_size = len(X_train) // 3
subset1_X_train, subset2_X_train, subset3_X_train = X_train[:subset_size], X_train[subset_size:2*subset_size], X_train[2*subset_size:]
subset1_y_train, subset2_y_train, subset3_y_train = y_train[:subset_size], y_train[subset_size:2*subset_size], y_train[2*subset_size:]

# Treinar modelos Logistic Regression em cada subconjunto de dados
classifier1 = LogisticRegression()
classifier1.fit(subset1_X_train, subset1_y_train)

classifier2 = LogisticRegression()
classifier2.fit(subset2_X_train, subset2_y_train)

classifier3 = LogisticRegression()
classifier3.fit(subset3_X_train, subset3_y_train)

# Criar o ensemble usando soft voting
ensemble_classifier = VotingClassifier(estimators=[('lr1', classifier1), ('lr2', classifier2), ('lr3', classifier3)], voting='soft')

# Treinar o ensemble
ensemble_classifier.fit(X_train, y_train)

model_file = 'WELFake_ensemble_model.sav'
pickle.dump(ensemble_classifier, open(model_file, 'wb'))
logger.info(f"Model saved as {model_file}")