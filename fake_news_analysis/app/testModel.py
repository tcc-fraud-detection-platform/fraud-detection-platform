import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt
import torch
import transformers as ppb 
from sklearn.linear_model import LogisticRegression
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Carregar o modelo treinado
model_file = 'WELFake_model.sav'
loaded_model = pickle.load(open(model_file, 'rb'))

# Carregar os dados de teste e validação do arquivo CSV
test_filename = './welfake_dataset/WELFAKE_Test.csv'
test_news = pd.read_csv(test_filename)

valid_filename = './welfake_dataset/WELFAKE_Valid.csv'
valid_news = pd.read_csv(valid_filename)

max_seq_length = 128

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

model_class, tokenizer_class, pretrained_weights = (ppb.BertModel, ppb.BertTokenizer, 'bert-base-uncased')
tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
model = model_class.from_pretrained(pretrained_weights)
model.to(device)  # Transferir o modelo para a GPU se disponível

# Pré-processar os dados de teste e validação da mesma forma que no arquivo original
def preprocess_with_bert(text):
    tokenized = tokenizer.encode(text, add_special_tokens=True)
    if len(tokenized) > max_seq_length:
        tokenized = tokenized[:max_seq_length]
    padded = tokenized + [0] * (max_seq_length - len(tokenized))
    input_ids = torch.tensor([padded]).to(device)
    return input_ids

test_news = test_news.dropna(subset=['Statement']).reset_index(drop=True)
valid_news = valid_news.dropna(subset=['Statement']).reset_index(drop=True)

test_news['Statement'] = test_news['Statement'].apply(lambda x: x[:max_seq_length] if len(x) > max_seq_length else x)
valid_news['Statement'] = valid_news['Statement'].apply(lambda x: x[:max_seq_length] if len(x) > max_seq_length else x)

test_inputs = test_news['Statement'].apply(preprocess_with_bert)
valid_inputs = valid_news['Statement'].apply(preprocess_with_bert)

# Obter embeddings para os dados de teste
test_embeddings = []
for input_ids in test_inputs:
    with torch.no_grad():
        outputs = model(input_ids)
    last_hidden_states = outputs.last_hidden_state
    test_embeddings.append(last_hidden_states.cpu().numpy())

# Obter embeddings para os dados de validação
valid_embeddings = []
for input_ids in valid_inputs:
    with torch.no_grad():
        outputs = model(input_ids)
    last_hidden_states = outputs.last_hidden_state
    valid_embeddings.append(last_hidden_states.cpu().numpy())

# Converter embeddings para numpy arrays e achatar para 2D
X_test = np.vstack(test_embeddings).reshape(len(test_embeddings), -1)
X_valid = np.vstack(valid_embeddings).reshape(len(valid_embeddings), -1)

# Preparar os rótulos
y_test = test_news['Label']
y_valid = valid_news['Label']

# Fazer previsões nos dados de teste
y_pred_test = loaded_model.predict(X_test)

# Avaliar o desempenho do modelo nos dados de teste
accuracy_test = accuracy_score(y_test, y_pred_test)
precision_test = precision_score(y_test, y_pred_test)
recall_test = recall_score(y_test, y_pred_test)
f1_test = f1_score(y_test, y_pred_test)

print("Performance on test data:")
print(f"Accuracy: {accuracy_test}")
print(f"Precision: {precision_test}")
print(f"Recall: {recall_test}")
print(f"F1-score: {f1_test}")
print(classification_report(y_test, y_pred_test))

# Fazer previsões nos dados de validação
y_pred_valid = loaded_model.predict(X_valid)

# Avaliar o desempenho do modelo nos dados de validação
accuracy_valid = accuracy_score(y_valid, y_pred_valid)
precision_valid = precision_score(y_valid, y_pred_valid)
recall_valid = recall_score(y_valid, y_pred_valid)
f1_valid = f1_score(y_valid, y_pred_valid)

print("Performance on validation data:")
print(f"Accuracy: {accuracy_valid}")
print(f"Precision: {precision_valid}")
print(f"Recall: {recall_valid}")
print(f"F1-score: {f1_valid}")
print(classification_report(y_valid, y_pred_valid))

# Plotar a Matriz de Confusão
cm_test = confusion_matrix(y_test, y_pred_test)
plt.figure(figsize=(8, 6))
plt.imshow(cm_test, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix - Test Data')
plt.colorbar()
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.xticks([0, 1], ['Real', 'Fake'])
plt.yticks([0, 1], ['Real', 'Fake'])
plt.savefig('model_metrics/confusion_matrix_test.png')

# Plotar a Curva ROC
fpr, tpr, _ = roc_curve(y_test, y_pred_test)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Test Data')
plt.legend(['ROC Curve', 'Random Guess'], loc='lower right')
plt.savefig('model_metrics/roc_curve_test.png')

# Plotar a Curva Precision-Recall
precision, recall, _ = precision_recall_curve(y_test, y_pred_test)
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='blue', lw=2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve - Test Data')
plt.savefig('model_metrics/pr_curve_test.png')

# Salvar as métricas em tabelas como imagens
metrics_table_test = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-score'],
    'Value': [accuracy_test, precision_test, recall_test, f1_test]
})

plt.figure(figsize=(8, 6))
plt.table(cellText=metrics_table_test.values, colLabels=metrics_table_test.columns, loc='center')
plt.title('Performance Metrics - Test Data')
plt.axis('off')
plt.savefig('model_metrics/metrics_table_test.png')