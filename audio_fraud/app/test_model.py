from keras.models import load_model

# Carregar o modelo salvo
model = load_model('src/saved_model/model')

# Exibir um resumo do modelo
model.summary()