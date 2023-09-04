import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Carregar o modelo treinado
modelo_carregado = load_model('modelo.h5')

texto_exemplo = input(str("Digite a frase:"))

# Pré-processar o texto de exemplo
max_words = 10000  # Mesmo valor usado no treinamento
max_review_length = 500  # Mesmo valor usado no treinamento

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texto_exemplo)
sequences = tokenizer.texts_to_sequences(texto_exemplo)
padded_text = pad_sequences(sequences, maxlen=max_review_length)

# Fazer a previsão
previsao = modelo_carregado.predict(padded_text)

# O resultado da previsão será uma probabilidade (0 a 1)
# Você pode definir um limiar para classificar como positivo ou negativo
limiar = 0.5  # Por exemplo, 0.5 para classificar como positivo se a probabilidade for maior ou igual a 0.5
if previsao[0][0] >= limiar:
    resultado = "Positivo"
else:
    resultado = "Negativo"

print("Texto de Exemplo:", texto_exemplo[0])
print("Probabilidade de Ser Positivo:", previsao[0][0])
print("Classificação:", resultado)