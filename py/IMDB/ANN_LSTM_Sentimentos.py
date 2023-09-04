# Importando bibliotecas
import nltk
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Importando biblioteca dataset imdb
from tensorflow.keras.datasets import imdb

# Importando biblioteca nltk stopwords
from nltk.corpus import stopwords

# Importando biblioteca rede neural
import keras
import tensorflow as tf

# Importando biblioteca keras tensorflow rede neural
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Embedding, Dense, Dropout
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import confusion_matrix, classification_report

## Dataset
# Carregar o conjunto de dados IMDB
max_words = 10000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=max_words)

# Vocabulário reverso para mapear índices de palavras para palavras reais
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# Função para converter uma sequência numérica em texto
def sequence_to_text(sequence):
    return ' '.join([reverse_word_index.get(idx - 3, '?') for idx in sequence])

# Converter as sequências numéricas em texto
sample_text = sequence_to_text(X_train[0])  # Converter a primeira sequência em texto

# Criar um DataFrame para o conjunto de treinamento
train_df = pd.DataFrame({'Review': X_train, 'Sentiment': y_train})

# Criar um DataFrame para o conjunto de teste
test_df = pd.DataFrame({'Review': X_test, 'Sentiment': y_test})

# Acessar os dados de treinamento
X_train_data = train_df['Review']
y_train_data = train_df['Sentiment']

# Acessar os dados de teste
X_test_data = test_df['Review']
y_test_data = test_df['Sentiment']

# Pré-processamento dos dados
max_review_length = 500

X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)

# **Rede Neural**
## Rede neural LSTM 1 - Simples

# Definir a arquitetura da rede neural LSTM
model = Sequential()
model.add(Embedding(max_words, 128, input_length=max_review_length))
model.add(LSTM(128))
model.add(Dense(1, activation='sigmoid'))

# Compilar o modelo
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

batch_size = 64
epochs = 100

# Treinar o modelo
model_lstm = model.fit(X_train,
                       y_train,
                       validation_data=(X_test, y_test),
                       batch_size=batch_size,
                       epochs=epochs)

# Extrair as métricas de treinamento
train_loss = model_lstm.history['loss']
val_loss = model_lstm.history['val_loss']

train_accuracy = model_lstm.history['accuracy']
val_accuracy = model_lstm.history['val_accuracy']

# Avaliar o modelo
scores = model.evaluate(X_test, y_test, verbose=0)
print("Acurácia Rede neural LSTM: %.2f%%" % (scores[1] * 100))

# Fazer previsões
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int)

from sklearn.metrics import classification_report
# Imprimir relatório de classificação
print("Relatório de Classificação - Rede neural LSTM:")
print(classification_report(y_test, y_pred))

## Resultados

# Criar a matriz de confusão
from sklearn.metrics import confusion_matrix, classification_report

confusion = confusion_matrix(y_test, y_pred)
print('Confusion matrix - Rede Neural LSTM \n\n', confusion)
print('\nVerdadeiro Positivo(TP) = ', confusion[0,0])
print('\nVerdadeiro Negativo(TN) = ', confusion[1,1])
print('\nFalso Positivo(FP) = ', confusion[0,1])
print('\nFalso Negativo(FN) = ', confusion[1,0])

# Plot matriz confussão 
plt.figure(figsize=(10, 5))
ax = plt.subplot()
sns.heatmap(confusion, annot=True, ax = ax, fmt = ".1f", cmap="Blues"); 
ax.set_title('Confusion Matrix - Rede Neural LSTM'); 
ax.xaxis.set_ticklabels(["Negativo", "Positivo"]); ax.yaxis.set_ticklabels(["Negativo", "Positivo"]);
plt.xlabel('Previsto')
plt.ylabel('Verdadeiro')
plt.show()

# Plotar as métricas
epochs_range = range(1, epochs + 1)

plt.figure(figsize=(15.5, 10))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, train_loss, label='Loss de Treinamento')
plt.plot(epochs_range, val_loss, label='Loss de Validação')
plt.legend()
plt.title('Loss de Treinamento e Validação')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, train_accuracy, label='Acurácia de Treinamento')
plt.plot(epochs_range, val_accuracy, label='Acurácia de Validação')
plt.legend()
plt.title('Acurácia de Treinamento e Validação')

plt.show()

# Salvando modelo 
# Salvando rede neural 
from tensorflow.keras.models import save_model

# Salvar o modelo treinado em um arquivo
# 'modelo.h5' é o nome do arquivo onde o modelo será salvo
model.save('modelo_sentiment_IMDB_1.h5')