import numpy as np
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from collections import defaultdict
import math
from termcolor import colored

# Dataset Spam email classification - Ashfak Yeafi
df = pd.read_csv("./dataset/email.csv")

# Pré-processamento
def processar_mensagem(text):
    text = text.lower()                                               # Converte o texto para minúsculas
    text = re.sub(r'\W', ' ', text)                                   # Remove caracteres especiais
    text = re.sub(r'\s+', ' ', text).strip()                          # Remove espaços em branco extras
    return text

# Aplicar o pré-processamento nas mensagens
df['mensagem_processada'] = df['Message'].apply(processar_mensagem)

# Adicionando uma coluna binária
df['Spam'] = df['Category'].apply(lambda x: 1 if x == 'spam' else 0)  # Converte (ham e spam) em valores numéricos (0 e 1)

# Divisão dos dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(df['mensagem_processada'], df['Spam'], test_size=0.3, random_state=42)

class NaiveBayesClassifier:

    def __init__(self):
        self.word_probs = defaultdict(lambda: [0, 0])
        self.class_probs = {}
        self.vocab = set()
        self.total_words = [0, 0]
        self.class_counts = [0, 0]
    
    def fit(self, X, y):

        # Contagem de palavras
        for message, label in zip(X, y):
            class_index = label
            self.class_counts[class_index] += 1
            words = message.split()
            self.total_words[class_index] += len(words)
            
            for word in words:
                self.vocab.add(word)
                self.word_probs[word][class_index] += 1
        
        # Calculo de probabilidade
        total_count = sum(self.class_counts)
        self.class_probs[0] = self.class_counts[0] / total_count
        self.class_probs[1] = self.class_counts[1] / total_count

    def predict(self, X):
      
        predictions = []
      
        for message in X:
            words = message.split()
            prob_ham = math.log(self.class_probs[0])
            prob_spam = math.log(self.class_probs[1])
          
            for word in words:
                if word in self.vocab:
                    prob_palavra_ham = (self.word_probs[word][0] + 1) / (self.total_words[0] + len(self.vocab))
                    prob_palavra_spam = (self.word_probs[word][1] + 1) / (self.total_words[1] + len(self.vocab))
                    prob_ham += math.log(prob_palavra_ham)
                    prob_spam += math.log(prob_palavra_spam)
                  
            if prob_ham > prob_spam:
                predictions.append(0)
              
            else:
                predictions.append(1)
              
        return predictions

# Treinamento do modelo
nb = NaiveBayesClassifier()
nb.fit(X_train, y_train)

# Testes do modelo
y_pred = nb.predict(X_test)

def evaluate(y_true, y_pred):
    verdadeiro_positivo = sum((y_true == 1) & (y_pred == 1))
    verdadeiro_negativo = sum((y_true == 0) & (y_pred == 0))
    falso_positivo = sum((y_true == 0) & (y_pred == 1))
    falso_negativo = sum((y_true == 1) & (y_pred == 0))

    precisao = verdadeiro_positivo / (verdadeiro_positivo + falso_positivo) if (verdadeiro_positivo + falso_positivo) > 0 else 0
    recall = verdadeiro_positivo / (verdadeiro_positivo + falso_negativo) if (verdadeiro_positivo + falso_negativo) > 0 else 0
    f1_score = 2 * (precisao * recall) / (precisao + recall) if (precisao + recall) > 0 else 0
    acuracia = (verdadeiro_positivo + verdadeiro_negativo) / len(y_true)
    
    return {
        'precisao': precisao,
        'recall': recall,
        'f1_score': f1_score,
        'acuracia': acuracia
    }

# Avaliação do modelo
resultados = evaluate(np.array(y_test), np.array(y_pred))
print(f"\nPrecisão: {resultados['precisao']}")
print(f"Recall: {resultados['recall']}")
print(f"F1 Score: {resultados['f1_score']}")
print(f"Acurácia: {resultados['acuracia']}\n")

# ===============================================================================================================================================
# ==========================================================( Teste com Novos E-mails )==========================================================
# ===============================================================================================================================================

mensagens_teste = [
    "Sounds great! Are you home now?",
    "Will u meet ur dream partner soon? Is ur career off 2 a flyng start? 2 find out free, txt HORO followed by ur star sign, e. g. HORO ARIES",
    "Congratulations! You've won a free ticket to Bahamas. Call now!",
    "Hi, I hope you're doing well. Can we schedule a meeting?",
    "Limited time offer: get a 50% discount on your next purchase!"
]

# Pré-processar os novos emails de teste
mensagens_text_processadas = [processar_mensagem(msg) for msg in mensagens_teste]

# Fazer previsões
predictions = nb.predict(mensagens_text_processadas)

# Resultados das previsões
for msg, pred in zip(mensagens_teste, predictions):
    color = 'red' if pred == 1 else 'green'
    label = 'spam' if pred == 1 else 'ham'
    print(f"\nMensagem: '{msg}' - Classificação: {colored(f'({label})', color)}")
