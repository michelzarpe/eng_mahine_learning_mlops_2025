# Projeto 3 - Implementação de Feature Store em Pipeline de Deploy de Machine Learning
# Módulo de Inferência

# Imports
import joblib
import pandas as pd

# Função para carregar o modelo treinado
def dsa_carrega_modelo(model_path):
    return joblib.load(model_path)

# Função para fazer previsão
def dsa_faz_previsao(model, features):
    predictions = model.predict(features)
    return predictions

# Função para executar as duas funções anteriores
def dsa_inferencia(model_path, features):
    model = dsa_carrega_modelo(model_path)
    predictions = dsa_faz_previsao(model, features)
    return predictions
