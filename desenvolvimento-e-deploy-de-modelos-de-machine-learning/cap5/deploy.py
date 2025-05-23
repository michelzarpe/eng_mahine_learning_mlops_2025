# Projeto 2 - Prevendo o Churn de Clientes com RandomForest - Da Concepção do Problema ao Deploy
# Deploy do Modelo

# Instale o streamlit: pip install streamlit

# Imports
import joblib
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Carregar o modelo e o scaler
modelo_dsa = joblib.load('dsa_modelo_final.pkl')
scaler = joblib.load('dsa_padronizador.pkl')

# Função para pré-processar os dados de entrada
# As colunas devem ser exatamente as mesmas usadas durante o treinamento
def preprocess_input(idade, 
                     uso_mensal, 
                     satisfacao_cliente, 
                     valor_mensal, 
                     plano_basico, 
                     plano_premium, 
                     plano_standard, 
                     contrato_curto, 
                     contrato_medio,
                     contrato_longo):
    
    # Dataframe
    data = pd.DataFrame({
        'Idade': [idade],
        'UsoMensal': [uso_mensal],
        'SatisfacaoCliente': [satisfacao_cliente],
        'ValorMensal': [valor_mensal],
        'Plano_Basico': [plano_basico],
        'Plano_Premium': [plano_premium],
        'Plano_Standard': [plano_standard],
        'TempoContrato_Curto': [contrato_curto],
        'TempoContrato_Longo': [contrato_longo],
        'TempoContrato_Medio': [contrato_medio]
    })

    # Lista de colunas
    numeric_cols = ['Idade', 
                    'UsoMensal', 
                    'SatisfacaoCliente', 
                    'ValorMensal', 
                    'Plano_Basico', 
                    'Plano_Premium', 
                    'Plano_Standard', 
                    'TempoContrato_Curto', 
                    'TempoContrato_Longo', 
                    'TempoContrato_Medio']

    # Aplicando padronização
    data[numeric_cols] = scaler.transform(data[numeric_cols])

    return data

# Função para fazer previsões
def predict(data):
    prediction = modelo_dsa.predict(data)
    return prediction

# Interface do Streamlit
st.title("DSA - Preditor de Churn com RandomForest")

# Criação de campos para entrada de dados
idade = st.number_input('Idade', min_value = 18, max_value = 100, value = 30)
uso_mensal = st.number_input('Uso Mensal', min_value = 0, max_value = 200, value = 50)
satisfacao_cliente = st.number_input('Satisfação do Cliente', min_value = 1, max_value = 5, value = 3)
valor_mensal = st.number_input('Valor Mensal', min_value = 0.0, max_value = 500.0, value = 100.0)
plano = st.selectbox('Plano', ['Basico', 'Premium', 'Standard'])
tempo_contrato = st.selectbox('Tempo de Contrato', ['Curto', 'Medio', 'Longo'])

# Botão para realizar previsões
if st.button('Prever Churn'):

    # Ajusta as variáveis pré-processadas com One-Hot Encoding

    # Plano
    plano_basico = 1 if plano == 'Basico' else 0
    plano_premium = 1 if plano == 'Premium' else 0
    plano_standard = 1 if plano == 'Standard' else 0

    # Tempo contrato
    tempo_contrato_curto = 1 if tempo_contrato == 'Curto' else 0
    tempo_contrato_medio = 1 if tempo_contrato == 'Medio' else 0
    tempo_contrato_longo = 1 if tempo_contrato == 'Longo' else 0

    # Executa a função de pré-processamento de dados
    input_data = preprocess_input(idade, 
                                  uso_mensal, 
                                  satisfacao_cliente, 
                                  valor_mensal, 
                                  plano_basico, 
                                  plano_premium, 
                                  plano_standard, 
                                  tempo_contrato_curto, 
                                  tempo_contrato_medio, 
                                  tempo_contrato_longo)

    # Faz a previsão com o modelo
    prediction = predict(input_data)

    st.write('Churn Previsto:' , 'Sim' if prediction[0] == 1 else 'Não')

    st.write('Obrigado - Data Science Academy')




