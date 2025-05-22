# Projeto 1 - Construção e Deploy de Modelo de Machine Learning

# Construção do Modelo de Machine Learning

# Imports
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

# Dados de de produtos
dsa_dados = {
    'Peso_Embalagem_Gr': [212, 215, 890, 700, 230, 240, 730, 780, 218, 750, 202, 680],
    'Tipo_Embalagem': ['Caixa de Papelão', 'Caixa de Papelão', 'Plástico Bolha', 'Plástico Bolha', 'Caixa de Papelão', 'Caixa de Papelão', 'Plástico Bolha', 'Plástico Bolha', 'Caixa de Papelão', 'Plástico Bolha', 'Caixa de Papelão', 'Plástico Bolha'],
    'Tipo_Produto': ['Smartphone', 'Tablet', 'Tablet', 'Tablet', 'Smartphone', 'Smartphone', 'Tablet', 'Smartphone', 'Smartphone', 'Tablet', 'Smartphone', 'Tablet']  # Alterei o segundo e o oitavo rótulo
}

# Converte o dicionário em dataframe
df = pd.DataFrame(dsa_dados)

# Separa X (entrada) e Y (saída)
X = df[['Peso_Embalagem_Gr', 'Tipo_Embalagem']]
y = df['Tipo_Produto']

# Divide os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)  

# Cria e ajusta os transformadores nos dados de treinamento

# Fit da variável categórica Tipo_Embalagem
le_tipo_embalagem = LabelEncoder()
le_tipo_embalagem.fit(X_train['Tipo_Embalagem'])

# Fit da variável categórica Tipo_Produto
le_tipo_produto = LabelEncoder()
le_tipo_produto.fit(y_train)

# Aplica a transformação nos dados de treinamento e teste da variável categórica Tipo_Embalagem
X_train['Tipo_Embalagem'] = le_tipo_embalagem.transform(X_train['Tipo_Embalagem'])
X_test['Tipo_Embalagem'] = le_tipo_embalagem.transform(X_test['Tipo_Embalagem'])

# Aplica a transformação nos dados de treinamento e teste da variável categórica Tipo_Produto
y_train = le_tipo_produto.transform(y_train)
y_test = le_tipo_produto.transform(y_test)

# Cria o modelo
modelo_dsa = DecisionTreeClassifier()

# Treina o modelo
modelo_dsa.fit(X_train, y_train)

# Faz previsão com o modelo
y_pred = modelo_dsa.predict(X_test)

# Calcula a acurácia
acc_modelo_dsa = accuracy_score(y_test, y_pred)

# Print
print(f"\nAcurácia: ", round(acc_modelo_dsa,2))

print("\nRelatório de Classificação:\n")

# Obtém o classification report
report = classification_report(y_test, y_pred)

# Imprimir o report
print(report)

# Salva o modelo treinado
joblib.dump(modelo_dsa, 'modelos/modelo_logistica.pkl')

# Salva os transformadores
joblib.dump(le_tipo_embalagem, 'modelos/transformador_tipo_embalagem.pkl')
joblib.dump(le_tipo_produto, 'modelos/transformador_tipo_produto.pkl')





