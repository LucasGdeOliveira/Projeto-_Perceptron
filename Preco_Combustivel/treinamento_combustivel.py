import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Configuração de exibição do Pandas
pd.set_option('display.max_columns', None)

# Carregamento dos datasets
combustiveis_brasil = pd.read_csv('combustiveis-brasil.csv')
combustiveis_estados = pd.read_csv('combustiveis-estados.csv')
combustiveis_regioes = pd.read_csv('combustiveis-regioes.csv')

# Função para visualizar os dados
def visualizar_dados(nivel, ano=None, regiao=None, estado=None):
    """
    Visualiza os dados a nível Brasil, Estado ou Região.
    """
    if nivel == 'Brasil':
        dados = combustiveis_brasil
    elif nivel == 'Estados':
        dados = combustiveis_estados
        if regiao:
            dados = dados[dados['regiao'].str.upper() == regiao.upper()]
        if estado:
            dados = dados[dados['estado'].str.upper() == estado.upper()]
    elif nivel == 'Regioes':
        dados = combustiveis_regioes
    else:
        print("Nível inválido. Escolha entre 'Brasil', 'Estados' ou 'Regioes'.")
        return

    if ano:
        dados = dados[dados['ano'] == int(ano)]
    
    print(f"\nExibindo dados a nível {nivel}:")
    print(dados.tail(12))

# Entrada do usuário
nivel_de_visualizacao = input("Escolha o nível de dados (Brasil, Estados, Regioes): ").strip().capitalize()

if nivel_de_visualizacao not in ['Brasil', 'Estados', 'Regioes']:
    print("Entrada inválida. Escolha entre Brasil, Estados ou Regioes.")
    exit()

regiao_escolhida = None
estado_escolhido = None

if nivel_de_visualizacao == 'Regioes':
    regiao_escolhida = input("Escolha a região (NORTE, SUL, NORDESTE, CENTRO-OESTE, SUDESTE): ")
elif nivel_de_visualizacao == 'Estados':
    estado_escolhido = input("Escolha o estado (ACRE, ALAGOAS, AMAPÁ, AMAZONAS, BAHIA, CEARÁ, DISTRITO FEDERAL, ESPÍRITO SANTO, GOIÁS, MARANHÃO, MATO GROSSO, MATO GROSSO DO SUL, MINAS GERAIS, PARÁ, PARAÍBA, PARANÁ, PERNAMBUCO, PIAUÍ, RIO DE JANEIRO, RIO GRANDE DO NORTE, RIO GRANDE DO SUL, RONDÔNIA, RORAIMA, SANTA CATARINA, SÃO PAULO, SERGIPE, TOCANTINS): ")

ano_escolhido = input("Escolha um ano entre 2001 a 2023: ")
visualizar_dados(nivel_de_visualizacao, ano_escolhido, regiao_escolhida, estado_escolhido)

# Seleção e validação de colunas
colunas_necessarias = ['ano', 'mes', 'gasolina_comum_preco_revenda_avg']
colunas_existentes = [col for col in colunas_necessarias if col in combustiveis_brasil.columns]

if len(colunas_existentes) < len(colunas_necessarias):
    print("Aviso: Algumas colunas necessárias não foram encontradas.")
    print("Colunas encontradas:", colunas_existentes)
else:
    dados = combustiveis_brasil[colunas_existentes]

# Separação de atributos e alvo
X = dados[['ano', 'mes']]
y = dados['gasolina_comum_preco_revenda_avg']

# Padronização dos dados
padronizador = StandardScaler()
X = padronizador.fit_transform(X)

# Divisão em treino e teste
X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinamento e avaliação de modelos
def treinar_modelo(ativacao, taxa_aprendizado):
    """
    Treina o modelo MLP com uma função de ativação e taxa de aprendizado específicas.
    """
    modelo = MLPRegressor(
        hidden_layer_sizes=(64, 32, 16),
        activation=ativacao,
        learning_rate_init=taxa_aprendizado,
        max_iter=1000,
        random_state=42
    )
    modelo.fit(X_treino, y_treino)
    y_predito = modelo.predict(X_teste)
    
    mae = mean_absolute_error(y_teste, y_predito)
    mse = mean_squared_error(y_teste, y_predito)
    
    print(f"\nAtivação: {ativacao}, Taxa de Aprendizado: {taxa_aprendizado}")
    print(f"MAE: {mae}, MSE: {mse}")
    print(f"\n")
    return modelo

# Treinando e comparando com diferentes funções de ativação
funcoes_ativacao = ['relu', 'logistic']  # ReLU e Sigmoid
taxa_aprendizado = 0.02

modelos = []
for ativacao in funcoes_ativacao:
    modelo = treinar_modelo(ativacao, taxa_aprendizado)
    modelos.append(modelo)

# Gráficos
def plotar_resultados(modelo, nome_ativacao):
    """
    Plota os gráficos de treinamento e comparação entre valores reais e previstos.
    """
    plt.figure(figsize=(12, 5))
    
    # Curva de erro
    plt.subplot(1, 2, 1)
    plt.plot(modelo.loss_curve_, label=f'Erro - {nome_ativacao}')
    plt.xlabel('Iterações')
    plt.ylabel('Erro')
    plt.title('Convergência do Erro')
    plt.legend()

    # Valores reais vs previstos
    y_predito = modelo.predict(X_teste)
    plt.subplot(1, 2, 2)
    plt.plot(y_teste.values, label='Real', color='blue')
    plt.plot(y_predito, label='Previsto', color='red', linestyle='--')
    plt.title(f'Valores Reais vs Previstos - {nome_ativacao}')
    plt.xlabel('Amostras')
    plt.ylabel('Preço (R$)')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Plotar resultados para cada modelo
for i, modelo in enumerate(modelos):
    plotar_resultados(modelo, funcoes_ativacao[i])
