Referencia do DataSet: https://www.kaggle.com/datasets/fidelissauro/combustiveis-brasil

RELATÓRIO DE ANÁLISE DO CÓDIGO

1. INTRODUÇÃO
O código fornecido implementa um modelo de rede neural para prever o preço médio de combustíveis no Brasil. Ele utiliza o modelo Perceptron Multicamadas (MLP) e emprega bibliotecas como pandas, NumPy, Matplotlib, e Scikit-learn. O objetivo principal é prever o preço médio da gasolina comum com base em atributos como ano e mês.

2. ETAPAS PRINCIPAIS DO CÓDIGO

2.1 Importação de Bibliotecas
As bibliotecas utilizadas são essenciais para manipulação de dados (pandas, NumPy), visualização (Matplotlib, Seaborn) e aprendizado de máquina (Scikit-learn).

2.2 Carregamento de Datasets
Três datasets relacionados a combustíveis no Brasil são carregados:
- combustiveis-brasil.csv: Dados a nível nacional.
- combustiveis-estados.csv: Dados por estado.
- combustiveis-regioes.csv: Dados por região.
Esses datasets são lidos utilizando pd.read_csv(), e nenhuma limpeza explícita é feita no código.

2.3 Visualização dos Dados
A função "visualizar_dados" permite exibir os dados filtrados por nível (Brasil, Estados, Regiões), ano, região ou estado. Essa função utiliza filtros em pandas para selecionar os dados relevantes.

2.4 Separação de Atributos e Alvo
Os atributos selecionados são:
- ano (Ano de referência).
- mes (Mês de referência).
O alvo é:
- gasolina_comum_preco_revenda_avg (Preço médio da gasolina comum).

2.5 Padronização dos Dados
Os atributos de entrada são padronizados usando "StandardScaler" para garantir que os dados estejam na mesma escala, o que melhora o desempenho do modelo de rede neural.

2.6 Divisão do Conjunto de Dados
Os dados são divididos em conjuntos de treino e teste, com 80% dos dados para treino e 20% para teste, utilizando "train_test_split".

2.7 Implementação do Modelo
A função "treinar_modelo" cria um modelo MLP com as seguintes características:
- Estrutura: Três camadas ocultas com 64, 32 e 16 neurônios.
- Função de ativação: Testa "relu" e "logistic".
- Taxa de aprendizado inicial: 0.02.
- Número máximo de iterações: 1000.
Após o treinamento, o modelo é avaliado utilizando as métricas MAE (Erro Médio Absoluto) e MSE (Erro Quadrático Médio).

2.8 Visualização dos Resultados
A função "plotar_resultados" exibe:
- Curva de erro ao longo das iterações de treinamento.
- Comparação entre valores reais e previstos no conjunto de teste.

3. CONCLUSÃO
O código implementa com sucesso um modelo MLP para prever preços de combustíveis, atendendo aos critérios básicos de aprendizado de máquina.

4.JUSTIFICATIVA DO TEMA
O tema "Preço dos Combustíveis no Brasil" foi escolhido devido à sua relevância socioeconômica e impacto direto na vida cotidiana da população e na economia do país. Os combustíveis representam um dos principais insumos para o transporte de pessoas e mercadorias, afetando desde o custo de vida dos cidadãos até os índices de inflação e competitividade dos produtos nacionais.
A escolha do tema reflete não apenas sua relevância no cenário econômico e social do Brasil, mas também sua adequação como um problema técnico para modelagem preditiva, contribuindo para análises mais precisas e decisões mais informadas.

5.Relu X Logistic (Sigmoid)

ReLU é uma função de ativação simples e eficiente para redes neurais. Ela retorna 0 para valores negativos e o próprio valor para entradas positivas.
Rápida e eficiente no treinamento.

Logistic (Sigmoid) é uma função de ativação que transforma entradas em um intervalo entre 0 e 1.
​Ideal para classificação binária, interpretando saídas como probabilidades.



