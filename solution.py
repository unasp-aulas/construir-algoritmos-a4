import pandas as pd #Importa a biblioteca pandas (trabalha cm dados em tabelas, CSV, etc)
import numpy as np #Importa a biblioteca numpy (operações matemáticas)

def regressao_linear(arquivo):
    df = pd.read_csv(arquivo) #Lê o arquivo
    area = df["area_util_m2"].to_list() #Pega a coluna com os dados da área e separa na variável area, separando em uma lista
    quartos = df["quantidade_quartos"].to_list() #Pega a coluna com os dados da qtde de quartos e separa na variável quartos, separando em uma lista
    preco = df["preco_venda"].to_list() #Pega a coluna com os dados do preço e separa na variável preco, separando em uma lista

    #Cria uma matriz onde os dados serão organizados pela seguinte ordem de colunas: número 1, area e quartos de cada apartamento
    X = np.column_stack(( 
        np.ones(len(area)), #Cria uma sequência de númro 1 repetidamente para a quantidade de apartamentos no arquivo
        area,
        quartos
    ))

    Y = np.array(preco) #Cria uma matriz Y com o preço de cada apartamento
    beta = np.linalg.inv(X.T @ X) @ X.T @ Y
    return beta.tolist() #Retorna os coeficientes
