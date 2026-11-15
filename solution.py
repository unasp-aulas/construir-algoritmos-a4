import numpy as np
import pandas as pd


def calcular_regressao(arquivo):
 
  df = pd.read_csv(arquivo)

  
  area = df["area_util_m2"].to_list()
  quartos = df["quantidade_quartos"].to_list()
  preco = df["preco_venda"].to_list()

 
  X = np.column_stack((np.ones(len(area)), area, quartos))

  
  Y = np.array(preco)

 
  beta = np.linalg.inv(X.T @ X) @ X.T @ Y

  
  return beta.tolist()



coeficientes = calcular_regressao("imoveis_lar_perfeito.csv")
print("Lista de coeficientes:", coeficientes)
print(
    f"Preco = {coeficientes[0]:.2f} + {coeficientes[1]:.2f}*Area +"
    f" {coeficientes[2]:.2f}*Quartos"
)
