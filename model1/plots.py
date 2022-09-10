# -*- coding: utf-8 -*-
"""
Created on Sat May 11 10:50:41 2019

@author: dtcas
"""
from model import get_observed_proportion_cash_payments
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import dill


with open("results/run_data_model1_w5.pkl",'rb') as f:
    run_data = dill.load(f)

# with open("results/run_data_teste.pkl",'rb') as f:
#     run_data_teste = dill.load(f)

# frames = [run_data_1, run_data_teste]
  
# run_data = pd.concat(frames)
# print(run_data)
    
######################## 
## Figure 1 - gráfico distribuição por valor

diary = pd.read_csv("../../pagtos_varejo_escolha/dados/export/diario_sub.csv", sep = ';', thousands='.', decimal=',')
max = diary['valor'].max()
interval_range = pd.interval_range(start=0, freq=2, end=max+2)

interval_total = pd.cut(diary['valor'], bins=interval_range) #, right = False)
dist_total = interval_total.value_counts(normalize=True, sort=False).to_frame()

plt.tight_layout()
dist_total.plot.line(title = "Percentage of payments per R\$ 2 value interval")
plt.xticks(rotation='vertical')
#plt.legend(["value"])
ax = plt.gca()
ax.legend_ = None

plt.tight_layout()
plt.savefig('dist_value.pdf')#, figsize=(10,10)) # ver o melhor fig_size
plt.show()

dist_200 = dist_total.iloc[0:251,0]
dist_200.plot.line(title = "Percentage of payments up to R\$200 per R\$2 value interval")
#plt.rc('xtick', labelsize=8)
plt.xticks(rotation='vertical')#, labels=[1,2,3,4,5,6])#, labels=["(0, 2]", "(100, 102]", "(200, 202]", "(300, 302]", "(400, 402]", "(500, 502]"])
#plt.legend(["value"])
ax = plt.gca()
ax.legend_ = None

plt.tight_layout()
plt.savefig('dist_value_500.pdf', figsize=(6,6)) # ver o melhor fig_size
plt.show()

# todo:
# tamanho da imagem no pdf
# alinhar os eixos das duas imagens

#########################

### Best mth
print(run_data.info())
print(run_data.head())
print(run_data[['threshold_cash_balance_mth', 'difference_proportion_cash_payments']])
min_index = np.argmin(run_data['difference_proportion_cash_payments']) 
print(min_index)
print("Melhor mth: ")
print(run_data.iloc[min_index])
print("Menor diferença: ")
print(run_data.iloc[min_index, 2])

#### Best mean

print('médias')
mean_differences_mth = run_data.groupby('threshold_cash_balance_mth')['difference_proportion_cash_payments'].mean()
print(mean_differences_mth)
min_index_means = np.argmin(mean_differences_mth) 
print(mean_differences_mth.iloc[min_index_means])
# imprimir o valor, que é 500

df_graph = run_data.groupby('threshold_cash_balance_mth').agg({'difference_proportion_cash_payments': ['mean', 'min', 'max']})
print(df_graph)

##############
## Figure 2

sim_mth = run_data[['threshold_cash_balance_mth', 'difference_proportion_cash_payments']]

### a
fig2a = sns.lineplot(data = sim_mth,
             x = 'threshold_cash_balance_mth',
             y = 'difference_proportion_cash_payments',
             ci=95,
             markers=True)

fig2a.set(title="Model 1 error by $m^{th}$")
fig2a.set_xlabel("$m^{th}$")
fig2a.set_ylabel("Difference (%) between observed and simulated")


plt.savefig('model1_sim_mth.pdf', figsize=(6,6)) # ver o melhor fig_size
plt.show()

### b
sim_mth_zoom = sim_mth[sim_mth['threshold_cash_balance_mth'] > 99]
print(sim_mth_zoom.head())

fig2b = sns.lineplot(data = sim_mth_zoom,
             x = 'threshold_cash_balance_mth',
             y = 'difference_proportion_cash_payments',
             ci=95,
             markers=True)

fig2b.set(title="Model 1 error by $m^{th}$ above R\$100")
fig2b.set_xlabel("$m^{th}$")
fig2b.set_ylabel("Difference (%) between observed and simulated")

plt.savefig('model1_sim_mth_zoom.pdf', figsize=(6,6)) # ver o melhor fig_size
plt.show()

##############
## Figure 3 - Comp observed x simulated
obs_cash = get_observed_proportion_cash_payments()
obs_200 = obs_cash.iloc[0:51, 0]
#obs_200.plot.line(title = "Percentual de pagamentos em dinheiro")
#plt.show()

sim_cash = run_data.iloc[min_index, 7] # 7 agora é o proportion_cash_payments 
sim_200 = sim_cash.iloc[0:51,0]

#print(sim_200)
#sim_200.plot.line()
#plt.show()

df_difference = pd.DataFrame({
    'Observed': obs_200,
    'Simulated': sim_200}, index = list(sim_200.index))

#plt.rc('xtick', labelsize=8)
plt.tight_layout()
#plt.margins(x=2)
df_difference.plot.line(title = "Percentage of cash payments per value in model 1")#,
                        #figsize=(7,5))
plt.savefig('model1_dist_comp.pdf')#, figsize=(7,5)) # ver o melhor fig_size
plt.tight_layout()
plt.show()

#### Plot das distribuições dos saques
saques = pd.read_csv("../data/saques_abm.csv", sep = ';')
#dist_w = saques[[label_dist]].dropna().to_numpy()
saques["valor1"].hist(bins=[10, 20, 50, 100, 200, 500, 1000, 5000], #, 5000], 
#                      density=1,
                      rwidth=0.8,
                      grid=False,
                      alpha=0.5)
plt.show()

#sns.distplot(a=saques['valor1'])
withdrawal_dist_1 = sns.histplot(data=saques, x='valor1', stat='probability')
withdrawal_dist_1.set(xlabel ="Withdrawal value", ylabel = "Frequency (%)", title ='Withdrawal distribution 1')

#plt.show()

plt.savefig('withdrawal_dist_1.pdf') # ver o melhor fig_size
plt.legend()
plt.show()

saques["valor2"].hist(bins=[10, 20, 50, 100, 200, 500, 1000], #, 5000], 
                      density=1,
                      alpha=0.5,
                      label="valor2")
plt.show()

withdrawal_dist_2 = sns.histplot(data=saques, x='valor2', stat='probability')
withdrawal_dist_2.set(xlabel ="Withdrawal value", ylabel = "Frequency (%)", title ='Withdrawal distribution 2')

plt.savefig('withdrawal_dist_2.pdf') # ver o melhor fig_size
plt.legend()
plt.show()

saques["valor3"].hist(bins=[10, 20, 50, 100, 200, 500, 1000], #, 5000], 
                      density=1,
                      alpha=0.5,
                      label="valor3")
plt.show()

withdrawal_dist_3 = sns.histplot(data=saques, x='valor3', stat='probability')
withdrawal_dist_3.set(xlabel ="Withdrawal value", ylabel = "Frequency (%)", title ='Withdrawal distribution 3')

plt.savefig('withdrawal_dist_3.pdf') # ver o melhor fig_size
plt.legend()
plt.show()

saques["valor4"].hist(bins=[10, 20, 50, 100, 200, 500, 1000], #, 5000], 
                      density=1,
                      alpha=0.5,
                      label="valor4")
plt.show()

withdrawal_dist_4 = sns.histplot(data=saques, x='valor4', stat='probability')
withdrawal_dist_4.set(xlabel ="Withdrawal value", ylabel = "Frequency (%)", title ='Withdrawal distribution 4')

plt.savefig('withdrawal_dist_4.pdf') # ver o melhor fig_size
plt.legend()
plt.show()

saques["valor5"].hist(bins=[10, 20, 50, 100, 200, 500, 1000], #, 5000], 
                      density=1,
                      alpha=0.5,
                      label="valor5")
plt.show()

withdrawal_dist_5 = sns.histplot(data=saques, x='valor5', stat='probability')
withdrawal_dist_5.set(xlabel ="Withdrawal value", ylabel = "Frequency (%)", title ='Withdrawal distribution 5')

plt.savefig('withdrawal_dist_5.pdf') # ver o melhor fig_size
plt.legend()
plt.show()

# saques["valor1"].hist(bins=[10, 20, 50, 100], density=1)
# plt.show()

saques["valor1"].hist(bins=[10, 20, 50, 100], 
                      density=1,
                      alpha=0.2)

saques["valor2"].hist(bins=[10, 20, 50, 100], 
                      density=1,
                      alpha=0.2,
                      label="valor2")

saques["valor3"].hist(bins=[10, 20, 50, 100], 
                      density=1,
                      alpha=0.2,
                      label="valor3")

saques["valor4"].hist(bins=[10, 20, 50, 100], 
                      density=1,
                      alpha=0.2,
                      label="valor4")

saques["valor5"].hist(bins=[10, 20, 50, 100], 
                      density=1,
                      alpha=0.2,
                      label="valor5")

plt.legend()
plt.show()
