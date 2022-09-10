# -*- coding: utf-8 -*-
"""
Created on Sat May 11 10:50:41 2019

@author: dtcas
"""
from model2 import initialize_observed_proportion_instruments_payments
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import dill


with open("results/run_data_model2_w5_d2_it1.pkl",'rb') as f:
    run_data1 = dill.load(f)

#print(run_data1.info())

with open("results/run_data_model2_w5_d2_it2.pkl",'rb') as f:
    run_data2 = dill.load(f)

#print(run_data2.info())

with open("results/run_data_model2_w5_d2_it3.pkl",'rb') as f:
    run_data3 = dill.load(f)

#print(run_data3.info())

with open("results/run_data_model2_w5_d2_it4.pkl",'rb') as f:
    run_data4 = dill.load(f)

#print(run_data4.info())


frames = [run_data1, run_data2, run_data3, run_data4]
  
run_data = pd.concat(frames, ignore_index=True)
#print(run_data.info())
    
######################## 

### Best mth
print(run_data[['threshold_cash_balance_mth', 'threshold_account_balance_mth', 'difference_proportion_instruments_payments']])
print(run_data.info())
#min_value = run_data['difference_proportion_cash_payments'].min()
min_index = np.argmin(run_data['difference_proportion_instruments_payments']) 
print("min_index")
print(min_index)
#print(run_data.iloc[min_index])
print("valor min index")
print(run_data.iloc[min_index, 6])
print(run_data.iloc[min_index, 7])
print(run_data.iloc[min_index, 11])

#### Best mean

print('médias')
mean_differences_mth = run_data.groupby('threshold_cash_balance_mth')['difference_proportion_instruments_payments'].mean()
print(mean_differences_mth)
min_index_means = np.argmin(mean_differences_mth) 
print(mean_differences_mth.iloc[min_index_means])

df_graph = run_data.groupby('threshold_cash_balance_mth').agg({'difference_proportion_instruments_payments': ['mean', 'min', 'max']})
print(df_graph)

##############
## Figure 2

# a
sim_mth = run_data[['threshold_cash_balance_mth', 
                    'threshold_account_balance_mth', 
                    'difference_proportion_instruments_payments']]


fig2a = sns.lineplot(data = sim_mth,
             x = 'threshold_cash_balance_mth',
             y = 'difference_proportion_instruments_payments',
             ci=95,
             markers=True)

fig2a.set(title="Model 2 error by $m^{th}$")
fig2a.set_xlabel("$m^{th}$")
fig2a.set_ylabel("Difference (%) between observed and simulated")

plt.savefig('model2_sim_cash_mth.pdf', figsize=(6,6)) # ver o melhor fig_size
plt.show()

fig3a = sns.lineplot(data = sim_mth,
             x = 'threshold_account_balance_mth',
             y = 'difference_proportion_instruments_payments',
             ci=95,
             markers=True)

fig3a.set(title="Model 2 error by $m2^{th}$")
fig3a.set_xlabel("$m2^{th}$")
fig3a.set_ylabel("Difference (%) between observed and simulated")

plt.savefig('model2_sim_account_mth.pdf', figsize=(6,6)) # ver o melhor fig_size
plt.show()

### b
sim_mth_zoom = sim_mth[sim_mth['threshold_cash_balance_mth'] > 99]
print(sim_mth_zoom.head())

fig2b = sns.lineplot(data = sim_mth_zoom,
             x = 'threshold_cash_balance_mth',
             y = 'difference_proportion_instruments_payments',
             ci=95,
             markers=True)

fig2b.set(title="Model 2 error by $m^{th}$ above R\$100")
fig2b.set_xlabel("$m^{th}$")
fig2b.set_ylabel("Difference (%) between observed and simulated")

plt.savefig('model2_sim_cash_mth_zoom.pdf', figsize=(6,6)) # ver o melhor fig_size
plt.show()

fig3b = sns.lineplot(data = sim_mth_zoom,
             x = 'threshold_account_balance_mth',
             y = 'difference_proportion_instruments_payments',
             ci=95,
             markers=True)

fig3b.set(title="Model 2 error by $m2^{th}$, for $m^{th}$ above R\$100")
fig3b.set_xlabel("$m2^{th}$")
fig3b.set_ylabel("Difference (%) between observed and simulated")

plt.savefig('model2_sim_account_mth_zoom.pdf', figsize=(6,6)) # ver o melhor fig_size
plt.show()

#long
sim_mth_long = pd.melt(sim_mth, id_vars=['difference_proportion_instruments_payments'],
                       value_vars=['threshold_cash_balance_mth', 'threshold_account_balance_mth'])

### não faz sentido mostrar pq a média do account balance fica alta pq pega quando cash_mth é baixo
sns.lineplot(data = sim_mth_long,
             x = 'value',
             y = 'difference_proportion_instruments_payments',
             hue = 'variable',
             ci=95,
             markers=True)

plt.show()

model2_sim_scatter = sns.scatterplot(data=sim_mth, 
                x="threshold_cash_balance_mth", 
                y="threshold_account_balance_mth", 
                size="difference_proportion_instruments_payments")

plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
model2_sim_scatter.set(title="Model 2 error by $m^{th}$ and $m2^{th}$")
model2_sim_scatter.set_xlabel("$m^{th}$")
model2_sim_scatter.set_ylabel("$m2^{th}$")

plt.tight_layout()
plt.savefig('model2_sim_scatter.pdf', figsize=(6,6)) # ver o melhor fig_size
plt.show()

##############
## Figure 3 - Comp observed x simulated
obs_cash, obs_debit = initialize_observed_proportion_instruments_payments()

obs_cash_200 = obs_cash.iloc[0:51, 0]
obs_debit_200 = obs_debit.iloc[0:51, 0]

sim_cash, sim_debit = run_data.iloc[min_index, 10] # o que eu quero é o proportion_instruments_payments
print('sim_cash')
#print(str(sim_cash))
print(sim_cash)
#print(sim_cash.info())
#print('1')
#print(sim_cash['proportion_cash_payments'])
#sim_cash = pd.to_numeric(sim_cash) 
#print('2')
#print(sim_cash['proportion_cash_payments'][0])
#sim_cash['proportion_cash_payments'][0].iloc[0:50,0].plot.line()
#plt.show()

#sim_200 = sim_cash['proportion_cash_payments'][0].iloc[0:50,0]
sim_cash_200 = sim_cash.iloc[0:51,0]
sim_debit_200 = sim_debit.iloc[0:51,0]

#print(sim_cash_200)
#sim_cash_200.plot.line()
#plt.show()

df_difference = pd.DataFrame({
    'Observed cash': obs_cash_200,
    'Observed debit': obs_debit_200,
    'Simulated cash': sim_cash_200,
    'Simulated debit': sim_debit_200}, index = list(sim_cash_200.index))

df_difference.plot.line(title = "Percentage of cash and debit payments per value in model 2")
plt.legend(loc='center left', bbox_to_anchor=(0.0, 0.5))
plt.savefig('model2_dist_comp.pdf', figsize=(6,7)) # ver o melhor fig_size
plt.show()

# df_difference_total = pd.DataFrame({
#     'Observed cash': obs_cash,
#     'Observed debit': obs_debit,
#     'Simulated cash': sim_cash,
#     'Simulated debit': sim_debit}, index = list(sim_cash.index))

# df_difference_total.plot.line(title = "Percentage of cash and debit payments per value")
# plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
# plt.savefig('model2_dist_comp_total.pdf', figsize=(6,7)) # ver o melhor fig_size
# plt.show()

#### Figura desaplicações
desaplicacoes = pd.read_csv("../data/desaplicacoes_abm.csv", sep = ';')
#dist_w = saques[[label_dist]].dropna().to_numpy()
desaplicacoes_dist_1 = desaplicacoes["valor1"].hist(bins=[10, 20, 50, 100, 200, 500, 1000], #, 5000], 
                      density=1,
                      alpha=0.5)
plt.show()

desaplicacoes_dist_1 = sns.histplot(data=desaplicacoes, x='valor1', stat='probability')
desaplicacoes_dist_1.set(xlabel ="Transfer value", ylabel = "Frequency (%)", title ='Transfers to payment account distribution 1')

plt.savefig('desaplicacoes_dist_1.pdf') # ver o melhor fig_size
plt.legend()
plt.show()

desaplicacoes_dist_2 = desaplicacoes["valor2"].hist(bins=[10, 20, 50, 100, 200, 500, 1000], #, 5000], 
                      density=1,
                      alpha=0.5,
                      label="valor2")
plt.show()

desaplicacoes_dist_2 = sns.histplot(data=desaplicacoes, x='valor2', stat='probability')
desaplicacoes_dist_2.set(xlabel ="Transfer value", ylabel = "Frequency (%)", title ='Transfers to payment account distribution 2')

plt.savefig('desaplicacoes_dist_2.pdf') # ver o melhor fig_size
plt.legend()
plt.show()