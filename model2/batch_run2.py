# -*- coding: utf-8 -*-
"""
Created on Sat May 11 10:50:41 2019

@author: dtcas
"""
from mesa.batchrunner import BatchRunner
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from model2 import CashManagementModel2, get_proportion_instruments_payments, get_difference_proportion_instruments, initialize_observed_proportion_instruments_payments, initialize_purchases_distribution, initialize_observed_proportion_by_value

purchases_distribution = initialize_purchases_distribution()
observed_proportion_purchases_by_value = initialize_observed_proportion_by_value()
observed_prop_cash, observed_prop_debit = initialize_observed_proportion_instruments_payments()

fixed_params = {"num_payers": 1519,
                "initial_cash_balance_m": 0,
                "initial_account_balance_m": 50,
                "purchases_distribution": purchases_distribution,
                "observed_proportion_purchases_by_value": observed_proportion_purchases_by_value,
                "observed_prop_cash": observed_prop_cash, 
                "observed_prop_debit": observed_prop_debit}

variable_params = {"threshold_cash_balance_mth": #list(range(1, 10, 1)) + 
                                                 #list(range(10, 20, 2)) + 
                                                 #list(range(20, 50, 5)) +
                                                 #list(range(50, 100, 10)) +
                                                 list(range(10, 100, 10)) + 
#                                                 list(range(100, 200, 20)) +
 #                                                list(range(200, 500, 50)) +
                                                 list(range(100, 500, 50)) +
                                                 list(range(500, 1000, 100)),
                   "threshold_account_balance_mth": #list(range(1, 10, 1)) + 
                                                 #list(range(10, 20, 2)) + 
                                                 #list(range(20, 50, 5)) +
                                                 #list(range(50, 100, 10)) +
                                                 #list(range(100, 200, 20)) +
                                                 list(range(200, 500, 50)) +
                                                 list(range(500, 1000, 100)),
#                                                 list(range(1000, 2000, 200)),
                    "number_withdrawal_distribution": [5],
                    "number_desinvestment_distribution": [2]}

batch_run = BatchRunner(CashManagementModel2,
                        fixed_parameters = fixed_params,
                        variable_parameters = variable_params,
                        iterations = 1,
                        max_steps = 11,
                        model_reporters={"proportion_instruments_payments": get_proportion_instruments_payments,
                                         "difference_proportion_instruments_payments": get_difference_proportion_instruments})
                                         #"Number discouting merchants": get_number_discounting_merchants})

batch_run.run_all()
run_data = batch_run.get_model_vars_dataframe()
print(run_data[['threshold_cash_balance_mth', 'threshold_account_balance_mth', 'difference_proportion_instruments_payments']])
print(run_data.info())
#min_value = run_data['difference_proportion_cash_payments'].min()
min_index = np.argmin(run_data['difference_proportion_instruments_payments']) 
print("min_index")
print(min_index)
#print(run_data.iloc[min_index])
print("valor min index")
print(run_data.iloc[min_index, 0])
print(run_data.iloc[min_index, 1])
print(run_data.iloc[min_index, 5])


# plotar um sobre o outro
obs_cash, obs_debit = initialize_observed_proportion_instruments_payments()
#print(obs_cash)
#obs_cash['valor'].plot.line()
#plt.show()

obs_cash_200 = obs_cash.iloc[0:50, 0]
obs_debit_200 = obs_debit.iloc[0:50, 0]

#obs_cash_200.plot.line(title = "Percentual de pagamentos em dinheiro")
#plt.show()

#sim_cash = run_data[['proportion_cash_payments']]
sim_cash, sim_debit = run_data.iloc[min_index, 6] # o que eu quero é o proportion_instruments_payments
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
sim_cash_200 = sim_cash.iloc[0:50,0]
sim_debit_200 = sim_debit.iloc[0:50,0]

#print(sim_cash_200)
#sim_cash_200.plot.line()
#plt.show()

df_difference = pd.DataFrame({
    'Observado dinheiro': obs_cash_200,
    'Observado débito': obs_debit_200,
    'Simulado dinheiro': sim_cash_200,
    'Simulado débito': sim_debit_200}, index = list(sim_cash_200.index))

df_difference.plot.line()
plt.show()

# graf_mth = run_data.plot(x = "threshold_cash_balance_mth", 
#                          y = "difference_proportion_instruments_payments",
#                          title = "Diferença entre simulado e observado para " + str(fixed_params.get('num_payers')) + " usuários")

# graf_mth.set_ylabel("Diferença % ponderada pelo nº de transações por valor")

# plt.show()

# TODO
# MELHORAR A DISTRIBUIÇÃO DO SAQUE -> ok
# SIMULAR COM VÁRIAS ITERAÇÕES E TRAÇAR GRÁFICO POR MTH
# SIMULAR COM 1519 usuários e 11 steps
# continuar vendo mais mths
# títulos dos gráficos e eixos e etc.
# testar com diferentes valores da distribuição de saques
# procurar na internet alguma distribuição de saques
# tirar o outlier de 8k
# aumentar os saques menores pra ver o que dá
# lance é criar um mth2 que a pessoa primeiro usa dinheiro se p < mth e depois usa debito ou pix se p < mth2; só então usa crédito
# criar uma classe merchant que aceita ou não e que pode ir aceitando aos poucos (como pix)
# ver como é holanda no paper
# gráfico pode ser uma série de boxplots se rodarmos com várias iterações
# fazer um gráfico para explicar a distribuição dos saques
# fazer um gráfico com obs vs simulado mostrando o efeito da dist saque