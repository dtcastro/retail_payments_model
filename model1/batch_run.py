# -*- coding: utf-8 -*-
"""
Created on Sat May 11 10:50:41 2019

@author: dtcas
"""
from mesa.batchrunner import batch_run #BatchRunner
import numpy as np
import pandas as pd
from model import CashManagementModel, get_observed_proportion_cash_payments
#, get_proportion_cash_payments, get_difference_proportion_cash, 
import dill

fixed_params = {"num_payers": 1519,
                "initial_cash_balance_m": 0}
variable_params = {"threshold_cash_balance_mth": #list(range(1, 10, 1)), 
                                                 list(range(10, 20, 2)), 
#                                                 list(range(20, 50, 5)) +
#                                                 list(range(50, 100, 10)) +
#                                                 list(range(100, 200, 20)) +
#                                                 list(range(200, 500, 50)) +
#                                                 list(range(500, 1000, 100)),
                   "withdrawal_distribution": [1, 2, 3, 4, 5]}

params = {"num_payers": 1519,
          "initial_cash_balance_m": 0,
          "threshold_cash_balance_mth": list(range(1, 10, 1)), 
#                                                 list(range(10, 20, 2)) + 
#                                                 list(range(20, 50, 5)) +
#                                                 list(range(50, 100, 10)) +
#                                                 list(range(100, 200, 20)) +
#                                                 list(range(200, 500, 50)) +
#                                                 list(range(500, 1000, 100)),
            "withdrawal_distribution": [1, 2, 3, 4, 5]}

print(params)

# batch_run = BatchRunner(CashManagementModel,
#                         fixed_parameters = fixed_params,
#                         variable_parameters = variable_params,
#                         iterations = 10,
#                         max_steps = 11,
#                         model_reporters={"proportion_cash_payments": get_proportion_cash_payments,
#                                          "difference_proportion_cash_payments": get_difference_proportion_cash})
#                                          #"Number discouting merchants": get_number_discounting_merchants})

results = batch_run(
    CashManagementModel,
    parameters=params,
    iterations=10,
    max_steps=11,
    number_processes=None, #Set it to None to use all the available processors
    data_collection_period=-1, #only at the end of each episode.
    display_progress=True,
)

#batch_run.run_all()
#run_data = batch_run.get_model_vars_dataframe()

run_data = pd.DataFrame(results)


with open("results/run_data_2.pkl",'wb') as f:   
    dill.dump(run_data, f)

print(run_data[['threshold_cash_balance_mth', 'difference_proportion_cash_payments']])
print(run_data.info())
#min_value = run_data['difference_proportion_cash_payments'].min()
min_index = np.argmin(run_data['difference_proportion_cash_payments']) 
print(min_index)
print(run_data.iloc[min_index])
print(run_data.iloc[min_index, 2])


# plotar um sobre o outro
obs_cash = get_observed_proportion_cash_payments()

with open("results/obs_cash.pkl",'wb') as f:   
    dill.dump(obs_cash, f)
