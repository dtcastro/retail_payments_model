# -*- coding: utf-8 -*-
"""
Created on Sat May 11 10:50:41 2019

@author: dtcas
"""
from mesa.batchrunner import BatchRunner
from model import RetailPaymentsModel, get_total_sales, get_number_discounting_merchants

fixed_params = {"num_consumers": 10,
                "consumer_discount_prob": 0.5}
variable_params = {"num_merchants": range(1, 4, 1)}

batch_run = BatchRunner(RetailPaymentsModel,
                        fixed_parameters = fixed_params,
                        variable_parameters = variable_params,
                        iterations = 3,
                        max_steps = 5,
                        model_reporters={"Total sales": get_total_sales,
                                         "Number discouting merchants": get_number_discounting_merchants})

batch_run.run_all()
run_data = batch_run.get_model_vars_dataframe()