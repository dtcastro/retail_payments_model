# -*- coding: utf-8 -*-
"""
Created on Sat Sep  1 10:40:37 2018

@author: dtcas
"""

from mesa.batchrunner import BatchRunner
from model import RetailPaymentsModel, Merchant  # omit this in jupyter notebooks

################# pra rodar sem o batch runner ##############
#model = RetailPaymentsModel(10, 2)
#for i in range(5):
#    print('rodada ' + str(i))
#    model.step()
#    for obj in model.schedule.agents:
#        if isinstance(obj, Merchant):
#            print('Lojista ' + str(obj.unique_id) + ' sales: ' + str(obj.sales) + ' lost sales: ' + str(obj.lost_sales))
##        if isinstance(obj, Portador):
##            print('Portador ' + str(obj.unique_id) + 'recursos: ' + str(obj.recursos))
#
#merchants_data = model.datacollector.get_agent_vars_dataframe()
#print(merchants_data.tail())

# TODO
# 1 COLOCAR O TEMPO -> SEJA COM O STAGEDSCHEDULER OU COM ALGUMA COISA COMO O
# DO CAUE
# 2 COLOCAR O ESPAÇO 
# 3 colocar o batch runner

################ pra rodar com o batch runner

fixed_params = {"num_consumers": 10,
                "consumer_discount_prob": 0.5}
variable_params = {"num_merchants": range(1, 3, 1)}

batch_run = BatchRunner(RetailPaymentsModel,
                        fixed_parameters = fixed_params,
                        variable_parameters = variable_params,
                        iterations = 2,
                        max_steps = 5,
                        model_reporters={})

batch_run.run_all()