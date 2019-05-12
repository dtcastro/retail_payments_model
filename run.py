# -*- coding: utf-8 -*-
"""
Created on Sat Sep  1 10:40:37 2018

@author: dtcas
"""

from model import RetailPaymentsModel, Merchant

################# pra rodar sem o batch runner ##############
model = RetailPaymentsModel(num_consumers = 10, num_merchants = 2, 
                            consumer_discount_prob = 0.5)
for i in range(5):
    """ Calls step n times"""
    print('rodada ' + str(i))
    model.step()
    for obj in model.schedule.agents:
        if isinstance(obj, Merchant):
            print('Lojista ' + str(obj.unique_id) + ' sales: ' + str(obj.sales) + ' lost sales: ' + str(obj.lost_sales))
#        if isinstance(obj, Portador):
#            print('Portador ' + str(obj.unique_id) + 'recursos: ' + str(obj.recursos))

merchants_data = model.datacollector.get_agent_vars_dataframe()
print(merchants_data.tail())

model_data = model.datacollector.get_model_vars_dataframe()

# TODO
# 2 COLOCAR O ESPAÇO 
# 3 colocar o batch runner


