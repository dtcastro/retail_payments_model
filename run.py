# -*- coding: utf-8 -*-
"""
Created on Sat Sep  1 10:40:37 2018

@author: dtcas
"""

from model import RetailPaymentsModel, Merchant  # omit this in jupyter notebooks

model = RetailPaymentsModel(10, 2)
for i in range(5):
    print('step ' + str(i))
    model.step()
    for obj in model.schedule.agents:
        if isinstance(obj, Merchant):
            print('Lojista ' + str(obj.unique_id) + ' sales: ' + str(obj.sales) + ' lost sales: ' + str(obj.lost_sales))
#        if isinstance(obj, Portador):
#            print('Portador ' + str(obj.unique_id) + 'recursos: ' + str(obj.recursos))

merchants_lost_sales = model.datacollector.get_agent_vars_dataframe()
print(merchants_lost_sales.tail())

# TODO
# 1 COLOCAR O TEMPO -> SEJA COM O STAGEDSCHEDULER OU COM ALGUMA COISA COMO O
# DO CAUE
# 2 COLOCAR O ESPAÇO 
# 3 colocar o batch runner