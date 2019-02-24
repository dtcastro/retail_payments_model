# -*- coding: utf-8 -*-
"""
Created on Sat Sep  1 10:40:37 2018

@author: dtcas
"""

from model import RetailPaymentsModel, Merchant, Portador  # omit this in jupyter notebooks

model = RetailPaymentsModel(10, 2)
for i in range(10):
    model.step()
    for obj in model.schedule.agents:
        if isinstance(obj, Merchant):
            print('Lojista ' + str(obj.unique_id) + ' produtos: ' + str(obj.produtos) + 'recursos: ' + str(obj.recursos))
        if isinstance(obj, Portador):
            print('Portador ' + str(obj.unique_id) + 'recursos: ' + str(obj.recursos))
            