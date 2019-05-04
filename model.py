# -*- coding: utf-8 -*-
"""
- Nessa versão, a ideia é testar se a concessão de desconto pelo lojista pode 
fazer com que o consumidor escolha um instrumento diferente do seu preferido.
- Instruments:
    - Não tem classe instrumento
    - Apenas dinheiro e cartão de débito
- Merchants:
    - Vendem um mesmo produto pelo mesmo preço
    - A única diferença é a concessão ou não de desconto
    - Concorrência perfeita
    - Quantidade de produtos que cada um vende é infinita
    - Têm instrumento preferido (atributos que levam a que determinado 
    instrumento seja o preferido não são levados em conta nesse momento)
    - Todos aceitam dinheiro; alguns aceitam débito também
    - Os que preferem dinheiro podem optar por dar desconto para que o 
    consumidor pague com esse instrumento
    - Os que preferem débito não dão desconto
    - No fim do período, contam quantas vendas perderam; se perderam mais que 
    n vendas, podem passar a aceitar débito ou a dar desconto
- Consumers:
    - Têm instrumento preferido (atributos não levados em conta)
    - Compram do merchant mais próximo
- IAP:
    - Decide o preço cobrado do lojista
    - Não cobra do consumidor

@author: Daniel

"""
import random
import numpy
from mesa import Agent, Model
#from mesa.time import BaseScheduler
from mesa.time import RandomActivation
#from mesa.time import StagedActivation
from mesa.datacollection import DataCollector

CASH = 0
DEBIT = 1

class Merchant(Agent):
    """An agent with fixed initial wealth."""
    def __init__(self, unique_id, model, favorite_instrument, discounts):
        super().__init__(unique_id, model)
        self.sales = 0
        self.lost_sales = 0
        self.favorite_instrument = favorite_instrument
        self.discounts = discounts
        
    def step1(self):
        print('Merchant ' + str(self.unique_id) + ' step1')    

    def step2(self):
        print('Merchant ' + str(self.unique_id) + ' step2')  
        
class Consumer(Agent):
    """An agent with fixed initial wealth."""
    def __init__(self, unique_id, model, favorite_instrument, discounts):
        super().__init__(unique_id, model)
        self.favorite_instrument = favorite_instrument
        self.discounts = discounts
        
    def step(self):
        print('Consumer ' + str(self.unique_id) + str(self.favorite_instrument))
        # chooses a merchant 
        merchants = [obj for obj in self.model.schedule.agents if isinstance(obj, Merchant)]
        merchant = random.choice(merchants)
        #print('lojista ' + str(merchant.unique_id))
        
        if merchant.favorite_instrument == self.favorite_instrument:
            #print('buy')
            merchant.sales += 1
        else:
            if merchant.favorite_instrument == CASH and self.favorite_instrument == DEBIT:
                if merchant.discounts and self.discounts:
                    merchant.sales += 1
                else:
                    merchant.lost_sales += 1
            else: # merchant.favorite_instrument == DEBIT and self.favorite_instrument == CASH
                merchant.sales += 1
                
    # preciso de um step 2 para o aprendizado; se lost_sales  > x; discounts
    
    def step1(self):
        print('Consumer ' + str(self.unique_id) + ' step1')    
        
    def step2(self):
        print('Consumer ' + str(self.unique_id) + ' step2')    
        
class RetailPaymentsModel(Model):
    """A model with some number of agents."""
    def __init__(self, M, N):
        self.num_consumers = M
        self.num_merchants = N
        self.schedule = RandomActivation(self)
        #self.schedule = BaseScheduler(self)
        #self.schedule = StagedActivation(self, ["step1", "step2"])
        
        unique_id = 0
        
        # Create consumers
        for i in range(self.num_consumers):
            favorite_instrument = numpy.random.binomial(1, .5, 1) # n, p, number of trials
            discounts = numpy.random.binomial(1, .5, 1) # n, p, number of trials
            consumers = Consumer(unique_id, self, favorite_instrument, discounts)
            unique_id += 1
            self.schedule.add(consumers)
            
        # Create merchants
        for i in range(self.num_merchants):
            favorite_instrument = numpy.random.binomial(1, .25, 1) # n, p, number of trials => 75% prefer CASH (0)
            discounts = numpy.random.binomial(1, .9, 1) # n, p, number of trials => 90% do not discount
            merchants = Merchant(unique_id, self, favorite_instrument, discounts)
            unique_id += 1
            self.schedule.add(merchants)
            
        self.datacollector = DataCollector(
            agent_reporters={"Lost sales": lambda a: getattr(a, 'lost_sales', None),
                             "Sales": lambda a: getattr(a, 'sales', None)})
    
        self.datacollector.collect(self)

    def step(self):
        '''Advance the model by one step.'''
        self.schedule.step()
        self.datacollector.collect(self)
        