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
#from mesa.time import RandomActivation
from mesa.time import StagedActivation
from mesa.datacollection import DataCollector

CASH = 0
DEBIT = 1
ratio_threshold_for_discounting = 0.25

class Merchant(Agent):
    """A Merchant
    attributes: 
        favorite_instrument
        discounts: determines if the merchant does or not offer discount based
            on the instrument
        sales: number of sales made
        lost_sales: number of sales lost because the merchant does not accept
            the favorite instrument of the consumer
    """
    def __init__(self, unique_id, model, favorite_instrument, discounts):
        super().__init__(unique_id, model)
        self.sales = 0
        self.lost_sales = 0
        self.favorite_instrument = favorite_instrument
        self.discounts = discounts
        self.lost_sales_ratio = 0
        
    def period1(self):
        print('Merchant ' + str(self.unique_id) + ' step1')
        print('Discounts: ' + str(self.discounts))

    def period2(self):
        print('Merchant ' + str(self.unique_id) + ' step2')
        self.lost_sales_ratio = self.lost_sales/(self.lost_sales + self.sales)
        if self.lost_sales_ratio > ratio_threshold_for_discounting:
            self.discounts = 1
        
class Consumer(Agent):
    """A Consumer  
    attributes: favorite instrument, changes instrument because of a discount
    """
    def __init__(self, unique_id, model, favorite_instrument, discounts):
        super().__init__(unique_id, model)
        self.favorite_instrument = favorite_instrument
        self.discounts = discounts
        
    def period1(self):
        print('Consumer ' + str(self.unique_id) + str(self.favorite_instrument) + 'step1')
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
    
    def period2(self):
        print('Consumer ' + str(self.unique_id) + ' step2')    
        
def get_total_sales(model):
    """total number of sales"""
    total_sales = sum(obj.sales for obj in model.schedule.agents if isinstance(obj, Merchant))
    #rich_agents = [a for a in model.schedule.agents if a.savings > model.rich_threshold]
    return total_sales

def get_total_lost_sales(model):
    """total number of sales"""
    total_lost_sales = sum(obj.lost_sales for obj in model.schedule.agents if isinstance(obj, Merchant))
    #rich_agents = [a for a in model.schedule.agents if a.savings > model.rich_threshold]
    return total_lost_sales

def get_number_discounting_merchants(model):
    """number of discounting merchants"""
    discounting_merchants = sum([obj.discounts for obj in model.schedule.agents if isinstance(obj, Merchant)])
    return discounting_merchants

def get_number_merchants_above_ratio(model):
    """number of discounting merchants"""
    merchants_above_ratio = [obj for obj in model.schedule.agents 
                             if isinstance(obj, Merchant) and obj.lost_sales_ratio > ratio_threshold_for_discounting]
    return len(merchants_above_ratio)

def get_number_discounting_consumers(model):
    """number of discounting consumers"""
    discounting_consumers = sum([obj.discounts for obj in model.schedule.agents if isinstance(obj, Consumer)])
    return discounting_consumers

class RetailPaymentsModel(Model):
    """A model with some number of agents."""
    def __init__(self, num_consumers, num_merchants, consumer_discount_prob):
        self.running = True
        self.num_consumers = num_consumers
        self.num_merchants = num_merchants
        #self.schedule = RandomActivation(self)
        #self.schedule = BaseScheduler(self)
        self.schedule = StagedActivation(self, ["period1", "period2"])
        #total_sales = 0
        
        unique_id = 0
        
        # Create consumers
        for i in range(self.num_consumers):
            favorite_instrument = numpy.random.binomial(1, .5, 1) # n, p, number of trials
            discounts = numpy.random.binomial(1, consumer_discount_prob, 1) # n, p, number of trials
            consumers = Consumer(unique_id, self, favorite_instrument, discounts)
            unique_id += 1
            self.schedule.add(consumers)
            
        # Create merchants
        for i in range(self.num_merchants):
            favorite_instrument = numpy.random.binomial(1, .25, 1) # n, p, number of trials => 75% prefer CASH (0)
            discounts = numpy.random.binomial(1, .25, 1) # n, p, number of trials => 90% do not discount
            merchants = Merchant(unique_id, self, favorite_instrument, discounts)
            unique_id += 1
            self.schedule.add(merchants)
            
        self.datacollector = DataCollector(
            model_reporters={"Total sale": get_total_sales,
                             "Total lost sales": get_total_lost_sales,
                             "Number of discounting merchants": get_number_discounting_merchants,
                             "Number of discounting consumers": get_number_discounting_consumers,
                             "Number of merchants above ratio": get_number_merchants_above_ratio},
            agent_reporters={"Lost sales": lambda a: getattr(a, 'lost_sales', None),
                             "Sales": lambda a: getattr(a, 'sales', None),
                             "Lost sales ratio": lambda a: getattr(a, 'lost_sales_ratio', None)})
    
        self.datacollector.collect(self)

    def step(self):
        '''Advance the model by one step.'''
        self.schedule.step()
        self.datacollector.collect(self)
        