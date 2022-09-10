# -*- coding: utf-8 -*-
"""
- Nessa versão, a ideia é testar se o modelo de cash first optimal choice se
aplica ao caso brasileiro e se, com a introdução de um instrumento instantâneo,
como se alteraria o modelo; outra possibilidade é ver se, com mais locais de
saque, como no pix saque, o modelo se altera
- Instruments:
    - Não tem classe instrumento
    - Apenas dinheiro e cartão [de crédito?]; posteriormente pix
- ATM:
    - Tem uma localização; aí podemos colocar que o consumer só saca se tiver
    ATM perto; senão não saca
- Consumers:
    - Têm instrumento preferido (atributos não levados em conta)
    - Compram do merchant mais próximo


@author: Daniel

"""
import random
import numpy as np
import pandas as pd
from mesa import Agent, Model
#from mesa.time import BaseScheduler
#from mesa.time import RandomActivation
from mesa.time import StagedActivation
from mesa.datacollection import DataCollector

#cash_payments_history = pd.Series()
#card_payments_history = pd.Series()
#distribution_p = np.array(0)
#distribution_w = np.array(0)

PRINTF = False
      
class Payer(Agent):
    """A Consumer  
    attributes: cash_balance_m;
    behavior: withdraw money and makes a purchase with either cash or card
    """
    def __init__(self, unique_id, model, cash_balance_m):
        super().__init__(unique_id, model)
        self.cash_balance_m = cash_balance_m
        
    def period1(self):
        """
        In the first subperiod, a representative agent decides whether to make
        a cash withdrawal. In accordance with the minimum cash holdings policy,
        he/she does so only if the level of his cash holdings is lower than 
        mth. In this case, the agent draws by chance an amount from a 
        distribution of cash with- drawals observed in the economy. In doing 
        so, we acknowledge that people have different withdrawal costs that 
        give rise to different cash withdrawal amounts; the simulations take 
        into account such hetero- geneity, which is specific to each economy. 
        We denote by W the support of the empirical distribution of cash 
        withdrawals, and by πw(W) the empirical density function of a cash 
        withdrawal w.
        """
        if PRINTF:
            print('Consumer ' + str(self.unique_id) + " " + str(self.cash_balance_m) + ' step1')       
        
        if self.cash_balance_m < self.model.threshold_cash_balance_mth:
            w = self.withdraw() # aqui tem que ter a distribuição de saques
            self.cash_balance_m = self.cash_balance_m + w
               
    def period2(self):
        """
        Next, in the second subperiod, the agent is confronted with a
transaction opportunity of size p. Departing from the standard assumptions 
in inventory models set up in continuous time and on exogenous consumption 
flows, we assume that transactions are of different size and uncertain but 
still exogenous. In other words, the agent is supposed to be well informed 
of the different transaction sizes he/she can face, but cannot correctly 
anticipate their timing. Thus the agent draws a random transaction size from 
the observed distribution of transactions in the economy, and decides which
 payment instrument to use according to the Cash-first optimal choice. If the
 agent has enough cash on hand, he/she uses cash; otherwise, he/she uses a 
 payment card. We let D refer to the support of the empirical distribution of transactions, and πp( )D
to the empirical density function at transaction size p.
        """
        if PRINTF:
            print('Payer ' + str(self.unique_id) + ' step2')
            
        p = self.pay()
        if p < self.cash_balance_m:
            #pay cash
            self.cash_balance_m = self.cash_balance_m - p
            self.update_cash_payments_history(p)
            # we measure the share of cash payments by transaction size
            # depois que rodar a simulação:
            # We define the indicator G(m), which measures more precisely, 
            # for a given threshold m, the percentage error between the 
            # predicted shares of cash payments and the observed shares of cash 
        else:
            #pay card
            self.update_card_payments_history(p)
            
    def update_cash_payments_history(self, p):
        self.model.cash_payments_history = np.concatenate((self.model.cash_payments_history, np.array(p)), axis=0)
        if PRINTF:
            print("self.model.cash_payments_history")
            print(self.model.cash_payments_history)
        
    def update_card_payments_history(self, p):
        self.model.card_payments_history = np.concatenate((self.model.card_payments_history, np.array(p)), axis=0)
        if PRINTF:
            print("self.model.card_payments_history")
            print(self.model.card_payments_history)

    def withdraw(self):
        '''
        Consumer withdraws money based on distribution
    
        Returns
        -------
        Amount withdrawn
    
        '''
        return random.choice(self.model.distribution_w)
    
    def pay(self):
        '''
        Consumer makes a purchase of a certain value.
    
        Returns
        -------
        Value p of the purchase.
    
        '''
       
        return random.choice(self.model.distribution_p)        
    
def initialize_purchases_distribution():
    diary = pd.read_csv("../../pagtos_varejo_escolha/dados/export/diario_sub.csv", sep = ';', thousands='.', decimal=',')
    return diary[['valor']].to_numpy()

def get_observed_proportion_by_value():
    diary = pd.read_csv("../../pagtos_varejo_escolha/dados/export/diario_sub.csv", sep = ';', thousands='.', decimal=',')
    max = diary['valor'].max()
    interval_range = pd.interval_range(start=0, freq=2, end=max)

    interval_total = pd.cut(diary['valor'], bins=interval_range) #, right = False)
    dist_total = interval_total.value_counts(normalize=True, sort=False).to_frame()
    return dist_total    

def get_observed_proportion_cash_payments():
    diary = pd.read_csv("../../pagtos_varejo_escolha/dados/export/diario_sub.csv", sep = ';', thousands='.', decimal=',')
    max = diary['valor'].max()
    interval_range = pd.interval_range(start=0, freq=2, end=max)

    interval_total = pd.cut(diary['valor'], bins=interval_range) #, right = False)
    dist_total = interval_total.value_counts(normalize=False, sort=False).to_frame()

    diary_cash = diary.loc[diary['eletronico'] == False, 'valor']
    if PRINTF:
        print("diary_cash")
        print(diary_cash)
    interval_cash = pd.cut(diary_cash, bins=interval_range) #, right = False)
    dist_cash = interval_cash.value_counts(normalize=False, sort=False).to_frame()

#    diary_card = diary.loc[diary['eletronico'] == True, 'valor']
#    interval_card = pd.cut(diario_card['valor'], bins=interval_range) #, right = False)
#    dist_card = interval_card.value_counts(normalize=False, sort=False).to_frame()

    observed_prop = dist_cash/dist_total
    #print("observed prop")
    #print(observed_prop)
    return observed_prop.fillna(0)

def initialize_withdrawals_distribution(withdrawal_distribution=1):
    label_dist = 'valor' + str(withdrawal_distribution)
    saques = pd.read_csv("../data/saques_abm.csv", sep = ';')
    dist_w = saques[[label_dist]].dropna().to_numpy()
    return dist_w

def get_proportion_cash_payments(model):
    """proportion of cash payments"""
    # aqui tem que pegar historico de pagamentos com dinheiro e cartão e 
    # ver a proporção por valor de p
    #max_p_cash = model.cash_payments_history.max(initial = 0)
    #max_p_card = model.card_payments_history.max(initial = 0)
    #max_p = max(max_p_cash, max_p_card)
    #if math.isnan(max_p):
    #    max_p = 0
    
    max_p = 8000 # valor do diario
    
    interval_range = pd.interval_range(start=0, freq=2, end=max_p)
    interval_p_cash = pd.cut(pd.Series(model.cash_payments_history), bins=interval_range) #, right = False)
    interval_p_card = pd.cut(pd.Series(model.card_payments_history), bins=interval_range) #, right = False)

    dist_cash = interval_p_cash.value_counts(normalize=False, sort=False).to_frame()
    dist_card = interval_p_card.value_counts(normalize=False, sort=False).to_frame()
    dist_total = dist_cash + dist_card
    prop_cash = (dist_cash/dist_total).fillna(0)

    if PRINTF:
        print('dist_cash')
        print(dist_cash)
        print('dist_card')
        print(dist_card)
        print('dist_total')
        print(dist_total)
        print('prop_cash')
        print(prop_cash)
    
    return prop_cash

def get_difference_proportion_cash(model):
    observed_proportion_cash_payments = model.observed_prop_cash
    #print("data collector")
    #print(type(model.datacollector))
    proportion_cash_payments = get_proportion_cash_payments(model) #['proportion_cash_payments']
    
#    proportion_cash_payments = model.datacollector.get_model_vars_dataframe() #['proportion_cash_payments']
    prop_difference = abs(proportion_cash_payments.iloc[:, 0] - observed_proportion_cash_payments['valor'])
    
    prop_difference_weighted = prop_difference * model.weights['valor']

    total_difference = prop_difference_weighted.sum()

    if PRINTF:
        print("prop cash payments")
        print(proportion_cash_payments)
        print("prop difference")
        print(prop_difference)
#        print("weights")
#        print(weights)
        
    return total_difference

class CashManagementModel(Model):
    """A model with some number of agents."""
    def __init__(self, num_payers, 
                 initial_cash_balance_m, 
                 threshold_cash_balance_mth, 
                 withdrawal_distribution):
        self.running = True
        self.num_payers = num_payers
        self.threshold_cash_balance_mth = threshold_cash_balance_mth
        self.cash_payments_history = np.empty(0)
        self.card_payments_history = np.empty(0)

        #self.schedule = RandomActivation(self)
        #self.schedule = BaseScheduler(self)
        self.schedule = StagedActivation(self, ["period1", "period2"])
        
        unique_id = 0
        
        self.distribution_w = initialize_withdrawals_distribution(withdrawal_distribution)
        #print(self.distribution_w)
        #print("w:")
        #print(self.distribution_w[15])
        #print(random.choice(self.distribution_w))
        self.distribution_p = initialize_purchases_distribution()
        #print(self.distribution_p)
        self.observed_prop_cash = get_observed_proportion_cash_payments()
        self.weights = get_observed_proportion_by_value()

        # At t = 0, the representative agent is initialized with zero cash balances.
        
        # Create consumers
        for i in range(self.num_payers):
            payer = Payer(unique_id, self, initial_cash_balance_m)
            unique_id += 1
            self.schedule.add(payer)
            
        self.datacollector = DataCollector(
            model_reporters={"proportion_cash_payments": get_proportion_cash_payments,
                             "difference_proportion_cash_payments": get_difference_proportion_cash},
            agent_reporters={})
    
        self.datacollector.collect(self)

    def step(self):
        '''Advance the model by one step.'''
        self.schedule.step()
        self.datacollector.collect(self)
        