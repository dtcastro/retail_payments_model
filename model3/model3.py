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

class Payee(Agent):
    """A Merchant
    attributes: 
        favorite_instrument
        discounts: determines if the merchant does or not offer discount based
            on the instrument
        sales: number of sales made
        lost_sales: number of sales lost because the merchant does not accept
            the favorite instrument of the consumer
    """
    def __init__(self, 
                 unique_id, 
                 model, 
                 accepts_pix,
                 accepts_debit,
                 accepts_credit):
        super().__init__(unique_id, model)
        self.payments = 0
        self.lost_payments = 0
        self.accepts_pix = accepts_pix
        self.accepts_debit = accepts_debit
        self.accepts_credit = accepts_credit
#        self.lost_payments_ratio = 0
        self.payments = 0
        
    def period1(self):
        print('Merchant ' + str(self.unique_id) + ' step1')
#        print('Discounts: ' + str(self.discounts))

    def period2(self):
        print('Merchant ' + str(self.unique_id) + ' step2')
        if(self.lost_payments + self.payments > 0):
            lost_payments_ratio = self.lost_payments/(self.lost_payments + self.payments)
        
            if lost_payments_ratio > self.model.ratio_threshold_for_accepting:
                self.accepts_pix = 1
                print('passa a aceitar')
       
  
class Payer(Agent):
    """A Consumer  
    attributes: cash_balance_m;
    behavior: withdraw money and makes a purchase with either cash or card
    """
    def __init__(self, 
                 unique_id, 
                 model, 
                 cash_balance_m, 
                 account_balance_m,
                 uses_pix,
                 prefers_debit):
        super().__init__(unique_id, model)
        self.cash_balance_m = cash_balance_m
        self.account_balance_m = account_balance_m
        self.uses_pix = uses_pix
        self.prefers_debit = prefers_debit
        self.lost_payments = 0
        self.cash_payments_history = np.empty(0)
        self.debit_payments_history = np.empty(0)
        self.pix_payments_history = np.empty(0)
        self.credit_payments_history = np.empty(0)

        
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
        
        if self.account_balance_m < self.model.threshold_account_balance_mth:
            w2 = self.desinvest() # aqui tem que ter a distribuição de saques
            self.account_balance_m = self.account_balance_m + w2
            
        # começa a usar pix
            
        n_payments = self.cash_payments_history.size + self.debit_payments_history.size + self.pix_payments_history.size + self.credit_payments_history.size
                     
        if(self.lost_payments + n_payments > 0):
            lost_payments_ratio = self.lost_payments/(self.lost_payments + n_payments)

            if (lost_payments_ratio > self.model.ratio_threshold_for_using):
                self.uses_pix = True
                print('passa a usar')
            
        # prefere pix
        if (self.pix_payments_history.size > self.debit_payments_history.size):
            self.prefers_debit = False
               
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
            
        # chooses a merchant 
        payees = [obj for obj in self.model.schedule.agents if isinstance(obj, Payee)]
        payee = random.choice(payees)
            
        p = self.pay()
        print("p:")
        print(p)
        if p < self.cash_balance_m:
            #pay cash
            self.cash_balance_m = self.cash_balance_m - p
            self.update_cash_payments_history(p)
            payee.payments = payee.payments + 1
            print("cash")
            
            # we measure the share of cash payments by transaction size
            # depois que rodar a simulação:
            # We define the indicator G(m), which measures more precisely, 
            # for a given threshold m, the percentage error between the 
            # predicted shares of cash payments and the observed shares of cash 
        elif p < self.account_balance_m:
            if payee.accepts_debit and self.prefers_debit:
                self.account_balance_m = self.account_balance_m - p
                self.update_debit_payments_history(p)
                payee.payments = payee.payments + 1
                print("debit")

            elif payee.accepts_pix and self.uses_pix:
                self.account_balance_m = self.account_balance_m - p
                self.update_pix_payments_history(p)
                payee.payments = payee.payments + 1
                print("pix")

            elif payee.accepts_debit:
                self.account_balance_m = self.account_balance_m - p
                self.update_debit_payments_history(p)
                payee.payments = payee.payments + 1
                print("debit2")

            elif payee.accepts_credit:
                self.update_credit_payments_history(p)
                payee.payments = payee.payments + 1
                print("credit")

            else:
                self.lost_payments = self.lost_payments + 1
                payee.lost_payments = payee.lost_payments + 1
                self.lost_payments = self.lost_payments + 1
                print("lost")

        else:
            #pay card
            self.update_credit_payments_history(p)
            payee.payments = payee.payments + 1
            print("credit2")

            
    def update_cash_payments_history(self, p):
        self.cash_payments_history = np.concatenate((self.cash_payments_history, np.array(p)), axis=0)
        if PRINTF:
            print("self.model.cash_payments_history")
            print(self.cash_payments_history)
        
    def update_debit_payments_history(self, p):
        self.debit_payments_history = np.concatenate((self.debit_payments_history, np.array(p)), axis=0)
        if PRINTF:
            print("self.model.debit_payments_history")
            print(self.debit_payments_history)

    def update_pix_payments_history(self, p):
        self.pix_payments_history = np.concatenate((self.pix_payments_history, np.array(p)), axis=0)
        if PRINTF:
            print("self.model.pix_payments_history")
            print(self.pix_payments_history)

    def update_credit_payments_history(self, p):
        self.credit_payments_history = np.concatenate((self.credit_payments_history, np.array(p)), axis=0)
        if PRINTF:
            print("self.model.credit_payments_history")
            print(self.credit_payments_history)

    def withdraw(self):
        '''
        Consumer withdraws money based on distribution
    
        Returns
        -------
        Amount withdrawn
    
        '''
        return random.choice(self.model.distribution_w)
    
    def desinvest(self):
        '''
        Consumer withdraws money based on distribution
    
        Returns
        -------
        Amount withdrawn
    
        '''
        return random.choice(self.model.distribution_w2)

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

def initialize_observed_proportion_by_value():
    diary = pd.read_csv("../../pagtos_varejo_escolha/dados/export/diario_sub.csv", sep = ';', thousands='.', decimal=',')
    max = diary['valor'].max()
    interval_range = pd.interval_range(start=0, freq=2, end=max)

    interval_total = pd.cut(diary['valor'], bins=interval_range) #, right = False)
    dist_total = interval_total.value_counts(normalize=True, sort=False).to_frame()
    return dist_total    

def initialize_observed_proportion_instruments_payments():
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

#    diary_debit = diary.loc[diary['meio'] == "Cartão de débito" or diary['meio'] == "Cartão pré-pago", 'valor']
    diary_debit = diary.loc[diary['meio'] == "Cartão de débito", 'valor']
    if PRINTF:
        print("diary_debit")
        print(diary_debit)
    interval_debit = pd.cut(diary_debit, bins=interval_range) #, right = False)
    dist_debit = interval_debit.value_counts(normalize=False, sort=False).to_frame()

    observed_prop_cash = dist_cash/dist_total
    observed_prop_debit = dist_debit/dist_total
    #print("observed prop")
    #print(observed_prop)
    return observed_prop_cash.fillna(0), observed_prop_debit.fillna(0)


def initialize_withdrawals_distribution(withdrawal_distribution=1):
    label_dist = 'valor' + str(withdrawal_distribution)
    saques = pd.read_csv("../data/saques_abm.csv", sep = ';')
    dist_w = saques[[label_dist]].dropna().to_numpy()
    return dist_w

def initialize_desinvestments_distribution(desinvestment_distribution=1):
    label_dist = 'valor' + str(desinvestment_distribution)
    desinvestimentos = pd.read_csv("../data/desaplicacoes_abm.csv", sep = ';')
    dist_w2 = desinvestimentos[[label_dist]].dropna().to_numpy()
    return dist_w2

def get_evolution_instruments_payments(model):
    """proportion of cash payments"""
    # aqui tem que pegar historico de pagamentos com dinheiro e cartão e 
    # ver a proporção por valor de p
    #max_p_cash = model.cash_payments_history.max(initial = 0)
    #max_p_card = model.card_payments_history.max(initial = 0)
    #max_p = max(max_p_cash, max_p_card)
    #if math.isnan(max_p):
    #    max_p = 0
    
    # max_p = 8000 # valor do diario
    
    # interval_range = pd.interval_range(start=0, freq=2, end=max_p)
    
    # interval_p_cash = pd.cut(pd.Series(model.cash_payments_history), bins=interval_range) #, right = False)
    # interval_p_debit = pd.cut(pd.Series(model.debit_payments_history), bins=interval_range) #, right = False)
    # interval_p_credit = pd.cut(pd.Series(model.credit_payments_history), bins=interval_range) #, right = False)

    # dist_cash = interval_p_cash.value_counts(normalize=False, sort=False).to_frame()
    # dist_debit = interval_p_debit.value_counts(normalize=False, sort=False).to_frame()
    # dist_credit = interval_p_credit.value_counts(normalize=False, sort=False).to_frame()

#     model_cash_history = np.array(0)
#     for obj in model.schedule.agents:
#         if(isinstance(obj, Payer)):
#             if(obj.cash_payments_history.size > 0):
#                 print("objcashhistory")
#                 print(obj.cash_payments_history)
#                 if(model_cash_history.size > 0):
#                     model_cash_history = np.append(model_cash_history, obj.cash_payments_history)
#                 else:
#                     model_cash_history = obj.cash_payments_history
# #    model_cash_history = [np.concatenate((model_cash_history, obj.cash_payments_history), axis=0) for obj in model.schedule.agents if isinstance(obj, Payer) and obj.cash_payments_history and model_cash_history.size]

#     model_debit_history = np.array(0)
#     for obj in model.schedule.agents:
#         if(isinstance(obj, Payer)):
#             if(obj.debit_payments_history.size > 0):
#                 print("objcashhistory")
#                 print(obj.debit_payments_history)
#                 if(model_debit_history.size > 0):
#                     model_debit_history = np.append(model_debit_history, obj.debit_payments_history)
#                 else:
#                     model_debit_history = obj.debit_payments_history


    n_cash = 0
    n_debit = 0
    n_pix = 0
    n_credit = 0
    for obj in model.schedule.agents:
        if(isinstance(obj, Payer)):
            n_cash = n_cash + obj.cash_payments_history.size
            n_debit = n_debit + obj.debit_payments_history.size
            n_pix = n_pix + obj.pix_payments_history.size
            n_credit = n_credit + obj.credit_payments_history.size

    # dist_total = 0    
    # if(model_cash_history.size > 0):
    #     dist_total = model_cash_history.size

    # if(model_debit_history.size > 0):
    #     dist_total = dist_total + model_debit_history.size
        
    #model_debit_history.size #+ model_cash_history + model_cash_history
#    prop_cash = (dist_cash/dist_total).fillna(0)
#    prop_debit = (dist_debit/dist_total).fillna(0)

    if PRINTF:
        print('dist_cash')
#        print(dist_cash)
        print('dist_debit')
#        print(dist_debit)
        print('dist_credit')
        #print(dist_credit)
        print('dist_total')
#        print(dist_total)
        print('prop_cash')
        #print(prop_cash)
    
    return n_cash, n_debit, n_pix, n_credit
    #eturn dist_total

# def get_difference_proportion_instruments(model):
#     # não precisa ler isso toda vez -> inicializa uma vez e tá ok
# #    observed_proportion_cash_payments, observed_proportion_debit_payments = get_observed_proportion_instruments_payments()

#     #print("data collector")
#     #print(type(model.datacollector))
#     proportion_cash_payments, proportion_debit_payments = get_proportion_instruments_payments(model) #['proportion_cash_payments']
 
# #    proportion_cash_payments = model.datacollector.get_model_vars_dataframe() #['proportion_cash_payments']
#     prop_difference_cash = abs(proportion_cash_payments.iloc[:, 0] - model.observed_prop_cash['valor'])
#     prop_difference_debit = abs(proportion_debit_payments.iloc[:, 0] - model.observed_prop_debit['valor'])
    
#     # não precisa ler sempre; basta uma vez e tá ok
# #    weights = get_observed_proportion_by_value()

#     prop_difference_weighted = (prop_difference_cash + prop_difference_debit) * model.weigths['valor']

#     total_difference = prop_difference_weighted.sum()

#     if PRINTF:
#         print("prop cash payments")
#         print(proportion_cash_payments)
#         print("prop difference_cash")
#         print(prop_difference_cash)
#         print("weights")
#         print(model.weigths)
        
#     return total_difference

class CashManagementModel3(Model):
    """A model with some number of agents."""
    def __init__(self, 
                 num_payers,
                 num_payees,
                 initial_cash_balance_m,
                 initial_account_balance_m,
                 threshold_cash_balance_mth,
                 threshold_account_balance_mth,
#                 purchases_distribution,
#                 observed_proportion_purchases_by_value,
                 number_withdrawal_distribution,
                 number_desinvestment_distribution,
#                 observed_prop_cash, 
#                 observed_prop_debit,
                 ratio_threshold_for_accepting,
                 ratio_threshold_for_using,
                 ):
        
        self.running = True
        self.num_payers = num_payers
        self.num_payees = num_payees
        self.threshold_cash_balance_mth = threshold_cash_balance_mth
        self.threshold_account_balance_mth = threshold_account_balance_mth
        self.ratio_threshold_for_accepting = ratio_threshold_for_accepting
        self.ratio_threshold_for_using = ratio_threshold_for_using
#        self.cash_payments_history = np.empty(0)
#        self.debit_payments_history = np.empty(0)
#        self.credit_payments_history = np.empty(0)

        #self.schedule = RandomActivation(self)
        #self.schedule = BaseScheduler(self)
        self.schedule = StagedActivation(self, ["period1", "period2"])
        
        unique_id = 0

        self.distribution_p = initialize_purchases_distribution()
#        self.weigths = observed_proportion_purchases_by_value
        self.distribution_w = initialize_withdrawals_distribution(number_withdrawal_distribution)
        self.distribution_w2 = initialize_desinvestments_distribution(number_desinvestment_distribution)
        #print(self.distribution_w)
        #print("w:")
        #print(self.distribution_w[15])
        #print(random.choice(self.distribution_w))
        #self.distribution_p = initialize_purchases_distribution()
        #self.weigths = initialize_observed_proportion_by_value()
        #self.observed_prop_cash, self.observed_prop_debit = initialize_observed_proportion_instruments_payments()
#        self.observed_prop_cash = observed_prop_cash
#        self.observed_prop_debit = observed_prop_debit
        #print(self.distribution_p)

        # At t = 0, the representative agent is initialized with zero cash balances.
        
        # Create consumers
        for i in range(self.num_payers):
            
            payer = Payer(unique_id, 
                          self, 
                          initial_cash_balance_m, 
                          initial_account_balance_m,
                          uses_pix=False,
                          prefers_debit=True)
            unique_id += 1
            self.schedule.add(payer)
            
                # Create merchants
        for i in range(self.num_payees):
            accepts_pix = np.random.binomial(1, .05, 1) # n, p, number of trials => 75% prefer CASH (0)
            accepts_debit = np.random.binomial(1, .70, 1) # fazer a conta levando em contas as pessoas tb
            #accepts_credit = np.random.binomial(1, .65, 1) # n, p, number of trials => 75% prefer CASH (0)
            accepts_credit = accepts_debit
            
            payee = Payee(unique_id, 
                          self, 
                          accepts_pix,
                          accepts_debit,
                          accepts_credit)
            unique_id += 1
            self.schedule.add(payee)
            
        self.datacollector = DataCollector(
            model_reporters={"evolution_instruments_payments": get_evolution_instruments_payments},
#                             "difference_proportion_instruments_payments": get_difference_proportion_instruments},
            agent_reporters={})
    
        self.datacollector.collect(self)

    def step(self):
        '''Advance the model by one step.'''
        self.schedule.step()
        self.datacollector.collect(self)
        