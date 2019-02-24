# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 13:35:16 2018

@author: deban.daniel
"""
import random
import numpy
from mesa import Agent, Model
from mesa.time import RandomActivation

class Merchant(Agent):
    """An agent with fixed initial wealth."""
    def __init__(self, unique_id, produtos, preco, model):
        super().__init__(unique_id, model)
        self.recursos = 0
        self.produtos = produtos
        self.preco = preco
        
class Portador(Agent):
    """An agent with fixed initial wealth."""
    def __init__(self, unique_id, recursos, model):
        super().__init__(unique_id, model)
        self.recursos = recursos
        
    def step(self):
        if self.recursos == 0:
            return
        # chooses a merchant 
        lojistas = [obj for obj in self.model.schedule.agents if isinstance(obj, Merchant)]
        lojista = random.choice(lojistas)
        print('lojista ' + str(lojista.unique_id))
        
        buy = numpy.random.binomial(1, .5, 1) # n, p, number of trials 
        if buy:
            print('buy')
            if lojista.preco <= self.recursos and lojista.produtos > 0:
                lojista.recursos += lojista.preco
                lojista.produtos -= 1 # pode não ser necessario; mlehor assumir produtos ilimitados
                self.recursos -= lojista.preco
        else:
            print('dont buy')
        """idade
        "renda"
        "escolaridade"
        "sexo"
        "dinheiro no bolso
        "limite no cartão
        "saldo em conta
        "instrumentos = debito, credito, dinheiro, mas dependendo da renda, escolaridade, sexo, etc."
    "definir funcao compra
    "definir funcao escolhe instrumento
        
" com base em idade, renda, escolaridade e sexo, sorteio quais instrumentos tem
" aí vou sorteando uma compra por passo, assumindo que as compras sempre são feitas; ou pode ser número de compras por dia tb
" cada compra é sorteada com um valor
" aí tem que ter uma função escolha do instrumento, com base no tipo do bem, 
" instrumento tem que ter conveniencia (dinheiro é o mais conveniente ou tem nota para conveniencia x); aí a função tem que escolher o instrumento com base no valor da compra e no tipo de coisa que está sendo comprado e na loja
" para compras pequenas, o peso maior é na conveniencia; para compras grandes, na segurança e benefícios
" se eu coloco um instrumento mais conveniente que dinheiro, como fica a escolha?
" o custo tem que ser a média da anuidade menos benefícios; valores absolutos? relativos por compra?
" escolhe se vai entrar no limite do cheque especial, se tem limite do cartão para fazer a compra, etc.
" pode ser necessário definir classe instrumento, mas é melhor deixar como parametro
"""
        
class RetailPaymentsModel(Model):
    """A model with some number of agents."""
    def __init__(self, M, N):
        self.num_portadores = M
        self.num_lojistas = N
        self.schedule = RandomActivation(self)
        # Create clients
        for i in range(self.num_portadores):
            renda = 10
            p = Portador(i, renda, self)
            self.schedule.add(p)
            
        # Create merchants
        for i in range(self.num_lojistas):
            produtos = 20
            preco = 2
            l = Merchant(i, produtos, preco, self)
            self.schedule.add(l)

    def step(self):
        '''Advance the model by one step.'''
        print('step')
        self.schedule.step()
        