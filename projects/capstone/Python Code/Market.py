# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 19:25:51 2017

@author: Carl
"""

import math

class Market:
    
    def __init__(self, stocks, starting_bank, max_purchase):
        
        self.stocks = stocks
        self.bank = starting_bank
        self.max_purchase = max_purchase
        self.portfolio = {}
        
    def calc_return_on_investment(self, ticker, start, end):
    
        start_value = self.stocks.data[ticker]['Adj Close'].truncate(start)[0]
        end_value = self.stocks.data[ticker]['Adj Close'].truncate(end)[0]
        
        return (end_value - start_value) / start_value
    
    def classifier_return(self, clf, start, end):
        
        #find if clf predicts up or down
        pred = clf.predict(X)
        
        if pred > 0:
            self.purchase_stock(ticker, date)
            
        elif pred < 0:
            self.sell_stock(ticker, date)
        
    def purchase_stock(self, ticker, date):
        if self.bank > 0 and ticker not in self.portfolio:
            value = self.stocks.data[ticker]['Adj Close'].truncate(date)[0]
            cash = self.bank * self.max_purchase
            number = math.floor(cash/value)
            self.bank -= number * value
            self.portfolio[ticker] = number
                
    def sell_stock(self, ticker, date):
        if ticker in self.portfolio:
            number = self.portfolio[ticker]
            value = self.stocks.data[ticker]['Adj Close'].truncate(date)[0]
            price = number * value
            self.bank += price
            del self.portfolio[ticker]