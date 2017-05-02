# -*- coding: utf-8 -*-
"""
Created on Sat Apr 01 12:58:17 2017

@author: Carl
"""
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import numpy as np

class Stocks:
    
    def __init__(self, start_time, end_time, tickers=[]):
        
        self.start_time = start_time
        self.end_time = end_time
        
        self.data={}
        
        for ticker in tickers:
            self.add_stock(ticker)
            
    def add_stock(self, ticker):
        print('Downloading ' + ticker + ' Data')
        self.data[ticker] = web.DataReader(ticker, 'yahoo', self.start_time, self.end_time)
        self.add_features(ticker)
        
    def remove_stock(self, ticker):
        del self.data[ticker]
        
    def get_stock_data(self, ticker):
        return self.data[ticker]
    
    def get_tickers(self):
        return self.data.keys()
    
    def get_features(self):
        tickers = self.get_tickers()
        if len(tickers) != 0:
            return list(self.data[tickers[0]].columns.values)
        else:
            return []
            
    def add_features(self, ticker):
        self.calc_daily_return(ticker)
        self.calc_daily_return_direction(ticker)
        self.calc_daily_log_return(ticker)
        self.calc_daily_log_return_direction(ticker)
        days = [5, 21, 252]
        for day in days:
            self.calc_momentum(ticker, day)
            self.calc_simple_moving_average(ticker, day)
            self.calc_exponentially_weighted_moving_average(ticker, day)
            self.calc_sma_delta(ticker, day)
            self.calc_ewma_delta(ticker, day)
        
    def calc_daily_return(self, ticker):
        self.data[ticker]['Daily Return'] = self.data[ticker]['Adj Close'].pct_change(1)
            
    def calc_momentum(self, ticker, days):
        self.data[ticker]['Momentum' + str(days)] = self.data[ticker]['Adj Close'].pct_change(days)
        
    def calc_simple_moving_average(self, ticker, days):
        self.data[ticker]['SMA' + str(days)] = self.data[ticker]['Adj Close'].rolling(days).mean()
        
    def calc_exponentially_weighted_moving_average(self, ticker, days):
        self.data[ticker]['EWMA' + str(days)] = self.data[ticker]['Adj Close'].ewm(span=days).mean()
        
    def calc_sma_delta(self, ticker, days):
        self.calc_simple_moving_average(ticker, days)
        self.data[ticker]['SMA' + str(days) + ' Delta'] = self.data[ticker]['Adj Close'] - self.data[ticker]['SMA' + str(days)]
        
    def calc_ewma_delta(self, ticker, days):
        self.calc_exponentially_weighted_moving_average(ticker, days)
        self.data[ticker]['EWMA' + str(days) + ' Delta'] = self.data[ticker]['Adj Close'] - self.data[ticker]['EWMA' + str(days)]
        
    def calc_daily_return_direction(self, ticker):
        self.calc_daily_return(ticker)
        self.data[ticker]['Daily Return Direction'] = np.sign(self.data[ticker]['Daily Return'])
        
    def calc_daily_log_return(self, ticker):
        self.data[ticker]['Daily Log Return'] = np.log(self.data[ticker]['Adj Close']) - np.log(self.data[ticker]['Adj Close'].shift(1))
        
    def calc_daily_log_return_direction(self, ticker):
        self.calc_daily_log_return(ticker)
        self.data[ticker]['Daily Log Return Direction'] = np.sign(self.data[ticker]['Daily Log Return'])
        
    def plot(self, ticker, keys):
        plt.figure()
        for key in keys:
            plt.plot(self.data[ticker][key])
        plt.title(ticker)
        plt.xlabel('Date')
        plt.legend()
        plt.show()
            
if __name__ == "__main__":
    
    import datetime
    
    start = datetime.date(2010,1,1)
    end = datetime.date(2017,3,31)
    
    stocks = Stocks(start, end, ['GOOG', 'AAPL', 'SPY']) 
    print(stocks.get_tickers())
    print(stocks.get_features())
    stocks.add_stock('YHOO')
    print(stocks.get_tickers())
    stocks.remove_stock('YHOO')
    print(stocks.get_tickers())
    
    stocks2 = Stocks(start, end)
    print(stocks2.get_tickers())
    print(stocks2.get_features())
    stocks2.add_stock('YHOO')
    print(stocks2.get_tickers())
    print(stocks2.get_features())
    stocks2.remove_stock('YHOO')
    print(stocks2.get_tickers())
    
    stocks.plot('GOOG', 'Adj Close')