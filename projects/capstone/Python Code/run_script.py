# -*- coding: utf-8 -*-
"""
Created on Sat Apr 01 13:20:39 2017

@author: Carl
"""
from Stocks import Stocks
from Learner import Learner
from Market import Market
from Twitter import Twitter

import datetime
   
start = datetime.date(2010,1,1)
end = datetime.date(2017,3,31)
    
stocks = Stocks(start, end, ['GOOG', 'AAPL', 'SPY']) 
learner = Learner(stocks)
market = Market(stocks, 1000000, 1)
twitter = Twitter('./PickledTweets')

ticker = stocks.get_tickers()[0]
features = [feature for feature in stocks.get_features() if feature not in ['Open', 'High', 'Low', 'Close']]
target = 'Daily Log Return Direction'
test_size = 0.1
scale_data = True

learner.prepare_data(ticker, features, target, test_size, scale_data)
print("---Classifiers---")
learner.train_classifiers()

target = 'Daily Log Return'

learner.prepare_data(ticker, features, target, test_size, scale_data)
print("---Regressors---")
learner.train_regressors()

learner.plot_correlation_heatmap(ticker)

start = datetime.date(2011,1,1)
end = datetime.date(2016,3,31)

market.purchase_stock('GOOG', start)
market.sell_stock('GOOG', end)

print(market.calc_return_on_investment('GOOG', start, end))
print(market.bank/1000000)

start_date = datetime.date(2017,4,3)
end_date = datetime.date(2017,4,4)

polarity = twitter.get_all_tweets('ibm', start_date, end_date, 5, 'en')

#TODO train on multiple stocks
#TODO add features
#TODO add twitter
#DONE try different learners
#DONE try classifier instead of regressor
#TODO use resulting model to guide buys and sells and copare against buy and hold SPY
#TODO save data and then use the saved data if available
#TODO PCA to see which features explain the data best
#TODO parameter search on classifier
#DONE heatmap of correlation between features