# -*- coding: utf-8 -*-
"""
Created on Sat Apr 01 13:20:39 2017

@author: Carl
"""
from Stocks import Stocks

from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

import datetime
from matplotlib import pyplot as plt

def run_ml(stocks, ticker, features, target, train_ratio, scale_flag):
    
    data = stocks.data[ticker].copy()
    
    data['target'] = data[target].shift(-1)
    data = data[features + ['target']]
    data = data.dropna()
    
    X = data[features]
    y = data['target']
    
    num_samples = X.shape[0]
    break_sample = int(train_ratio*num_samples)
    
    X_train = X[:break_sample]
    y_train = y[:break_sample]
    X_test = X[break_sample:]
    y_test = y[break_sample:]
    
    if scale_flag:
        scaler = StandardScaler()
        
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    
    linear = linear_model.LinearRegression()
    
    linear.fit(X_train, y_train)
    
    y_pred = linear.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    plt.figure()
    plt.plot(y_test.values)
    plt.plot(y_pred)
    plt.legend(['test', 'pred'])
    plt.title(ticker + ' ' + target + ' mse:' + str(mse) + ' r2:' + str(r2))
    plt.show()
    
    return linear
    
start = datetime.datetime(2010,1,1)
end = datetime.datetime(2017,3,31)
    
stocks = Stocks(start, end, ['GOOG', 'AAPL', 'SPY']) 

ticker = stocks.get_tickers()[0]
features = [feature for feature in stocks.get_features() if feature not in ['Open', 'High', 'Low', 'Close']]
target = 'Daily Return'
train_ratio = 0.9

run_ml(stocks, ticker, features, target, train_ratio, False)

ticker = stocks.get_tickers()[0]
features = [feature for feature in stocks.get_features() if feature not in ['Open', 'High', 'Low', 'Close']]
target = 'Adj Close'
train_ratio = 0.9

run_ml(stocks, ticker, features, target, train_ratio, False)

stocks.plot('GOOG', ['Adj Close', 'SMA21', 'SMA252', 'EWMA21', 'EWMA252'])
stocks.plot('GOOG', ['SMA21 Delta', 'SMA252 Delta', 'EWMA21 Delta', 'EWMA252 Delta'])