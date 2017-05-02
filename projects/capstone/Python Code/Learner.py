# -*- coding: utf-8 -*-
"""
Created on Tue Apr 04 19:02:20 2017

@author: Carl
"""

from sklearn.model_selection import TimeSeriesSplit, cross_val_score, train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

#regressors
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LassoLars, BayesianRidge
from sklearn.svm import SVR
from sklearn.linear_model import SGDRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.neural_network import MLPRegressor

#classifiers
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from matplotlib import pyplot as plt
import seaborn as sns

class Learner:
    
    def __init__(self, stocks):
        
        self.stocks = stocks;
        
    def prepare_data(self, ticker, features, target, test_size, scale_data):
        
        data = self.stocks.data[ticker].copy()
        
        data['target'] = data[target].shift(-1)
        data = data[features + ['target']]
        data = data.dropna()
        
        X = data[features]
        y = data['target']
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        if scale_data:
            scaler = StandardScaler()
            
            self.X_train = scaler.fit_transform(self.X_train)
            self.X_test = scaler.transform(self.X_test)
        
    def train_regressors(self):
        
        names = ["Linear Regression",
                 "Ridge",
                 "Lasso",
                 "LARS Lasso",
                 "Bayesian Ridge",
                 "SVR",
                 "SGD",
                 "Nearest Neighbours",
                 "Decision Tree",
                 "Random Forest",
                 "Adaboost",
                 "Neural Net"]
        
        classifiers = [
            LinearRegression(),
            Ridge(alpha = 0.5),
            Lasso(alpha = 0.1),
            LassoLars(alpha = 0.1),
            BayesianRidge(),
            SVR(),
            SGDRegressor(),
            KNeighborsRegressor(),
            DecisionTreeRegressor(),
            RandomForestRegressor(),
            AdaBoostRegressor(),
            MLPRegressor()]
        
        for name, clf in zip(names, classifiers):

            clf.fit(self.X_train, self.y_train)
            score = clf.score(self.X_test, self.y_test)
            print(name + ': ' + str(score))
        
    def train_classifiers(self):
               
        names = ["Nearest Neighbors", 
                #"Linear SVM", 
                "RBF SVM", 
                "Gaussian Process", 
                "Decision Tree", 
                "Random Forest", 
                "Neural Net", 
                "AdaBoost",
                "Naive Bayes", 
                "QDA"]
        
        classifiers = [
            KNeighborsClassifier(3),
            #SVC(kernel="linear", C=0.025),
            SVC(gamma=2, C=1),
            GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True),
            DecisionTreeClassifier(max_depth=5),
            RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
            MLPClassifier(alpha=1),
            AdaBoostClassifier(),
            GaussianNB(),
            QuadraticDiscriminantAnalysis()]
        
        for name, clf in zip(names, classifiers):

            clf.fit(self.X_train, self.y_train)
            score = clf.score(self.X_test, self.y_test)
            print(name + ': ' + str(score))
    
    def plot_correlation_heatmap(self, ticker):
        sns.set(context='paper', font='monospace')
        corrmat = self.stocks.data[ticker].corr()
        f, ax = plt.subplots(figsize=(12,9))
        sns.heatmap(corrmat, square=True)
        f.tight_layout()
        plt.yticks(rotation=0)
        plt.xticks(rotation=90)