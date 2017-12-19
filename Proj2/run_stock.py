'''
Created on Dec 12, 2017

@author: martin
'''

from Search import Search
from Test import Test

import time

from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVR
from sklearn import preprocessing

from numpy import arange

import pandas as pd
import numpy as np

import Stock

import Stuff

def run_knn(X, y, results,t):
    parameters= {'n_neighbors' : range(1,30), 'weights':['uniform', 'distance'],  'p':[1, 2]}
    # parameters= {'n_neighbors' : range(4, 8), 'weights':['uniform', 'distance'],  'p':[1, 2]}
    h= Search(X, y, KNeighborsRegressor(), parameters, verbose=False)
    h.run()
    print(h.getBest())
    results['KNN '+t]= h.getBestResult()
    h.report("../Slides2/tables/stock.knn.tex")
    # h.report()
    # print(h.reportTable())
    
def run_tree(X, y, results,t):
    parameters= {'criterion' : ['mse', 'friedman_mse', 'mae'], 'splitter' : ['best', 'random'], 'max_features': ['auto', 'sqrt', 'log2']}
    h= Search(X, y, DecisionTreeRegressor(), parameters, verbose=False)
    h.run()
    print(h.getBest())
    results['Decision Tree '+t]= h.getBestResult()
    h.report("../Slides2/tables/stock.dt.tex")
   
def run_bayes(X, y, results,t):
    yt= y.apply(lambda x: x*100).astype('int64')
    h= Search(X, yt, GaussianNB(), {}, verbose=False)
    h.run()
    print(h.getBest())
    results['GaussianNB '+t]= h.getBestResult()
    
def run_svm(X, y, results,t):
    C_range = 10. ** arange(-3, 8)
    gamma_range = 10. ** arange(-5, 4)
    parameters={'kernel' : ['linear','sigmoid', 'rbf', 'poly'], 'C' : C_range, 'gamma':gamma_range}
    # parameters={'kernel' : ['rbf'], 'C' : [50,90,99,100,101], 'gamma':gamma_range}
    h= Search(X, y, SVR(), parameters, verbose=False)
    h.run()
    print(h.getBest())
    results['SVR '+t]= h.getBestResult()
    h.report("../Slides2/tables/stock.svm.tex")

def main():
    data= Stock.load_data()
    
    data= Stock.prepare(data)
    
    print(data.shape)
    
    # Stuff.plot_corr(data, "stock_final.png")
    
    X= data[["Large B/P", "Large ROE", "Large S/P", "Large Return Rate in the last quarter", "Large Market Value", "Small systematic Risk"]]
    targets= ["Annual Return", "Excess Return", "Systematic Risk", "Total Risk", "Abs. Win Rate", "Rel. Win Rate"]
    targets= ["Annual Return"]
    col= "Annual Return"
    y= data[col]
    
    results= {}
    
    for col in targets:
        print(col)
        y= data[col]
        t1= time.time()
        run_knn(X, y, results, col)
        run_tree(X, y, results, col)
        run_bayes(X, y, results, col)
        run_svm(X, y, results, col)
        t2= time.time()
    
    print("Elapsed", (t2-t1))
    
    Stuff.writeResultsTable(results, "../Slides2/tables/stock.tex")
    
if __name__ == '__main__':
    main()
