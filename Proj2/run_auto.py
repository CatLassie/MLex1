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
from sklearn import preprocessing

import pandas as pd

import Auto

def run_knn(X, y):
    parameters= {'n_neighbors' : range(1,30), 'weights':['uniform', 'distance'],  'p':[1, 2]}
    # parameters= {'n_neighbors' : range(4, 8), 'weights':['uniform', 'distance'],  'p':[1, 2]}
    h= Search(X, y, KNeighborsRegressor(), parameters, verbose=False)
    h.run()
    print(h.getBest())
    # h.report()
    # print(h.reportTable())
    
def run_tree(X, y):
    parameters= {'criterion' : ['mse', 'friedman_mse', 'mae'], 'splitter' : ['best', 'random'], 'max_features': ['auto', 'sqrt', 'log2']}
    h= Search(X, y, DecisionTreeRegressor(), parameters, verbose=False)
    h.run()
    print(h.getBest())
    
def run_bayes(X, y):
    parameters= {}
    h= Search(X, y, GaussianNB(), parameters, verbose=False)
    h.run()
    print(h.getBest())
    
def predict(X, y, scaler):
    ind,_= Auto.load_data("test")
    ind= Auto.missing_vals(Auto.clean(ind))
    idx= ind['id']
    del(ind['id'])
    o= scaler.transform(ind)
    h= Test(X, y, KNeighborsRegressor(n_neighbors=29, p=2, weights='distance'))
    pred= h.predict(o)
    o= pd.DataFrame()
    o['mpg']= pred
    o['id']= idx
    o[list(['id', 'mpg'])].to_csv("out.auto.csv", columns=['id','mpg'],index=False)

def main():
    data, _= Auto.load_data()
    
    del(data['id'])
    y= data['mpg']
    del(data['mpg'])
    
    data= Auto.clean(data)
    
    data= Auto.missing_vals(data)
    
    print(data.shape)
    
    scaler= preprocessing.MinMaxScaler()
    scaler.fit(data)
    data= scaler.transform(data)
    
    X= data
    
    t1= time.time()
    # run_knn(X, y)
    # run_tree(X, y)
    run_bayes(X, y)
    t2= time.time()
    
    print("Elapsed", (t2-t1))
    
    predict(X,y, scaler)
    
if __name__ == '__main__':
    main()
