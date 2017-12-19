'''
Created on Dec 12, 2017

@author: martin
'''

from Search import Search
from Test import Test
import Stuff

import time

import pandas as pd

from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn import preprocessing

import KDD

def run_knn(X, y):
    parameters= {'n_neighbors' : range(1,300,10), 'weights':['uniform', 'distance'],  'p':[1, 2]}
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
    
def predict(X, y, scaler, variant=1):
    o, _= KDD.load_data("test")
    idx= o['CONTROLN']
    del(o['CONTROLN'])
    if variant == 1:
        o= scaler.transform(KDD.remove_missing_vals(KDD.transform(KDD.clean(o))))
    else:
        o= scaler.transform(KDD.impute_mean(KDD.transform(KDD.clean(o))))
    #h= Test(X, y, SVC(C=1, kernel='sigmoid', gamma=0.001))
    #h.run()
    #o['Class']= h.predict(o[features])
    #o['Class']= o['Class'].apply(lambda x: labels[x])
    h= Test(X, y, KNeighborsRegressor(n_neighbors=290, weights='uniform', p=2))
    pred= h.predict(o)
    o= pd.DataFrame()
    o['TARGET_D']= pred
    o['CONTROLN']= idx
    o[list(['CONTROLN', 'TARGET_D'])].to_csv("out.kdd.csv", columns=['CONTROLN','TARGET_D'],index=False)

def main():
    data, _= KDD.load_data()
    
    del(data['CONTROLN'])
    y= data['TARGET_D']
    del(data['TARGET_D'])
    
    Stuff.plot_corr(data, "kdd.orig.png")
    
    data= KDD.transform(KDD.clean(data))
    
    # of the cleaned cols about 20 (22) have missing values
    dataa= KDD.remove_missing_vals(data)
    
    Stuff.plot_corr(dataa, "kdd.clean.png")
    
    scaler= preprocessing.MinMaxScaler()
    scaler.fit(dataa)
    dataa= scaler.transform(dataa)
    # data= preprocessing.scale(data)
    
    print(dataa.shape)
    
    X= dataa
    
    t1= time.time()
    run_knn(X, y)
    run_tree(X, y)
    t2= time.time()
    
    predict(X, y, scaler)
    
    print("Elapsed", (t2-t1))
    
    datab= KDD.impute_mean(data)
    
    Stuff.plot_corr(datab, "kdd.clean.png")
    
    scaler= preprocessing.MinMaxScaler()
    scaler.fit(datab)
    datab= scaler.transform(datab)
    # data= preprocessing.scale(data)
    
    print(datab.shape)
    
    X= datab
    
    t1= time.time()
    run_knn(X, y)
    run_tree(X, y)
    t2= time.time()
    
    # predict(X, y, scaler)
    
    print("Elapsed", (t2-t1))
    
if __name__ == '__main__':
    main()
