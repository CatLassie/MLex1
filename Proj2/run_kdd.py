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
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVR
from sklearn import preprocessing

from numpy import arange

import KDD

def run_knn(X, y, results,t):
    parameters= {'n_neighbors' : range(1,300,10), 'weights':['uniform', 'distance'],  'p':[1, 2]}
    # parameters= {'n_neighbors' : range(4, 8), 'weights':['uniform', 'distance'],  'p':[1, 2]}
    h= Search(X, y, KNeighborsRegressor(), parameters, verbose=False)
    h.run()
    print(h.getBest())
    results['KNN '+t]= h.getBestResult()
    h.report("../Slides2/tables/kdd.knn.tex")
    # print(h.reportTable())
    
def run_tree(X, y, results, t):
    parameters= {'criterion' : ['mse', 'friedman_mse', 'mae'], 'splitter' : ['best', 'random'], 'max_features': ['auto', 'sqrt', 'log2']}
    h= Search(X, y, DecisionTreeRegressor(), parameters, verbose=False)
    h.run()
    print(h.getBest())
    results['Decision Tree '+t]= h.getBestResult()
    h.report("../Slides2/tables/kdd.dt.tex")
    
def run_bayes(X, y, results, t):
    parameters= {}
    h= Search(X, y, GaussianNB(), parameters, verbose=False)
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
    h.report("../Slides2/tables/kdd.svm.tex")
    
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
    
    Stuff.plot_corr(dataa, "kdd_final.png")
    
    scaler= preprocessing.MinMaxScaler()
    scaler.fit(dataa)
    dataa= scaler.transform(dataa)
    # data= preprocessing.scale(data)
    
    print(dataa.shape)
    
    X= dataa
    
    results={}
    
    t1= time.time()
    run_knn(X, y,results, 'RM')
    run_tree(X, y,results, 'RM')
    run_bayes(X, y,results, 'RM')
    run_svm(X, y,results, 'RM')
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
    run_knn(X, y,results,   'IM')
    run_tree(X, y,results,  'IM')
    run_bayes(X, y,results, 'IM')
    run_svm(X, y,results,   'IM')
    t2= time.time()
    
    # predict(X, y, scaler)
    
    print("Elapsed", (t2-t1))
    
    Stuff.writeResultsTable(results, "../Slides2/tables/kdd.tex")
    
if __name__ == '__main__':
    main()
