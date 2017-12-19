'''
Created on Dec 12, 2017

@author: martin
'''

from Search import Search

import Stuff

import time

from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVR

from numpy import arange

import Communities

def run_knn(X, y, results):
    parameters= {'n_neighbors' : range(1, 80), 'weights':['uniform', 'distance'],  'p':[1, 2]}
    # parameters= {'n_neighbors' : range(39, 40), 'weights':['uniform', 'distance'],  'p':[1, 2]}
    # parameters= {'n_neighbors' : range(4, 8), 'weights':['uniform', 'distance'],  'p':[1, 2]}
    h= Search(X, y, KNeighborsRegressor(), parameters, verbose=False)
    h.run()
    print(h.getBest())
    results['KNN']= h.getBestResult()
    # h.report()
    # print(h.reportTable())
    
def run_tree(X, y, results):
    parameters= {'criterion' : ['mse', 'friedman_mse', 'mae'], 'splitter' : ['best', 'random'], 'max_features': ['auto', 'sqrt', 'log2']}
    h= Search(X, y, DecisionTreeRegressor(), parameters, verbose=False)
    h.run()
    print(h.getBest())
    results['Decision Tree']= h.getBestResult()
    
def run_bayes(X, y, results):
    yt= y.apply(lambda x: x*100).astype('int64')
    h= Search(X, yt, GaussianNB(), {}, verbose=False)
    h.run()
    print(h.getBest())
    results['GaussianNB']= h.getBestResult()
    
def run_svm(X, y, results):
    C_range = 10. ** arange(-3, 8)
    gamma_range = 10. ** arange(-5, 4)
    parameters={'kernel' : ['linear','sigmoid', 'rbf', 'poly'], 'C' : C_range, 'gamma':gamma_range}
    # parameters={'kernel' : ['rbf'], 'C' : [50,90,99,100,101], 'gamma':gamma_range}
    h= Search(X, y, SVR(), parameters, verbose=False)
    h.run()
    print(h.getBest())
    results['SVR']= h.getBestResult()

def main():
    data, _= Communities.load_data()
    
    data= Communities.clean(data, True)
    
    # of the cleaned cols about 20 (22) have missing values
    data= Communities.missing_vals(data)
    
    print(data.shape)
    # Now just to do smthg
    # data= data.dropna()
    
    print(data.shape)
    y= data['ViolentCrimesPerPop']
    del(data['ViolentCrimesPerPop'])
    X= data
    # print(y)
    
    results= {}
    
    t1= time.time()
    run_knn(X, y, results)
    run_tree(X, y, results)
    run_bayes(X, y, results)
    run_svm(X, y, results)
    t2= time.time()
    
    print("Elapsed", (t2-t1))
    
    Stuff.writeResultsTable(results, "../Slides2/tables/communities.tex")
    
    # h.report()
    # print(h.reportTable())
    
#     h= Test(X, y, DecisionTreeRegressor())
#     h.run()
#     h= Test(X, y, KNeighborsRegressor())
#     h.run()
    # h.labelFun(labelFunD)
    # h.report(fn="../Report/results/congress.knn.cm.tex")
    
if __name__ == '__main__':
    main()
