'''
Created on Dec 12, 2017

@author: martin
'''

from Search import Search

import time

from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor

import Communities

def run_knn(X, y):
    parameters= {'n_neighbors' : range(1, 80), 'weights':['uniform', 'distance'],  'p':[1, 2]}
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

def main():
    data, _= Communities.load_data()
    
    data= Communities.clean(data, False)
    
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
    
    t1= time.time()
    run_knn(X, y)
    run_tree(X, y)
    t2= time.time()
    
    print("Elapsed", (t2-t1))
    
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
