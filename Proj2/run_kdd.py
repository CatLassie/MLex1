'''
Created on Dec 12, 2017

@author: martin
'''

from Search import Search
from Test import Test

import time

from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor

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
    
def predict(X, y):
    o,f= KDD.load_data("test")
    idx= o['CONTROLN']
    del(o['CONTROLN'])
    o= KDD.missing_vals(KDD.clean(KDD.xxx(o)))
    #h= Test(X, y, SVC(C=1, kernel='sigmoid', gamma=0.001))
    #h.run()
    #o['Class']= h.predict(o[features])
    #o['Class']= o['Class'].apply(lambda x: labels[x])
    h= Test(X, y, KNeighborsRegressor(n_neighbors=290, weights='uniform', p=2))
    o['TARGET_D']= h.predict(o)
    o['CONTROLN']= idx
    o[list(['CONTROLN', 'TARGET_D'])].to_csv("out.kdd.csv", columns=['CONTROLN','TARGET_D'],index=False)

def main():
    data, features= KDD.load_data()
    
    h= data['CONTROLN']
    del(data['CONTROLN'])
    y= data['TARGET_D']
    del(data['TARGET_D'])
    
    data= KDD.clean(KDD.xxx(data))
    
    # of the cleaned cols about 20 (22) have missing values
    data= KDD.missing_vals(data)
    
    print(data.shape)
    
    X= data
    
    t1= time.time()
    # run_knn(X, y)
    # run_tree(X, y)
    t2= time.time()
    
    predict(X, y)
    
    print("Elapsed", (t2-t1))
    
if __name__ == '__main__':
    main()
