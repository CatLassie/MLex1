'''
Created on Dec 12, 2017

@author: martin
'''

from Search import Search
from Test import Test

import time

from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor

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
    
def predict(X, y):
    ind,_= Auto.load_data("test")
    ind= Auto.missing_vals(Auto.clean(ind))
    idx= ind['id']
    del(ind['id'])
    h= Test(X, y, KNeighborsRegressor(n_neighbors=29, p=2, weights='distance'))
    ind['mpg']= h.predict(ind)
    ind['id']= idx
    ind[list(['id', 'mpg'])].to_csv("out.auto.csv", columns=['id','mpg'],index=False)

def main():
    data, features= Auto.load_data()
    
    h= data['id']
    del(data['id'])
    y= data['mpg']
    del(data['mpg'])
    
    data= Auto.clean(data)
    
    data= Auto.missing_vals(data)
    
    print(data.shape)
    
    X= data
    
    t1= time.time()
    run_knn(X, y)
    # run_tree(X, y)
    t2= time.time()
    
    print("Elapsed", (t2-t1))
    
    predict(X,y)
    
if __name__ == '__main__':
    main()
