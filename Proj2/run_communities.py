'''
Created on Dec 12, 2017

@author: martin
'''

from Search import Search
from Test import Test

from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor

import Communities

def main():
    data, features= Communities.load_data()
    
    data= Communities.clean(data, True)
    
    data= data.where(data != "?", None)
    
    print(data.shape)
    # Now just to do smthg
    data= data.dropna()
    
    print(data.shape)
    y= data['ViolentCrimesPerPop']
    del(data['ViolentCrimesPerPop'])
    X= data
    # print(y)
    
    # parameters= [{'n_neighbors' : range(4, 8), 'weights':['uniform', 'distance'], 'p':[1, 2]}]
    parameters= {'n_neighbors' : range(4, 8)}
    h= Search(X, y, KNeighborsRegressor(), parameters)
    h.run()
    
    h= Test(X, y, DecisionTreeRegressor())
    h.run()
    h= Test(X, y, KNeighborsRegressor())
    h.run()
    # h.labelFun(labelFunD)
    # h.report(fn="../Report/results/congress.knn.cm.tex")
    
    print(data.shape)
    
if __name__ == '__main__':
    main()
