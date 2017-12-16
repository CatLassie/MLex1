'''
Created on Nov 22, 2017

@author: martin
'''

from __future__ import print_function
try:
    from StringIO import StringIO as StringIO
except ImportError:
    from io import StringIO
import cProfile
import pstats
from copy import copy
# from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_predict
from sklearn import metrics
import time
import numpy as np

def split(X,y, n=0.8):
    return train_test_split(X, y, random_state=1234, test_size=n)

class Result:
    t_fit= 0
    t_predict= 0
    rmse= 0
    
    def __repr__(self):
        return "[" + str(self.t_fit) + "," + str(self.t_predict) + "]"

class Search(object):
    '''
    classdocs
    '''
    
    def __init__(self, X, y, model, params):
        '''
        Constructor
        '''
        self.X= X
        self.y= y
        # self.train_x, self.test_x, self.train_y, self.test_y= split(X,y)
        self.model= model
        self.params= params
        
    def runTest(self, params):
        print(params)
        self.results[str(params)]= []
        
        self.pred_yy= np.empty([len(self.y), 1])
        
        def tt(train, test, params):
            # construct model
            m= self.model.set_params(**params)          
            
            t1= time.clock()
            xxx= m.fit(self.X.iloc[train], self.y.iloc[train])
            t2= time.clock()
            self.pred_y[test, 0]= xxx.predict(self.X.iloc[test])
            t3= time.clock()
            
            # evaluate
            r= Result()
            r.t_fit= t2 - t1
            r.t_predict= t3 - t2
            r.rmse= metrics.mean_squared_error(self.y.iloc[test], self.pred_y[test, 0])
            
            self.results[str(params)].append(r)
        
        # Parallel(n_jobs=2)(delayed(tt)(train, test, params) for (train, test) in self.splits)
        for (train, test) in self.splits:
            tt(train, test, params) 
    
    def report(self, x= -1, fn=None):
        if x == -1:
            y= self.y
            pred_y= self.pred_y
        else:
            y= self.y.iloc[self.splits[x]]
            pred_y= self.pred_yy[self.splits[x], 0]
        
        print(metrics.mean_squared_error(y, pred_y))
#         confusion= metrics.confusion_matrix(y, pred_y)
# #         TP= confusion[1, 1]
# #         FP= confusion[0, 1]
# #         TN= confusion[0, 0]
# #         FN= confusion[1, 0]
#         print(confusion)
#                         
# #         print(TP, FP, TN, FN)
#         
#         rec= metrics.recall_score(y, pred_y, average='macro')
#         acc= metrics.accuracy_score(y, pred_y, normalize=True)
#         if type == 'binary':
#             prec= metrics.precision_score(y, pred_y)
#         else:
#             prec= metrics.precision_score(y, pred_y, average='macro')
#         f= metrics.precision_recall_fscore_support(y, pred_y, average='macro') 
#         print(rec, acc, prec, f)
#         
#         self.confusion= confusion
#         self.writeConfusion(fn)

    def _createList(self):
        def h(params, done=[], values={}):
            if len(done) == len(params.keys()):
                if values not in self.values:
                    self.values.append(copy(values))
                return
            
            for key in sorted(params.keys()):
                if key in done: continue
                
                for value in sorted(params[key]):
                    x= values
                    x[key]= value
                    h(params, done+[key], x)
        
        self.values= []
        h(self.params)
        print(self.values)
 
    def run(self):
        self.splits= []
        k_fold= KFold(n_splits=10)
        for train_index, test_index in k_fold.split(self.X):
            self.splits.append((train_index, test_index))
        self.pred_y= np.empty([len(self.y), 1])
        
        self._createList()
        
        self.results={}
        
        for params in self.values:
            self.runTest(params)
            
        print(self.results)
        
    def predict(self, X):
        model= self.model.fit(self.X, self.y)
        return model.predict(X)
        

if __name__ == '__main__':
    x= Search(0,0,0,{'a':[0,1,2], 'b':[3], 'c':[4,5]})
    x.run()
