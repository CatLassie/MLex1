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
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_predict
from sklearn import metrics
import numpy as np

def split(X,y, n=0.8):
    return train_test_split(X, y, random_state=1234, test_size=n)

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
        
    def train(self, params):
        print(params)
        self.pred_yy= np.empty([len(self.y), 1])
        
        def tt(train, test, params):
            pass
        
        Parallel(n_jobs=2)(delayed(tt)(train, test, params) for (train, test) in self.splits)
    
    def trainOld(self, full=False, k=10):
        
        k_fold= KFold(n_splits=k)
        # res= cross_validate(self.model, self.X, self.y, cv=k_fold, n_jobs=-1)
        # print(res)
        self.pred_y= cross_val_predict(self.model, self.X, self.y, cv=k_fold)
        
        if full == True:
            self.pred_yy= np.empty([len(self.y), 1])
            x= 0
            self.models= {}
            self.splits= {}
            for train_idx, test_idx in k_fold.split(self.X):
                if type(self.X) == np.ndarray:
                    train= self.X[train_idx]
                else:
                    train= self.X.iloc[train_idx]
                model= self.model.fit(train, self.y.iloc[train_idx])
                self.pred_yy[test_idx, 0]= model.predict(self.X.iloc[test_idx])
                self.models[x]= model
                self.splits[x]= test_idx
                x+= 1
    
    def labelFun(self, fun):
        self.lblFun= fun
    
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
        
        self._createList()
        
        for params in self.values:
            self.train(params)
        
        #self.report()
#         for i in range(0, 10):
#             print(i)
#             self.report(i)
        # print(self.models)
        #for train_idx, test_idx in k_fold.split(self.X):
        
    def predict(self, X):
        model= self.model.fit(self.X, self.y)
        return model.predict(X)
        

if __name__ == '__main__':
    x= Search(0,0,0,{'a':[0,1,2], 'b':[3], 'c':[4,5]})
    x.run()
