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
try:
    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import KFold
    from sklearn.model_selection import cross_validate
    from sklearn.model_selection import cross_val_predict
except ImportError:
    from sklearn.cross_validation import train_test_split
    from sklearn.cross_validation import cross_validation
from sklearn import metrics
import numpy as np

def split(X,y, n=0.8):
    return train_test_split(X, y, random_state=1234, test_size=n)

class Test(object):
    '''
    classdocs
    '''
    
    def __init__(self, X, y, model, type='binary'):
        '''
        Constructor
        '''
        self.X= X
        self.y= y
        # self.train_x, self.test_x, self.train_y, self.test_y= split(X,y)
        self.model= model
        self.type= type
    
    def train(self):
        k_fold= KFold(n_splits=10)
        # res= cross_validate(self.model, self.X, self.y, cv=k_fold, n_jobs=-1)
        # print(res)
        self.pred_y= cross_val_predict(self.model, self.X, self.y, cv=k_fold)
        self.pred_yy= np.empty([len(self.y), 1])
        x= 0
        self.models= {}
        self.splits= {}
        for train_idx, test_idx in k_fold.split(self.X):
            #print(self.y[train_idx])
            model= self.model.fit(self.X.iloc[train_idx], self.y.iloc[train_idx])
            self.pred_yy[test_idx, 0]= model.predict(self.X.iloc[test_idx])
            self.models[x]= model
            self.splits[x]= test_idx
            x+= 1
    
    def report(self, x= -1):
        if x == -1:
            y= self.y
            pred_y= self.pred_y
        else:
            y= self.y.iloc[self.splits[x]]
            pred_y= self.pred_yy[self.splits[x], 0]
        
        confusion= metrics.confusion_matrix(y, pred_y)
        TP= confusion[1, 1]
        FP= confusion[0, 1]
        TN= confusion[0, 0]
        FN= confusion[1, 0]
        print(confusion)
                        
        print(TP, FP, TN, FN)
        
        rec= metrics.recall_score(y, pred_y, average='macro')
        acc= metrics.accuracy_score(y, pred_y, normalize=True)
        if type == 'binary':
            prec= metrics.precision_score(y, pred_y)
        else:
            prec= metrics.precision_score(y, pred_y, average='macro')
        f= metrics.precision_recall_fscore_support(y, pred_y, average='macro') 
        print(rec, acc, prec, f)
        
        # recall= metrics.recall_score(self.y, self.pred_y)
        # print(recall)
        
    def run_old(self):
        pr = cProfile.Profile()
        pr.enable()
        self.train()
        pr.disable()
        self.predict()
        s = StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())
        self.report()
        
    def run(self):
        self.train()
        self.report()
#         for i in range(0, 10):
#             print(i)
#             self.report(i)
        # print(self.models)
        #for train_idx, test_idx in k_fold.split(self.X):
        
    def predict(self, X):
        model= self.model.fit(self.X, self.y)
        return model.predict(X)
        
            
    
