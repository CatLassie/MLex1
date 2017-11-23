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

def split(X,y, n=0.8):
    return train_test_split(X, y, random_state=1234, test_size=n)

class Test(object):
    '''
    classdocs
    '''
    
    def __init__(self, X, y, model):
        '''
        Constructor
        '''
        self.X= X
        self.y= y
        # self.train_x, self.test_x, self.train_y, self.test_y= split(X,y)
        self.model= model
    
    def train(self):
        k_fold= KFold(n_splits=10)
        # res= cross_validate(self.model, self.X, self.y, cv=k_fold, n_jobs=-1)
        # print(res)
        self.pred_y= cross_val_predict(self.model, self.X, self.y, cv=k_fold)
    
    def report(self):
        from sklearn import metrics
        confusion= metrics.confusion_matrix(self.y, self.pred_y)
        print(confusion)
        TP= confusion[1, 1]
        FP= confusion[0, 1]
        TN= confusion[0, 0]
        FN= confusion[1, 0]
        print(TP, FP, TN, FN)
        
        rec= metrics.recall_score(self.y, self.pred_y, average='micro')
        acc= metrics.accuracy_score(self.y, self.pred_y, normalize=True)
        print(rec, acc)
        
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
        #for train_idx, test_idx in k_fold.split(self.X):
            
    