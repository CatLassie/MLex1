'''
Created on Nov 22, 2017

@author: martin
'''

from __future__ import print_function

def split(X,y, n=0.8):
    from sklearn.cross_validation import train_test_split
    return train_test_split(X, y, random_state=1234, test_size=n)

class Test(object):
    '''
    classdocs
    '''
    
    def __init__(self, X, y):
        '''
        Constructor
        '''
        self.X= X
        self.y= y
        self.train_x, self.test_x, self.train_y, self.test_y= split(X,y)
    
    def train(self):
        pass
    
    def report(self):
        from sklearn import metrics
        confusion= metrics.confusion_matrix(self.test_y, self.pred_y)
        print(confusion)
        TP= confusion[1, 1]
        FP= confusion[0, 1]
        TN= confusion[0, 0]
        FN= confusion[1, 0]
        
        recall= metrics.recall_score(self.test_y, self.pred_y)
        print(recall)
        
    def run(self):
        self.train()
        self.report()
