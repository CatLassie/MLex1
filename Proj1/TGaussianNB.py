'''
Created on Nov 22, 2017

@author: martin
'''

from Test import Test

from sklearn.naive_bayes import GaussianNB

class TGaussianNB(Test):
    '''
    classdocs
    '''
    
    def train(self):
        model= GaussianNB()
        
        # train model
        self.pred_y= model.fit(self.train_x, self.train_y).predict(self.test_x)    