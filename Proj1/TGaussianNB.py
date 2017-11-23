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
        gnb= GaussianNB()
        
        # train model
        self.model= gnb.fit(self.train_x, self.train_y)
        
    def predict(self):
        self.pred_y= self.model.predict(self.test_x)    
