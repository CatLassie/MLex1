from sklearn.model_selection import GridSearchCV
from sklearn import metrics

def search(model, parameters):
    pass

class Search:
    def __init__(self, X, y, model, parameters):
        '''
        Constructor
        '''
        self.X= X
        self.y= y
        self.model= model
        self.parameters= parameters
    
    def search(self, metric='accuracy'):
        self.clf= GridSearchCV(self.model, self.parameters, cv=10, scoring=metric)
        self.clf.fit(self.X, self.y)
        
    def report(self):
        pred_y= self.clf.predict(self.X)
        confusion= metrics.confusion_matrix(self.y, pred_y)
        
        print(confusion)
        
        print(self.clf.best_params_)
        
        rec= metrics.recall_score(self.y, pred_y, average='macro')
        acc= metrics.accuracy_score(self.y, pred_y, normalize=True)
        if type == 'binary':
            prec= metrics.precision_score(self.y, pred_y)
        else:
            prec= metrics.precision_score(self.y, pred_y, average='macro')
        f= metrics.precision_recall_fscore_support(self.y, pred_y, average='macro') 
        print(rec, acc, prec, f)

    def predict(self, X):
        return self.clf.predict(X)