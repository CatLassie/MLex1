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
    
    def search(self):
        self.clf= GridSearchCV(self.model, self.parameters, cv=10)
        self.clf.fit(self.X, self.y)
        
    def report(self):
        pred_y= self.clf.predict(self.X)
        confusion= metrics.confusion_matrix(self.y, pred_y)
        
        print(confusion)
        
        print(self.clf.get_params())
        
        rec= metrics.recall_score(self.y, pred_y, average='macro')
        acc= metrics.accuracy_score(self.y, pred_y, normalize=True)
        if type == 'binary':
            prec= metrics.precision_score(self.y, pred_y)
        else:
            prec= metrics.precision_score(self.y, pred_y, average='macro')
        f1= 2 * ( prec * rec) / (prec+rec)
        print(rec, acc, prec, f1)

    def predict(self, X):
        return self.clf.predict(X)