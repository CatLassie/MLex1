from sklearn.model_selection import GridSearchCV
from sklearn import metrics

def search(model, parameters):
    pass

class Search:
    def __init__(self, X, y, model, parameters):
        '''
        Constructor
        '''
        self.lblFun= None
        
        self.X= X
        self.y= y
        self.model= model
        self.parameters= parameters
    
    def search(self, metric='accuracy', cv=10):
        self.clf= GridSearchCV(self.model, self.parameters, cv=cv, scoring=metric)
        self.clf.fit(self.X, self.y)
        
    def report(self, tblfn=None, cmfn=None):
        pred_y= self.clf.predict(self.X)
        self.labels= list(self.y.unique())
        confusion= metrics.confusion_matrix(self.y, pred_y, labels= self.labels)
        
        print(confusion)
        self.confusion= confusion
        
        self.writeResultsTbl(tblfn)
        self.writeConfusion(cmfn)
        
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
    
    def labelFun(self, fun):
        self.lblFun= fun
    
    def writeConfusion(self, fn=None):
        if fn == None:
            return
        
        print("writing confusion", fn)
        
        print(self.lblFun)
        if self.lblFun == None:
            self.lblFun= lambda x: str(x)
        classes= [self.lblFun(l) for l in self.labels]
        
        out= open(fn,'w')
        
        out.write("\\begin{tabular}{l|" + (len(classes)*"c") + "}\n")
        out.write("\\toprule\n") 
        out.write("&")
        out.write(" & ".join(["\\textbf{" + c.replace("_", "\_") + "}" for c in classes]))
        out.write("\\\\\n")
        out.write("\\midrule\n") 
        
        i= 0
        for row in self.confusion:
            c= classes[i]
            out.write("\\textbf{" + c.replace("_", "\_") + "} & ")
            out.write(" & ".join([str(i) for i in row]))
            out.write("\\\\\n")
            i+= 1
            
        out.write("\\bottomrule\n")
        out.write("\\end{tabular}\n")
        
        out.close()
        
    def writeResultsTbl(self, fn=None):
        if fn == None:
            return
        
        if self.clf.cv_results_['mean_fit_time'].mean() < 0.7:
            scale= 1000
            tt= " (ms)"
        else:
            scale= 1
            tt= " (s)"
            
        ps= list(self.parameters[0].keys())
        
        out= open(fn,'w')
        
        out.write("\\begin{tabular}{" + (len(ps)*"c") + "rr}\n")
        out.write("\\toprule\n") 
        # out.write("\\textbf{Tool} & \textbf{Basic principle} & \textbf{WMM support} & \textbf{Source code available}\\\\\n")
        out.write(" & ".join(["\\textbf{" + k.replace("_", "\_") + "}" for k in ps]))
        out.write(" & \\textbf{time" + tt + "} & \\textbf{score}\\\\\n")
        out.write("\\midrule\n") 
        data= zip(self.clf.cv_results_['params'], self.clf.cv_results_['mean_fit_time'],self.clf.cv_results_['std_fit_time'],
                         self.clf.cv_results_['mean_test_score'], self.clf.cv_results_['std_test_score'])
        if len(ps) > 0:
            key= ps[0]
            sort= sorted(data, key= lambda k: k[0][key])
        else:
            sort= data
        for p,tm,ts,sm,ss in sort:
            pc= " & ".join([str(p[k]) for k in ps])
            s= "{4} & ${0:.2f} \\pm {1:.2f}$ & ${2:.2f} \\pm {3:.2f}$\\\\".format(tm*scale,ts*scale, sm, ss,pc)
            out.write(s + "\n")
            # print(s)
        out.write("\\bottomrule\n")
        out.write("\\end{tabular}\n")
        
        out.close()
    
# if __name__ == "__main__":
#     from sklearn.neighbors import KNeighborsClassifier
#     parameters= [{'n_neighbors' : range(1, 10), 'weights':['uniform', 'distance'], 'p':[1, 2]}]
#     s= Search(X, y, KNeighborsClassifier(), parameters)
#     s.search()
#     s.report()