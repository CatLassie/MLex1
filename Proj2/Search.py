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
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_predict
from sklearn import metrics
import time
import numpy as np

def split(X,y, n=0.8):
    return train_test_split(X, y, random_state=1234, test_size=n)

class SingleResult:
    t_fit= 0
    t_predict= 0
    rmse= 0
    
    def __repr__(self):
        return "[" + str(self.t_fit) + "," + str(self.t_predict) + "]"
    
class Result:
    results= []

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
        # print(params)
        self.results[str(params)]= Result()
        
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
            r= SingleResult()
            r.t_fit= t2 - t1
            r.t_predict= t3 - t2
            r.rmse= metrics.mean_squared_error(self.y.iloc[test], self.pred_y[test, 0])
            
            self.results[str(params)].results.append(r)
        
        # Parallel(n_jobs=2)(delayed(tt)(train, test, params) for (train, test) in self.splits)
        for (train, test) in self.splits:
            tt(train, test, params)
        
        res= self.results[str(params)].results
        rx= self.results[str(params)]
        
        # compute aggregates
        rx.tfm= np.mean([r.t_fit for r in res])
        rx.tfs= np.std([r.t_fit for r in res])
        rx.tpm= np.mean([r.t_predict for r in res])
        rx.tps= np.std([r.t_predict for r in res])
        rx.sm= np.mean([r.rmse for r in res])
        rx.ss= np.std([r.rmse for r in res])
        
        if rx.sm < self.min:
            self.min= rx.sm
            self.best= params
    
    def reportTable(self):
        # ps= list(self.params.keys())
        ps= list()
        ps.append('mean_fit')
        ps.append('sd_fit')
        ps.append('mean_predict')
        ps.append('sd_predict')
        ps.append('mean_rmse')
        ps.append('sd_rmse')
        
        tfm= []
        tfs= []
        tpm= []
        tps= []
        sm= []
        ss= []
        
        pp= {}
        for k in self.params:
            pp[k]= []
        
        for params in self.values:
            # pc= " & ".join([str(params[k]) for k in params])
            for k in params:
                pp[k].append(params[k])
            
            res= self.results[str(params)]
            
            tfm.append(res.tfm)
            tfs.append(res.tfs)
            tpm.append(res.tpm)
            tps.append(res.tps)
            sm.append(res.sm)
            ss.append(res.ss)
        
        df= pd.DataFrame()
        for k in self.params:
            df[k]= pp[k]
        df['mean_fit']= tfm
        df['sd_fit']= tfs
        df['mean_predict']= tpm
        df['sd_predict']= tps
        df['mean_rmse']= sm
        df['sd_rmse']= ss
        
        return df
        
    def report(self, fn=None):
        # TODO determine scale for all
#         if self.clf.cv_results_['mean_fit_time'].mean() < 0.7:
#             scale= 1000
#             tt= " (ms)"
#         else:
#             scale= 1
#             tt= " (s)"
        print("Best", self.best)
        
        scale= 1000
        tt= " (ms)"
        
        ps= list(self.params.keys())
        
        if fn != None:
            out= open(fn,'w')
        else:
            import sys
            out= sys.stdout
        
        out.write("\\begin{tabular}{" + (len(ps)*"c") + "rr}\n")
        out.write("\\toprule\n")
        out.write(" & ".join(["\\textbf{" + k.replace("_", "\_") + "}" for k in ps]))
        out.write(" & \\textbf{time" + tt + "} & \\textbf{time p" + tt + "} & \\textbf{score}\\\\\n")
        out.write("\\midrule\n")
        
        for params in self.values:
            pc= " & ".join([str(params[k]) for k in params])
            
            res= self.results[str(params)]
            
            s= "{6} & ${0:.2f} \\pm {1:.2f}$ & ${2:.2f} \\pm {3:.2f}$ & ${4:.2f} \\pm {5:.2f}$\\\\".format(res.tfm*scale,res.tfs*scale, res.tpm*scale,res.tps*scale, res.sm, res.ss, pc)
            out.write(s + "\n")
            
            pass

        out.write("\\bottomrule\n")
        out.write("\\end{tabular}\n")
        
        if fn != None:
            out.close()

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
        # print(self.values)
 
    def run(self):
        self.splits= []
        k_fold= KFold(n_splits=10)
        for train_index, test_index in k_fold.split(self.X):
            self.splits.append((train_index, test_index))
        self.pred_y= np.empty([len(self.y), 1])
        
        self._createList()
        
        self.results={}
        
        self.min= np.inf
        
        for params in self.values:
            self.runTest(params)
            
        # print(self.results)
        
    def predict(self, X):
        model= self.model.fit(self.X, self.y)
        return model.predict(X)
        

if __name__ == '__main__':
    x= Search(0,0,0,{'a':[0,1,2], 'b':[3], 'c':[4,5]})
    x.run()
