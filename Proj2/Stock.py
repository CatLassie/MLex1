'''
Created on Dec 14, 2017

@author: martin
'''

import pandas as pd

def load_data(x="train"):
    data= pd.read_csv("data/stock portfolio performance data set.csv")
    
    return data

def prepare(data):
    cols= ["Annual Return", "Excess Return", "Total Risk", "Abs. Win Rate", "Rel. Win Rate"]
    
    for col in cols:
        print(col)
        data[col]= data[col].apply(lambda x: float(x[:-1].replace(',','.'))/100)
    
    cols= ["Large B/P", "Large ROE", "Large S/P", "Large Return Rate in the last quarter", "Large Market Value", "Small systematic Risk", "Systematic Risk"]
    for col in cols:
        print(col)
        data[col]= data[col].apply(lambda x: float(x.replace(',','.')))
    
        
    del(data['ID'])
    
    return data
