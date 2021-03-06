'''
Created on Nov 22, 2017

@author: martin
'''

import pandas as pd

from Search import Search
from Test import Test

from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier as DTree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

def remove_missing(data):
    # ret= data[data[4]!="?"]
    ret= data.loc[lambda df: (df[4] != "?") & (df[38] != "?"), :]
    ret.loc[:,4]= ret[4].astype(int)
    ret.loc[:,38]= ret[38].astype(int)
    return ret

def impute_col_avg(data, idx):
    mean= int(data[data[idx]!="?"][idx].astype(int).mean())
    data.loc[data[idx]=="?", idx]= mean

def impute_missing(data):
    impute_col_avg(data, 4)
    impute_col_avg(data, 38)
    return data

def impute_class_avg(data, idx, cidx=0):
    classes= data[data[idx]=="?"][cidx].unique()
    print(classes)
    cmeans= {} #class means
    print(cmeans)
    for clasz in classes:
        cmeans[int(clasz)]= int(data[(data[cidx]==clasz) & (data[idx] != "?")][idx].astype(int).mean())
    print(cmeans)
    for key, val in cmeans.items():
        data.loc[(data[idx]=="?") & (data[cidx]==key), idx]= val

def impute_missing2(data):
    impute_class_avg(data, 4)
    impute_class_avg(data, 38)
    return data

def main(fn):
    # data= pd.read_csv("../data/kddcup.data_10_percent_corrected", names=cols)
    data= pd.read_csv(fn, header=-1)
    
    # data= remove_missing(data)
    # data= impute_missing(data)
    data= impute_missing2(data)       
    
    # Features to be used in classification
    features= [x for x in range(1, len(data.columns))]
    
    X= data[features]
    y= data[0]
    
    #h= TGaussianNB(X, y)
    #h.run()
    print("GaussianNB")
    h= Test(X, y, GaussianNB())
    h.run()
    h.report(fn="../Report/results/cancer.gnb.cm.tex")
    s= Search(X, y, GaussianNB(), [{}])
    s.search()
    s.report("../Report/results/cancer.gnb.tex")
    
    print("DTree neu")
    parameters=[{'criterion' :['gini', 'entropy'], 'max_features': ['auto', 'sqrt', 'log2']}]
    s= Search(X, y, DTree(), parameters)
    s.search()
    s.report("../Report/results/cancer.dt.tex")
    h= Test(X, y, DTree(max_features='log2',criterion='gini',random_state=1234))
    h.run()
    h.report(fn="../Report/results/cancer.dt.cm.tex")
    print("RF")
    parameters= [{'n_estimators' : range(1, 15), 'criterion' :['gini', 'entropy'], 'max_features': ['auto', 'sqrt', 'log2']}]
    s= Search(X, y, RandomForestClassifier(), parameters)
    s.search()
    s.report("../Report/results/cancer.rf.tex", )
    h= Test(X, y, RandomForestClassifier(n_estimators= 6, criterion='gini', max_features='sqrt',random_state=1234))
    h.run()
    h.report(fn="../Report/results/cancer.rf.cm.tex")
    
    parameters=[{'kernel' : ['linear','sigmoid', 'rbf', 'poly'], 'C' : [0.1, 1, 10, 11, 20]}]
    print("SVM")
    from sklearn import preprocessing
    X_scaled = preprocessing.scale(X)
    s= Search(X_scaled, y, SVC(), parameters)
    s.search()
    s.report("../Report/results/cancer.svm.tex")
    h= Test(X, y, SVC(C=11, kernel='poly'))
    h.run()
    h.report(fn="../Report/results/cancer.svm.cm.tex")
    
    print("KNeighborsClassifier")
    parameters= [{'n_neighbors' : range(4, 8), 'weights':['uniform', 'distance'], 'p':[1, 2]}]
    s= Search(X, y, KNeighborsClassifier(), parameters)
    s.search()
    s.report("../Report/results/cancer.knn.tex")
    h= Test(X, y, KNeighborsClassifier(n_neighbors=5, weights='uniform', p=2))
    h.run()
    h.report(fn="../Report/results/cancer.knn.cm.tex")

if __name__ == '__main__':
    # main("../data/kddcup.data.corrected")
    main("../data/lung-cancer.data")

