'''
Created on Dec 12, 2017

@author: martin
'''

import pandas as pd

from Stuff import plot_corr
from Test import Test

from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor

def main():
    h= open("../data/communities.names").read().split("\n")
    cols= [x.split(" ")[0] for x in h]
    cols= cols[:-1]
    
    data= pd.read_csv("../data/communities.data", names=cols)
    
    # Remove non-predictive features as per 
    del(data['state'])
    del(data['county'])
    del(data['community'])
    del(data['communityname'])
    del(data['fold'])
    
    # plot_corr(data, "communities.cor.orig.png")
    
    del(data['numbUrban'])
    del(data['NumUnderPov'])
    del(data['NumIlleg'])
    del(data['NumImmig'])
    del(data['NumInShelters'])
    del(data['NumStreet'])
    del(data['NumKindsDrugsSeiz'])
    
    # plot_corr(data, "communities.cor.withoutnum.png")
    
    # 'a corse the correlate. We just hope it works when we only keep agePct16t24
    del(data['agePct12t29'])
    del(data['agePct12t21'])
    
    # -- PctImmigRecent: percentage of _immigrants_ who immigated within last 3 years (numeric - decimal)
    # -- PctImmigRec5: percentage of _immigrants_ who immigated within last 5 years (numeric - decimal)
    # -- PctImmigRec8: percentage of _immigrants_ who immigated within last 8 years (numeric - decimal)
    # -- PctRecentImmig: percent of _population_ who have immigrated within the last 3 years (numeric - decimal)
    # -- PctRecImmig5: percent of _population_ who have immigrated within the last 5 years (numeric - decimal)
    # -- PctRecImmig8: percent of _population_ who have immigrated within the last 8 years (numeric - decimal)
    # -- PctRecImmig10: percent of _population_ who have immigrated within the last 10 years (numeric - decimal)
    del(data['PctImmigRecent'])
    del(data['PctImmigRec10'])
    del(data['PctImmigRec8'])
    del(data['PctImmigRec5'])
    data['PctRecImmig8to10']= data['PctRecImmig10'] - data['PctRecImmig8']
    data['PctRecImg5to8']= data['PctRecImmig8'] - data['PctRecImmig5']
    data['PctRecImm3to5']= data['PctRecImmig5'] - data['PctRecentImmig']
    del(data['PctRecImmig10'])
    del(data['PctRecImmig8'])
    del(data['PctRecImmig5'])
    
    # -- PctFam2Par: percentage of families (with kids) that are headed by two parents (numeric - decimal)
    # -- PctKids2Par: percentage of kids in family housing with two parents (numeric - decimal)
    # -- PctYoungKids2Par: percent of kids 4 and under in two parent households (numeric - decimal)
    # -- PctTeen2Par: percent of kids age 12-17 in two parent households (numeric - decimal)
    # -- PctWorkMomYoungKids: percentage of moms of kids 6 and under in labor force (numeric - decimal)
    # -- PctWorkMom: percentage of moms of kids under 18 in labor force (numeric - decimal)
    del(data['PctKids2Par'])
    del(data['PctYoungKids2Par'])
    del(data['PctTeen2Par'])
    del(data['PctWorkMom'])
    
    # -- MalePctDivorce: percentage of males who are divorced (numeric - decimal)
    # -- MalePctNevMarr: percentage of males who have never married (numeric - decimal)
    # -- FemalePctDiv: percentage of females who are divorced (numeric - decimal)
    # -- TotalPctDiv: percentage of population who are divorced (numeric - decim
    del(data['TotalPctDiv'])
    
    # plot_corr(data, "communities.cor.final.png")
    
    data= data.where(data != "?", None)
    
    print(data.shape)
    # Now just to do smthg
    data= data.dropna()
    
    print(data.shape)
    y= data['ViolentCrimesPerPop']
    del(data['ViolentCrimesPerPop'])
    X= data
    # print(y)
    
    h= Test(X, y, DecisionTreeRegressor())
    h.run()
    h= Test(X, y, KNeighborsRegressor())
    h.run()
    # h.labelFun(labelFunD)
    # h.report(fn="../Report/results/congress.knn.cm.tex")
    
    print(data.shape)
    
if __name__ == '__main__':
    main()
