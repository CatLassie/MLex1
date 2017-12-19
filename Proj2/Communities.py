'''
Created on Dec 14, 2017

@author: martin
'''

import pandas as pd

from Stuff import plot_corr

def load_data():
    h= open("../data/communities.names").read().split("\n")
    cols= [x.split(" ")[0] for x in h]
    cols= cols[:-1]
    
    data= pd.read_csv("../data/communities.data", names=cols)
    
    data= data.where(data != "?", None)
    
    return data, cols

def clean(data, save_plt= False):
    # Remove non-predictive features as per 
    del(data['state'])
    del(data['county'])
    del(data['community'])
    del(data['communityname'])
    del(data['fold'])
    
    if save_plt:
        plot_corr(data, "communities_orig.png")
    
    del(data['numbUrban'])
    del(data['NumUnderPov'])
    del(data['NumIlleg'])
    del(data['NumImmig'])
    del(data['NumInShelters'])
    del(data['NumStreet'])
    del(data['NumKindsDrugsSeiz'])
    
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
    
    # -- OwnOccLowQuart: owner occupied housing - lower quartile value (numeric - decimal)
    # -- OwnOccMedVal: owner occupied housing - median value (numeric - decimal)
    # -- OwnOccHiQuart: owner occupied housing - upper quartile value (numeric - decimal)
    del(data['OwnOccLowQuart'])
    del(data['OwnOccHiQuart'])
    
    # -- medFamInc: median family income (differs from household income for non-family households) (numeric - decimal)
    # -- perCapInc: per capita income (numeric - decimal)
    # -- whitePerCap: per capita income for caucasians (numeric - decimal)
    # -- blackPerCap: per capita income for african americans (numeric - decimal)
    del(data['medFamInc'])
    del(data['perCapInc'])
    
    # -- RentLowQ: rental housing - lower quartile rent (numeric - decimal)
    # -- RentMedian: rental housing - median rent (Census variable H32B from file STF1A) (numeric - decimal)
    # -- RentHighQ: rental housing - upper quartile rent (numeric - decimal)
    # -- MedRent: median gross rent (Census variable H43A from file STF3A - includes utilities) (numeric - decimal)
    # -- MedRentPctHousInc: median gross rent as a percentage of household income (numeric - decimal)
    # -- MedOwnCostPctInc: median owners cost as a percentage of household income - for owners with a mortgage (numeric - decimal)
    # -- MedOwnCostPctIncNoMtg: median owners cost as a percentage of household income - for owners without a mortgag
    del(data['RentMedian'])
    del(data['RentLowQ'])
    del(data['RentHighQ'])
    del(data['MedRentPctHousInc'])
    del(data['MedOwnCostPctInc'])
    del(data['MedOwnCostPctIncNoMtg'])
    
    del(data['PctNotHSGrad'])
    
    del(data['PctLargHouseOccup'])
    
    # correlates with PersPerFam
    del(data['PersPerOccupHous'])
    
    # correlates with agePct65up -> Pct households with social sec income
    del(data['pctWSocSec'])
    
    # oddly it correlates with population
    del(data['HousVacant'])
    
    # correlates with malediv
    del(data['FemalePctDiv'])
    
    del(data['MedRent'])
    
    del(data['racePctWhite'])
    del(data['racePctAsian'])
    del(data['racePctHisp'])
    
    del(data['PctForeignBorn'])
    
    del(data['householdsize'])
    
    del(data['LandArea'])
    
    # correlates with PctBSorMore !!!
    del(data['PctOccupMgmtProf'])
    
    # Correlates with  PctHousOwnOcup
    del(data['PctPersOwnOccup'])
    
    if save_plt:
        plot_corr(data, "communities_final.png")
    
    return data

def missing_vals(data):
    cols= data.columns[data.isnull().any()]
    
    for col in cols:
        #print(col, 'before', data[col])
        #data[col]= data[col].interpolate()
        #print(col, 'after', data[col])
        del(data[col])

    return data

