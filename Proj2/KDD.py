'''
Created on Dec 14, 2017

@author: martin
'''

import pandas as pd

def load_data(x="train"):
    # dtypes= {"ODATEDW":"category", "OSOURCE":"category", "TCODE":"category", "STATE":"category", "ZIP":"category", "MAILCODE":"category", "PVASTATE":"category", "DOB":"category", "NOEXCH":"float", "RECINHSE":"category", "RECP3":"category", "RECPGVG":"category", "RECSWEEP":"category", "MDMAUD":"category", "DOMAIN":"category", "CLUSTER":"float", "AGE":"float", "AGEFLAG":"category", "HOMEOWNR":"category", "CHILD03":"category", "CHILD07":"category", "CHILD12":"category", "CHILD18":"category", "NUMCHLD":"float", "INCOME":"float", "GENDER":"category", "WEALTH1":"float", "HIT":"float", "MBCRAFT":"float", "MBGARDEN":"float", "MBBOOKS":"float", "MBCOLECT":"float", "MAGFAML":"float", "MAGFEM":"float", "MAGMALE":"float", "PUBGARDN":"float", "PUBCULIN":"float", "PUBHLTH":"float", "PUBDOITY":"float", "PUBNEWFN":"float", "PUBPHOTO":"float", "PUBOPP":"float", "DATASRCE":"float", "MALEMILI":"float", "MALEVET":"float", "VIETVETS":"float", "WWIIVETS":"float", "LOCALGOV":"float", "STATEGOV":"float", "FEDGOV":"float", "SOLP3":"float", "SOLIH":"float", "MAJOR":"float", "WEALTH2":"float", "GEOCODE":"float", "COLLECT1":"category", "VETERANS":"category", "BIBLE":"category", "CATLG":"category", "HOMEE":"category", "PETS":"category", "CDPLAY":"category", "STEREO":"category", "PCOWNERS":"category", "PHOTO":"category", "CRAFTS":"category", "FISHER":"category", "GARDENIN":"category", "BOATS":"category", "WALKER":"category", "KIDSTUFF":"category", "CARDS":"category", "PLATES":"category", "LIFESRC":"float", "PEPSTRFL":"category", "POP901":"float", "POP902":"float", "POP903":"float", "POP90C1":"float", "POP90C2":"float", "POP90C3":"float", "POP90C4":"float", "POP90C5":"float", "ETH1":"float", "ETH2":"float", "ETH3":"float", "ETH4":"float", "ETH5":"float", "ETH6":"float", "ETH7":"float", "ETH8":"float", "ETH9":"float", "ETH10":"float", "ETH11":"float", "ETH12":"float", "ETH13":"float", "ETH14":"float", "ETH15":"float", "ETH16":"float", "AGE901":"float", "AGE902":"float", "AGE903":"float", "AGE904":"float", "AGE905":"float", "AGE906":"float", "AGE907":"float", "CHIL1":"float", "CHIL2":"float", "CHIL3":"float", "AGEC1":"float", "AGEC2":"float", "AGEC3":"float", "AGEC4":"float", "AGEC5":"float", "AGEC6":"float", "AGEC7":"float", "CHILC1":"float", "CHILC2":"float", "CHILC3":"float", "CHILC4":"float", "CHILC5":"float", "HHAGE1":"float", "HHAGE2":"float", "HHAGE3":"float", "HHN1":"float", "HHN2":"float", "HHN3":"float", "HHN4":"float", "HHN5":"float", "HHN6":"float", "MARR1":"float", "MARR2":"float", "MARR3":"float", "MARR4":"float", "HHP1":"float", "HHP2":"float", "DW1":"float", "DW2":"float", "DW3":"float", "DW4":"float", "DW5":"float", "DW6":"float", "DW7":"float", "DW8":"float", "DW9":"float", "HV1":"float", "HV2":"float", "HV3":"float", "HV4":"float", "HU1":"float", "HU2":"float", "HU3":"float", "HU4":"float", "HU5":"float", "HHD1":"float", "HHD2":"float", "HHD3":"float", "HHD4":"float", "HHD5":"float", "HHD6":"float", "HHD7":"float", "HHD8":"float", "HHD9":"float", "HHD10":"float", "HHD11":"float", "HHD12":"float", "ETHC1":"float", "ETHC2":"float", "ETHC3":"float", "ETHC4":"float", "ETHC5":"float", "ETHC6":"float", "HVP1":"float", "HVP2":"float", "HVP3":"float", "HVP4":"float", "HVP5":"float", "HVP6":"float", "HUR1":"float", "HUR2":"float", "RHP1":"float", "RHP2":"float", "RHP3":"float", "RHP4":"float", "HUPA1":"float", "HUPA2":"float", "HUPA3":"float", "HUPA4":"float", "HUPA5":"float", "HUPA6":"float", "HUPA7":"float", "RP1":"float", "RP2":"float", "RP3":"float", "RP4":"float", "MSA":"float", "ADI":"float", "DMA":"float", "IC1":"float", "IC2":"float", "IC3":"float", "IC4":"float", "IC5":"float", "IC6":"float", "IC7":"float", "IC8":"float", "IC9":"float", "IC10":"float", "IC11":"float", "IC12":"float", "IC13":"float", "IC14":"float", "IC15":"float", "IC16":"float", "IC17":"float", "IC18":"float", "IC19":"float", "IC20":"float", "IC21":"float", "IC22":"float", "IC23":"float", "HHAS1":"float", "HHAS2":"float", "HHAS3":"float", "HHAS4":"float", "MC1":"float", "MC2":"float", "MC3":"float", "TPE1":"float", "TPE2":"float", "TPE3":"float", "TPE4":"float", "TPE5":"float", "TPE6":"float", "TPE7":"float", "TPE8":"float", "TPE9":"float", "PEC1":"float", "PEC2":"float", "TPE10":"float", "TPE11":"float", "TPE12":"float", "TPE13":"float", "LFC1":"float", "LFC2":"float", "LFC3":"float", "LFC4":"float", "LFC5":"float", "LFC6":"float", "LFC7":"float", "LFC8":"float", "LFC9":"float", "LFC10":"float", "OCC1":"float", "OCC2":"float", "OCC3":"float", "OCC4":"float", "OCC5":"float", "OCC6":"float", "OCC7":"float", "OCC8":"float", "OCC9":"float", "OCC10":"float", "OCC11":"float", "OCC12":"float", "OCC13":"float", "EIC1":"float", "EIC2":"float", "EIC3":"float", "EIC4":"float", "EIC5":"float", "EIC6":"float", "EIC7":"float", "EIC8":"float", "EIC9":"float", "EIC10":"float", "EIC11":"float", "EIC12":"float", "EIC13":"float", "EIC14":"float", "EIC15":"float", "EIC16":"float", "OEDC1":"float", "OEDC2":"float", "OEDC3":"float", "OEDC4":"float", "OEDC5":"float", "OEDC6":"float", "OEDC7":"float", "EC1":"float", "EC2":"float", "EC3":"float", "EC4":"float", "EC5":"float", "EC6":"float", "EC7":"float", "EC8":"float", "SEC1":"float", "SEC2":"float", "SEC3":"float", "SEC4":"float", "SEC5":"float", "AFC1":"float", "AFC2":"float", "AFC3":"float", "AFC4":"float", "AFC5":"float", "AFC6":"float", "VC1":"float", "VC2":"float", "VC3":"float", "VC4":"float", "ANC1":"float", "ANC2":"float", "ANC3":"float", "ANC4":"float", "ANC5":"float", "ANC6":"float", "ANC7":"float", "ANC8":"float", "ANC9":"float", "ANC10":"float", "ANC11":"float", "ANC12":"float", "ANC13":"float", "ANC14":"float", "ANC15":"float", "POBC1":"float", "POBC2":"float", "LSC1":"float", "LSC2":"float", "LSC3":"float", "LSC4":"float", "VOC1":"float", "VOC2":"float", "VOC3":"float", "HC1":"float", "HC2":"float", "HC3":"float", "HC4":"float", "HC5":"float", "HC6":"float", "HC7":"float", "HC8":"float", "HC9":"float", "HC10":"float", "HC11":"float", "HC12":"float", "HC13":"float", "HC14":"float", "HC15":"float", "HC16":"float", "HC17":"float", "HC18":"float", "HC19":"float", "HC20":"float", "HC21":"float", "MHUC1":"float", "MHUC2":"float", "AC1":"float", "AC2":"float", "ADATE_2":"float", "ADATE_3":"float", "ADATE_4":"float", "ADATE_5":"float", "ADATE_6":"float", "ADATE_7":"float", "ADATE_8":"float", "ADATE_9":"float", "ADATE_10":"float", "ADATE_11":"float", "ADATE_12":"float", "ADATE_13":"float", "ADATE_14":"float", "ADATE_15":"float", "ADATE_16":"float", "ADATE_17":"float", "ADATE_18":"float", "ADATE_19":"float", "ADATE_20":"float", "ADATE_21":"float", "ADATE_22":"float", "ADATE_23":"float", "ADATE_24":"float", "RFA_2":"category", "RFA_3":"category", "RFA_4":"category", "RFA_5":"category", "RFA_6":"category", "RFA_7":"category", "RFA_8":"category", "RFA_9":"category", "RFA_10":"category", "RFA_11":"category", "RFA_12":"category", "RFA_13":"category", "RFA_14":"category", "RFA_15":"category", "RFA_16":"category", "RFA_17":"category", "RFA_18":"category", "RFA_19":"category", "RFA_20":"category", "RFA_21":"category", "RFA_22":"category", "RFA_23":"category", "RFA_24":"category", "CARDPROM":"float", "MAXADATE":"float", "NUMPROM":"float", "CARDPM12":"float", "NUMPRM12":"float", "RDATE_3":"float", "RDATE_4":"float", "RDATE_5":"float", "RDATE_6":"float", "RDATE_7":"float", "RDATE_8":"float", "RDATE_9":"float", "RDATE_10":"float", "RDATE_11":"float", "RDATE_12":"float", "RDATE_13":"float", "RDATE_14":"float", "RDATE_15":"float", "RDATE_16":"float", "RDATE_17":"float", "RDATE_18":"float", "RDATE_19":"float", "RDATE_20":"float", "RDATE_21":"float", "RDATE_22":"float", "RDATE_23":"float", "RDATE_24":"float", "RAMNT_3":"float", "RAMNT_4":"float", "RAMNT_5":"float", "RAMNT_6":"float", "RAMNT_7":"float", "RAMNT_8":"float", "RAMNT_9":"float", "RAMNT_10":"float", "RAMNT_11":"float", "RAMNT_12":"float", "RAMNT_13":"float", "RAMNT_14":"float", "RAMNT_15":"float", "RAMNT_16":"float", "RAMNT_17":"float", "RAMNT_18":"float", "RAMNT_19":"float", "RAMNT_20":"float", "RAMNT_21":"float", "RAMNT_22":"float", "RAMNT_23":"float", "RAMNT_24":"float", "RAMNTALL":"float", "NGIFTALL":"float", "CARDGIFT":"float", "MINRAMNT":"float", "MINRDATE":"float", "MAXRAMNT":"float", "MAXRDATE":"float", "LASTGIFT":"float", "LASTDATE":"float", "FISTDATE":"float", "NEXTDATE":"float", "TIMELAG":"float", "AVGGIFT":"float", "CONTROLN":"float", "TARGET_D":"float", "HPHONE_D":"float", "RFA_2R":"category", "RFA_2F":"float", "RFA_2A":"category", "MDMAUD_R":"category", "MDMAUD_F":"category", "MDMAUD_A":"category", "CLUSTER2":"float", "GEOCODE2":"category"}
    dtypes= {"RECPGVG":"object"}
    
    #data= pd.read_csv("data/cup98ID.shuf.5000." + type + ".csv", na_values= " ")
    data= pd.read_csv("data/cup98ID.shuf.5000." + x + ".csv", na_values= " ", dtype= dtypes)
    
    data= data.where(data != "?", None)
    
    return data, data.columns

def clean(data, save_plt= False):
    del(data['NOEXCH'])
    
    del(data['PVASTATE'])
    del(data['FISTDATE'])
    del(data['NEXTDATE'])
    
    del(data['SOLIH'])
    
    del(data['POP902'])
    del(data['POP903'])
    
    del(data['AGE902'])
    del(data['AGE903'])
    del(data['AGE904'])
    del(data['AGE905'])
    del(data['AGE906'])
    del(data['AGE907'])
    del(data['HHAGE2'])
    del(data['HHAGE3'])
    
    del(data['HHD3'])
    del(data['HHP1'])
    del(data['HHP2'])
    
    del(data['DW5'])
    del(data['DW6'])
    
    del(data['HV2'])
    del(data['HV4'])
    
    del(data['IC2'])
    del(data['IC3'])
    del(data['IC4'])
    del(data['IC15'])
    del(data['IC16'])
    del(data['IC17'])
    del(data['IC18'])
    del(data['IC19'])
    del(data['IC20'])
    del(data['IC21'])
    del(data['IC22'])
    del(data['IC23'])
    
    del(data['RHP2'])
    del(data['RHP3'])
    
    del(data['MDMAUD'])
    del(data['MDMAUD_R'])
    del(data['MDMAUD_F'])
    del(data['MDMAUD_A'])
    
    del(data['TPE4'])
    
    del(data['LFC2'])
    del(data['LFC3'])
    del(data['LFC4'])
    del(data['LFC5'])
    
    del(data['EIC1'])
    
    del(data['AFC5'])
    
    del(data['RFA_4'])
    del(data['RFA_12'])
    
    del(data['RDATE_3'])
    del(data['RDATE_8'])
    
    del(data['RAMNT_3'])
    del(data['RAMNT_4'])
    del(data['RAMNT_6'])
    del(data['RAMNT_7'])
    del(data['RAMNT_8'])
    del(data['RAMNT_9'])
    del(data['RAMNT_10'])
    del(data['RAMNT_11'])
    del(data['RAMNT_12'])
    del(data['RAMNT_13'])
    del(data['RAMNT_14'])
    del(data['RAMNT_15'])
    del(data['RAMNT_16'])
    del(data['RAMNT_17'])
    del(data['RAMNT_18'])
    del(data['RAMNT_19'])
    del(data['RAMNT_20'])
    del(data['RAMNT_21'])
    del(data['RAMNT_22'])
    del(data['RAMNT_23'])
    del(data['RAMNT_24'])
    
    del(data['WEALTH2'])
    
    del(data['ETH13'])
    
    del(data['HHN4'])
    
    return data

def xxx(data):
    cat_cols= data.select_dtypes(['object']).columns
    for col in cat_cols:
        data[col]= data[col].astype('category')
    data[cat_cols] = data[cat_cols].apply(lambda x: x.cat.codes)
    
    return data

def missing_vals(data):
    cols= data.columns[data.isnull().any()]
    
    for col in cols:
        #print(col, 'before', data[col])
        #data[col]= data[col].interpolate()
        #print(col, 'after', data[col])
        del(data[col])

    return data

