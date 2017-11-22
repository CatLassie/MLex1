'''
Created on Nov 22, 2017

@author: martin
'''

import pandas as pd
from TGaussianNB import TGaussianNB

def main():
    data= pd.read_csv("../data/forestfires.csv")
    
    data['fire']= data['area'] > 0

    # Features to be used in classification
    features= ['X', 'Y', 'FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind', 'rain']
    
    X= data[features]
    y= data['fire'].astype(int)
    
    h= TGaussianNB(X, y)
    h.run()

if __name__ == '__main__':
    main()