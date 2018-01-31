import numpy as np
import pandas as pd
import sklearn as sk

def trainDefault():
	datasetName = "Congress.csv"
	print("Training default classifier (GaussianNB) on default dataset ("+datasetName+")")

	df = pd.read_csv("datasets/"+datasetName, header=None)
	print(df)
	
