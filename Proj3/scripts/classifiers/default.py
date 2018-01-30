import numpy as np
import pandas as pd
import sklearn as sk

def trainDefault():
	datasetName = "testDataset"
	print("Training default classifier (GaussianNB) on default dataset ("+datasetName+")")

	df = pd.read_csv("datasets/"+datasetName+".data")
	print(df)
	
