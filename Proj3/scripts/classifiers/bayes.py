import numpy as np
import pandas as pd
import sklearn as sk

def trainBayes(args):
	if(len(args) > 2):
		datasetName = args[2]
	else:
		datasetName = "Congress.csv"

	print("training GaussianNB on "+ datasetName +" dataset")
	df = pd.read_csv("datasets/"+datasetName)
