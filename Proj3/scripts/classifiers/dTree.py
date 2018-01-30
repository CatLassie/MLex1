import numpy as np
import pandas as pd
import sklearn as sk

def trainDTree(args):
	if(len(args) > 2):
		datasetName = args[2]
	else:
		datasetName = "defaultDatasetName"

	print("training DecisionTreeClassifier on "+ datasetName +" dataset")
	df = pd.read_csv("datasets/"+datasetName+".data")
