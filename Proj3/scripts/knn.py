import numpy as np
import pandas as pd
import sklearn as sk

def trainKNN(args):
	if(len(args) > 2):
		datasetName = args[2]
	else:
		datasetName = "defaultDatasetName"
		
	print("training KNeighborsClassifier on "+ datasetName +" dataset")