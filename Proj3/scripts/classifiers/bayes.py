import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

def trainBayes(args):
	if(len(args) > 2):
		datasetName = args[2]
	else:
		datasetName = "Congress.csv"

	print("training GaussianNB on "+ datasetName +" dataset")
	df = pd.read_csv("datasets/"+datasetName, header=None)
	
	df.replace(to_replace="n", value=0, inplace=True)
	df.replace(to_replace="y", value=1, inplace=True)
	df.replace(to_replace="?", value=2, inplace=True)
	#print(df.loc[0:1])
	
	#feature/label train/test splitting
	features = df.columns.tolist()
	features.remove(0)
	labels = [0]
	X = df[features].values
	y = df[labels].values
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1234)
	X_test_1, X_test_2, y_test_1, y_test_2 = train_test_split(X_test, y_test, test_size=0.5, random_state=1234)
	
	model = GaussianNB()
	model.fit(X_train, y_train.ravel())
	
	print(model.score(X_test_1, y_test_1))
	print(model.score(X_test_2, y_test_2))