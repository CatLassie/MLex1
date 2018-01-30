import sys

from default import trainDefault
from bayes import trainBayes
from dTree import trainDTree
from knn import trainKNN
from svm import trainSVM

def main():
    if(len(sys.argv) == 1):
        trainDefault()
    elif(len(sys.argv) > 1):
        chooseClassifier()

def chooseClassifier():
	if(sys.argv[1] == "Bayes"):
		trainBayes(sys.argv)
	elif(sys.argv[1] == "DTree"):
		trainDTree(sys.argv)
	elif(sys.argv[1] == "KNN"):
		trainKNN(sys.argv)
	elif(sys.argv[1] == "SVM"):
		trainSVM(sys.argv)
	else:
		print("Classifier name not recognized!")
		
if __name__ == '__main__':
	main()