import sys

from classifiers.bayes import trainBayes
from classifiers.dTree import trainDTree
from classifiers.knn import trainKNN
from classifiers.svm import trainSVM

def main():
    if(len(sys.argv) == 1):
        trainBayes(sys.argv)
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