import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy import exp
from scipy.linalg import eigh
from sklearn import svm, preprocessing
from sklearn.metrics import classification_report as cr
from sklearn.decomposition import KernelPCA

def mergeData(trainData, testData):
	x = np.zeros((trainData.shape[0] + testData.shape[0], trainData.shape[1]))
	x[:trainData.shape[0], :] = trainData
	x[trainData.shape[0]:, :] = testData
	return x

def getDataMatrix(file, intOrFloat):
	#intOrFloat decides whether data should be int or float
	if (intOrFloat == 1): 
		featureVectors = []
		for line in file :
			vector = line.strip().lower().split(' ')
			featureVectors.append(vector)
		data = np.array(featureVectors)
		data = data.astype(float)
	else:
		trainLabels = []
		for line in file :
			vector = line
			trainLabels.append(vector)
		data = np.array(trainLabels)
		data = data.astype(int)
	return data

def addLabels(data, trainLabels):
	b = np.zeros((data.shape[0], data.shape[1] + 1))
	b[:, :-1] = data
	b[:, -1] = trainLabels
	return b

def kPCA(X, gamma, k):
	distances = pdist(X, 'sqeuclidean')
	symmetricDistances = squareform(distances)
	K = exp(-gamma * symmetricDistances)
	N = K.shape[0]
	one_N = np.ones((N,N))/N
	normalizedK = K - one_N.dot(K) - K.dot(one_N)	+ one_N.dot(K).dot(one_N)
	eigenValues, eigenVectors = eigh(normalizedK)
	alphas = np.column_stack((eigenVectors[:,-i] for i in range(1,k+1)))
	lambdas = [eigenValues[-i] for i in range(1,k+1)]
	return alphas, lambdas

def project(testData, X, k, gamma, alphas, lambdas): 
	Data = np.zeros((testData.shape[0], k))
	for i in xrange(testData.shape[0]):
		Data[i, :] = project_x(testData[i], X, gamma, alphas, lambdas)
	return Data 

def project_x(x_new, X, gamma, alphas, lambdas):
    pair_dist = np.array([np.sum((x_new-row)**2) for row in X])
    k = np.exp(-gamma * pair_dist)
    return k.dot(alphas / lambdas)

file = open('arcene_train.data.txt')
X = getDataMatrix(file, 1)
file = open('arcene_train.labels.txt')
trainLabels = getDataMatrix(file, 0)
file = open('arcene_valid.data.txt')
testData = getDataMatrix(file, 1)
file = open('arcene_valid.labels.txt')
testLabels = getDataMatrix(file, 0)

K = 100
gamma = 15
alphas, lambdas = kPCA(X, gamma, K)
testData = project(testData, X, K, gamma, alphas, lambdas)

