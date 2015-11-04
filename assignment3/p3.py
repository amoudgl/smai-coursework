import numpy as np
from numpy import linalg as LA
import math

def gaussian(x, v, M, V):
	G = (math.exp(-(math.pow((v - M[x]),2)/(2 * V[x]))))
	return G
	
def findVariance(C):
	return np.var(C, axis = 0)	

def findMean(C):
	return np.mean(C, axis = 0)

def pcaTransform(data, k):	
	data = data - np.mean(data, axis = 0)
	covarianceMatrix = np.cov(np.transpose(data))
	eigenValues, eigenVectors = LA.eig(covarianceMatrix)
	s = np.argsort(eigenValues)[::-1]
	ev = np.zeros(eigenVectors.shape)
	for i in xrange(eigenValues.shape[0]):
		ev[:, i] = eigenVectors[:, s[i]]
	data = np.matrix(data)
	data = np.transpose(data)
	eigenVectors = ev[:, :k]
	eigenVectors = np.matrix(eigenVectors)
	eigenVectors = np.transpose(eigenVectors)
	return eigenVectors

def addLabels(data, trainLabels):
	b = np.zeros((data.shape[0], data.shape[1] + 1))
	b[:, :-1] = data
	b[:, -1] = trainLabels
	return b

def mergeData(trainData, testData):
	x = np.zeros((trainData.shape[0] + testData.shape[0], trainData.shape[1]))
	x[:trainData.shape[0], :] = trainData
	x[trainData.shape[0]:, :] = testData
	return x

def project(data, eigenVectors):
	data = data - np.mean(data, axis = 0)
	data = np.matrix(data)
	data = np.transpose(data)
	newData = eigenVectors * data	
	newData = np.transpose(newData)
	newData = np.array(newData)
	return newData


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

file = open('arcene_train.data.txt')
data = getDataMatrix(file, 1)
file = open('arcene_train.labels.txt')
trainLabels = getDataMatrix(file, 0)
file = open('arcene_valid.data.txt')
testData = getDataMatrix(file, 1)
file = open('arcene_valid.labels.txt')
testLabels = getDataMatrix(file, 0)
#PCA
k = 1000 
ev = pcaTransform(data, k)
trainData = project(data, ev)
testData = project(testData, ev)
trainData = addLabels(trainData, trainLabels)
testData = addLabels(testData, testLabels)

C0 = trainData[trainData[:, -1] == -1]
C1 = trainData[trainData[:, -1] == 1]
V0 = findVariance(C0[:, :-1])
V1 = findVariance(C1[:, :-1])
M0 = findMean(C0[:, :-1])
M1 = findMean(C1[:, :-1])
pr0 = float(C0.shape[0])/float(trainData.shape[0])
pr1 = float(C1.shape[0])/float(trainData.shape[0])
if (pr0 > pr1):
	maxPrior = int(-1)
else:
	maxPrior = int(1)
print maxPrior
L = math.pow(10, -323)
MAX = -math.pow(10, 300)

#Testing phase
totalValues = testData.shape[0]	
myPrediction = np.zeros([totalValues])
for i in xrange(0, totalValues):
	sample = testData[i, :]
	sample = sample.tolist()
	ans0 = math.log(float(pr0))
	ans1 = math.log(float(pr1))
	count = 0;
	for j in xrange(0, len(sample) - 1):
		g1 = gaussian(j, sample[j], M0, V0)
		g2 = gaussian(j, sample[j], M1, V1)
		if (g1 < L):
			ans0 = MAX
		if(g2 < L):
			ans1 = MAX
		if(ans0 > MAX):
			ans0 = ans0 + math.log(g1)
		if(ans1 > MAX):
			ans1 = ans1 + math.log(g2)
	print "ans0 ", ans0
	print "ans1 ", ans1
	if (ans0 > ans1):
		myPrediction[i] = int(-1)
	elif(ans1 > ans0): 
		myPrediction[i] = int(1)
	elif(ans1 == ans0):
		print "Max - ", maxPrior
		myPrediction[i] = maxPrior
trueAns = testData[:, -1]
correctValues = 0
for i in range(totalValues):
	if (myPrediction[i] == trueAns[i]):
		correctValues = correctValues + 1

correctValues = float(correctValues)
totalValues = float(totalValues)
accuracy = correctValues/totalValues * 100
print accuracy


