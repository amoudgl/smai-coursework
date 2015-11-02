import numpy as np
from numpy import linalg as LA
import math
from sklearn import preprocessing

def gaussian(v, M, V):
	G = (math.exp(-(math.pow((v - M),2)/(2 * V))))
	return G

def findVariance(C):
	return np.var(C, axis = 0)

def findMean(C):
	return np.mean(C, axis = 0)

def mergeData(trainData, testData):
	x = np.zeros((trainData.shape[0] + testData.shape[0], trainData.shape[1]))
	x[:trainData.shape[0], :] = trainData
	x[trainData.shape[0]:, :] = testData
	return x

def ldaTransform(data):
	C0 = data[data[:, -1] == -1]
	C1 = data[data[:, -1] == 1]
	C0 = C0[:, :-1]
	C1 = C1[:, :-1]	
	S0 = np.cov(np.transpose(C0))
	S1 = np.cov(np.transpose(C1))
	SW = S0 + S1
	Mu0 = np.mean(C0, axis = 0)
	Mu1 = np.mean(C1, axis = 0)
	Mu = np.mean(data, axis = 0)
	Mu = Mu[:-1]
	Mu = np.matrix(Mu)
	Mu0 = np.matrix(Mu0)
	Mu1 = np.matrix(Mu1)
	SB = C0.shape[0] * np.transpose(Mu0 - Mu) * (Mu0 - Mu) + C1.shape[0] * np.transpose(Mu1 - Mu) * (Mu1 - Mu)
	#t = Mu0 - Mu1
	#t = np.matrix(t)
	#SB = np.transpose(t) * t
	Swin = LA.pinv(SW) #costly 
	Swin = np.matrix(Swin)
	SwinSB = Swin * SB #costly 
	e, v = LA.eig(SwinSB) #costly 
	s = np.argsort(e)[::-1]
	v = np.array(v)
	ev = np.zeros(v.shape)
	for i in xrange(e.shape[0]):
		ev[:, i] = v[:, s[i]]
	w = ev[:, 0]
	w = np.matrix(w)
	return w

def project(data, w):
	data = np.matrix(data)
	data = np.transpose(data)
	newData = w * data
	newData = np.transpose(newData)
	newData = np.array(newData)
	return newData

def addLabels(data, trainLabels):
	b = np.zeros((data.shape[0], data.shape[1] + 1))
	b[:, :-1] = data
	b[:, -1] = trainLabels
	return b

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

#LDA
trainData = addLabels(data, trainLabels)
ev = ldaTransform(trainData)
trainData = trainData[:, :-1]
trainData = project(trainData, ev)
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
L = math.pow(10, -323)
MAX = -math.pow(10, 300)

#Testing phase
totalValues = testData.shape[0]	
myPrediction = np.zeros([totalValues])
j = 0
for i in xrange(0, totalValues):
	sample = testData[i, :]
	sample = sample.tolist()
	ans0 = math.log(float(pr0))
	ans1 = math.log(float(pr1))
	count = 0;
	g1 = gaussian(sample[j], M0, V0)
	g2 = gaussian(sample[j], M1, V1)
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
	else: 
		myPrediction[i] = int(1)

trueAns = testData[:, -1]
correctValues = 0
for i in range(totalValues):
	if (myPrediction[i] == trueAns[i]):
		correctValues = correctValues + 1
correctValues = float(correctValues)
totalValues = float(totalValues)
accuracy = correctValues/totalValues * 100
print accuracy

