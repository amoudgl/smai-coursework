import numpy as np
from numpy import linalg as LA
import math
from sklearn import preprocessing
from sklearn import svm, preprocessing
from sklearn.metrics import classification_report as cr

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


def train(X, y):
	clf = svm.SVC(kernel='linear', C = 1.0, max_iter = -1)
	clf.fit(X, y)
	return clf

def predict(model, vector):
	return model.predict(vector)

def classify(model, featureVectors):
	true = 0
	total = 0
	z = []
	for feature in featureVectors:
		if feature[-1] == predict(model, feature[:-1]):
			true += 1
		z = z + predict(model, feature[:-1]).astype(np.int).tolist()
		total += 1
	data = featureVectors[:,-1].flatten()
	data = data.astype(np.int).tolist()
	print z
	print cr(data, z)
	print "Accuracy : ",
	print (true * 100) / total


file = open('arcene_train.data.txt')
data = getDataMatrix(file, 1)
file = open('arcene_train.labels.txt')
trainLabels = getDataMatrix(file, 0)
file = open('arcene_valid.data.txt')
testData = getDataMatrix(file, 1)
file = open('arcene_valid.labels.txt')
testLabels = getDataMatrix(file, 0)

trainData = addLabels(data, trainLabels)
ev = ldaTransform(trainData)
trainData = trainData[:, :-1]
trainData = project(trainData, ev)
testData = project(testData, ev)
model = train(trainData, trainLabels)
testData = addLabels(testData, testLabels)
classify(model, testData)


