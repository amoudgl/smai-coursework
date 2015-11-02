import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
import random
import copy
import math

def encode(vector):
	le = preprocessing.LabelEncoder()
	le.fit(vector)
	vector = le.transform(vector)
	return vector

def encodeAll(mat):
	categoricalIndices = [1,2,3,4,5,6,7,8,9,14]
	for i in categoricalIndices:
		mat[:,i] = encode(mat[:,i])
	return mat.astype(np.float)

def deleteFloatData(mat):
	idx_OUT_columns = [10,15,16,17,18,19]
	idx_IN_columns = [i for i in xrange(np.shape(mat)[1]) if i not in idx_OUT_columns]
	extractedData = mat[:,idx_IN_columns]
	return extractedData

def findProbabilityMatrix0(featureMatrix, C0):
	l = C0.shape[1]
	p0 = copy.deepcopy(featureMatrix)
	for i in xrange(l - 1):
		featureVector = featureMatrix[i]
		x0 = C0[:, i].tolist()
		j = 0
		for item in featureVector:
			c0 = float(x0.count(item))
			p0[i][j] = c0/float(C0.shape[0])
			j = j + 1
	return p0

def findProbabilityMatrix1(featureMatrix, C1):
	l = C1.shape[1]
	p1 = copy.deepcopy(featureMatrix)
	for i in xrange(0, l - 1):
		featureVector = featureMatrix[i]
		x1 = C1[:, i].tolist()
		j = 0
		for item in featureVector:
			c1 = float(x1.count(item))
			p1[i][j] = c1/float(C1.shape[0])
			j = j + 1
	return p1

def findFeatureMatrix(mat):
	l = mat.shape[1]
	featureMatrix = []
	for i in xrange(0, l - 1):
		featureVector = np.unique(mat[:, i])
		featureMatrix.append(featureVector.tolist())
	return featureMatrix

#Training phase
file = open('bank-additional.csv')
featureVectors = []
i = 0
for line in file :
	vector = line.strip().lower().split(';')
	if i != 0:		
		if vector[-1] == '"no"':
			vector[-1] = 0
		else:
			vector[-1] = 1
		featureVectors.append(vector)
	i = i + 1

numberOfRuns = 10
results = np.zeros([numberOfRuns])
for t in xrange(numberOfRuns):
	random.shuffle(featureVectors)
	mat = np.array(featureVectors)
	mat = encodeAll(mat)
	mat = deleteFloatData(mat)
	mat = mat.astype(int)
	N = 2500
	trainData = mat[:N, :]
	testData = mat[N:, :]
	featureMatrix = findFeatureMatrix(trainData)
	C0 = trainData[trainData[:, -1] == 0]
	C1 = trainData[trainData[:, -1] == 1]
	p0 = findProbabilityMatrix0(featureMatrix, C0)
	p1 = findProbabilityMatrix1(featureMatrix, C1)
	pr0 = float(C0.shape[0])/float(trainData.shape[0])
	pr1 = float(C1.shape[0])/float(trainData.shape[0])
	#print pr0
	#print pr1
	if (pr0 > pr1):
		maxPrior = int(0) 
	else:
		maxPrior = int(1)
	#Testing phase
	totalValues = testData.shape[0]	
	#print totalValues
	#print testData[:, 10]
	myPrediction = np.zeros([totalValues])
	for i in xrange(0, totalValues):
		sample = testData[i, :]
		sample = sample.tolist()
		ans0 = (float(pr0))
		ans1 = (float(pr1))
		count = 0;
		for j in xrange(0, len(sample) - 1):
			flag = 0 
			for k in xrange(0, len(featureMatrix[j])):
				if (sample[j] == featureMatrix[j][k]):
					if (p0[j][k] != 0):
						ans0 = ans0 * (p0[j][k])
					else: 
						ans0 = ans0 * p0[j][k]
					if (p1[j][k] != 0):
						ans1 = ans1 * (p1[j][k])
					else:
						ans1 = ans1 * p1[j][k]
				#	print "Found!"
					count = count + 1
					flag = 1
					break
			#if (flag == 0):
		#		print sample[j], " ", i + trainData.shape[0], " - ", j
		print ans0
		print ans1
		if (ans0 > ans1):
			myPrediction[i] = int(0)
		elif(ans0 < ans1): 
			myPrediction[i] = int(1)
		else:
			myPrediction[i] = maxPrior

	trueAns = testData[:, -1]
	#print len(myPrediction[myPrediction == 0])
	#print len(myPrediction[myPrediction == 1])
	#print len(trueAns[trueAns == 0])
	#print len(trueAns[trueAns == 1])
	correctValues = 0
	for i in range(totalValues):
		if (myPrediction[i] == trueAns[i]):
			correctValues = correctValues + 1
		#else:
	#		print i + trainData.shape[0]

	correctValues = float(correctValues)
	totalValues = float(totalValues)
	accuracy = correctValues/totalValues * 100
	#print correctValues
	results[t] = accuracy
	#print accuracy

meanAccuracy = np.mean(results)
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = results
plt.figure()
#plt.plot(x,y, marker='o', color='b')
plt.errorbar(x, y, np.std(results, axis = 0))
plt.show()
#stD = np.std(results)
print meanAccuracy