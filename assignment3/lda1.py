import numpy as np
from numpy import linalg as LA
import math
from sklearn import preprocessing

def gaussian(x, v, M, V):
	#G = (1/math.sqrt(2 * math.pi * V[x])) * (math.exp(-(math.pow((v - M[x]),2)/(2 * V[x]))))
#	G = (math.sqrt((2 * V[x])/math.pi)) * ((math.exp(-(math.pow((v - M[x]),2)/(2 * V[x]))))- (math.exp(-(math.pow((mins[x] - M[x]),2)/(2 * V[x]))))) 
	G = (math.exp(-(math.pow((v - M[x]),2)/(2 * V[x]))))
	return G

def findVariance(C):
	#d = C.shape[1]
	#n = C.shape[0]
	#V = np.zeros(d)
	#for i in xrange(d):
	#	V[i] = np.var(C[:, i])
	#return V
	return np.var(C, axis = 0)

def findMean(C):
	#d = C.shape[1]
	#n = C.shape[0]
	#M = np.zeros(d)
	#for i in xrange(d):
	#	M[i] = np.mean(C[:, i])
	#return M
	return np.mean(C, axis = 0)

def ldaTransform(X):
	y = X[:, -1]
	X = X[:, :-1]
	mean_vectors = []
	mean_vectors.append(np.mean(X[y==-1])[:-1], axis=0)
	mean_vectors.append(np.mean(X[y==1])[:-1], axis=0)
	S_W = np.zeros((10000,10000))
	for cl,mv in zip(range(1,3), mean_vectors):
	    class_sc_mat = np.zeros((10000,10000))                  # scatter matrix for every class
	    for row in X[y == cl]:
	        row, mv = row.reshape(10000,1), mv.reshape(10000,1) # make column vectors
	        class_sc_mat += (row-mv).dot((row-mv).T)
	    S_W += class_sc_mat                             # sum class scatter matrices
	overall_mean = np.mean(X, axis=0)

	S_B = np.zeros((10000,10000))
	for i,mean_vec in enumerate(mean_vectors):
	    n = X[y==i+1,:].shape[0]
	    mean_vec = mean_vec.reshape(10000,1) # make column vector
	    overall_mean = overall_mean.reshape(10000,1) # make column vector
	    S_B += n * (mean_vec - overall_mean).dot((mean_vec - overall_mean).T)
	eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))

	# Make a list of (eigenvalue, eigenvector) tuples
	eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
	for i in range(len(eig_vals)):
	    eigvec_sc = eig_vecs[:,i].reshape(10000,1)

	# Sort the (eigenvalue, eigenvector) tuples from high to low
	eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)
	W = eig_pairs[0][1].reshape(10000,1)
	X_lda = X.dot(W)
	X_lda = addLabels(X, y)
	return X_lda

def addLabels(data, trainLabels):
	b = np.zeros((data.shape[0], data.shape[1] + 1))
	b[:, :-1] = data
	b[:, -1] = trainLabels
	return b

file = open('arcene_train.data.txt')
featureVectors = []
for line in file :
	vector = line.strip().lower().split(' ')
	featureVectors.append(vector)
data = np.array(featureVectors)
data = data.astype(float)
preprocessing.scale(data, axis=0, with_mean=True, with_std=True, copy=False)
file = open('arcene_valid.data.txt')
featureVectors = []
for line in file :
	vector = line.strip().lower().split(' ')
	featureVectors.append(vector)
testData = np.array(featureVectors)
testData = data.astype(float)
preprocessing.scale(testData, axis=0, with_mean=True, with_std=True, copy=False)
file = open('arcene_train.labels.txt')
trainLabels = []
for line in file :
	vector = line
	trainLabels.append(vector)
trainLabels = np.array(trainLabels)
trainLabels = trainLabels.astype(int)
file = open('arcene_valid.labels.txt')
testLabels = []
for line in file :
	vector = line
	testLabels.append(vector)
testLabels = np.array(testLabels)
testLabels = testLabels.astype(int)
#LDA
trainData = addLabels(data, trainLabels)
trainData = ldaTransform(trainData)
testData = addLabels(testData, testLabels)
testData = ldaTransform(testData)

C0 = trainData[trainData[:, -1] == -1]
C1 = trainData[trainData[:, -1] == 1]
V0 = findVariance(C0[:, :-1])
V1 = findVariance(C1[:, :-1])
M0 = findMean(C0[:, :-1])
M1 = findMean(C1[:, :-1])
pr0 = float(C0.shape[0])/float(trainData.shape[0])
pr1 = float(C1.shape[0])/float(trainData.shape[0])

#mins = findMinimums(trainData)
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
	else: 
		myPrediction[i] = int(1)

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
print accuracy
#Accuracy is less : Random probes added in data, features may not be gaussian, PCA reduces dimensions

#################################################


