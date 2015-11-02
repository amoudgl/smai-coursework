import numpy as np
from sklearn.decomposition import PCA

def reduceDimensionPCA(mat, k):
	print mat.shape
	pca = PCA(n_components = k)
	data = pca.fit_transform(mat)
	newData = np.zeros((len(mat), k + 1))
	print newData.shape
	print data.shape
	return newData