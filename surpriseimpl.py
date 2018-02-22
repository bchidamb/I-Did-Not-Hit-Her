from surprise import SVD
from surprise import Dataset # pretty sure we don't need
from surprise import Reader
from surprise.model_selection import cross_validate
import numpy as np
import os


def loadData_surprise(trainFile, testFile):

    Y_train = np.loadtxt(trainFile).astype(int)
    Y_test = np.loadtxt(testFile).astype(int)

    return [Y_train, Y_test]




Y_train, Y_test = loadData_surprise('data/train.txt', 'data/test.txt')
data = Dataset.load_builtin('ml-100k')

file_pathTrain = os.path.expanduser('./data/train.txt')
reader = Reader(sep='\t')
dataLocalTrain = Dataset.load_from_file(file_pathTrain, reader=reader)

file_pathTest = os.path.expanduser('./data/train.txt')
reader = Reader(sep='\t')
dataLocalTest = Dataset.load_from_file(file_pathTrain, reader=reader)


alg = SVD() # using the SVD algorithm

K = 20 # our set value of K in the matrices
alg.n_factors = K # n_factors is the number of factors
alg.fit(dataLocalTrain)

#cross_validate(alg, dataLocalTrain, measures=['RMSE', 'MAE'], cv=5, verbose=True)

print('pu array:', alg.pu)
print('qi array:', alg.qi)



