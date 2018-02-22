from surprise import SVD
from surprise import Dataset # pretty sure we don't need
from surprise import Reader
from surprise import accuracy
from surprise.model_selection import train_test_split
from surprise.model_selection import cross_validate
from basic_viz import load_data
import numpy as np
import os



def train_model(trainFilePath, testFilePath, K, eta, reg, Y_train):

    file_pathTrain = os.path.expanduser(trainFilePath)  #'./data/train.txt')
    #reader = Reader(
    reader = Reader(sep='\t')
    dataLocalTrain = Dataset.load_from_file(file_pathTrain, reader=reader)
    
    file_pathTest = os.path.expanduser(testFilePath) #'./data/train.txt')
    reader = Reader(sep='\t')
    dataLocalTest = Dataset.load_from_file(file_pathTrain, reader=reader)
    
    trainset, testset = train_test_split(dataLocalTrain, test_size=.1)
    
    alg = SVD() # using the SVD algorithm
    
    alg.n_factors = K # n_factors is the number of factors, K in matrices
    alg.n_epochs = 100
    alg.lr_all = eta # set the learning rate
    alg.reg_all = reg # the reglarization constant

    #result = cross_validate(alg, dataLocalTrain, measures=['RMSE', 'MAE'], cv=5, verbose=True)

    #rmse = sum(result['test_rmse'])/len(result['test_rmse'])
    #print('result[test_rmse] average', rmse)

    trainset = dataLocalTrain.build_full_trainset()
    alg.fit(trainset)
    testset = trainset.build_testset()
    predictions = alg.test(testset)
    error = accuracy.rmse(predictions, verbose=True)
    
    #cross_validate(alg, dataLocalTrain, measures=['RMSE', 'MAE'], cv=5, verbose=True)
    
    # NOTE: Have to call alg.fit() for this to work
    # alg.pu is the numpy array of user factors U?
    # alg.qi is the numpy array of item factors V?
    # alg.bu is the numpy array of user biases
    # alg.bi is the numpy array of item biases
    print('pu array:', alg.pu) 
    print('qi array:', alg.qi)


    return alg.pu, alg.qi, alg.bu, alg.bi, error

