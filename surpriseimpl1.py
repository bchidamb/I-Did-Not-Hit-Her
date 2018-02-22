from surprise import SVD
from surprise import Dataset # pretty sure we don't need
from surprise import Reader
from surprise import accuracy
from surprise.model_selection import train_test_split
from surprise.model_selection import cross_validate
from basic_viz import load_data
import numpy as np
import os


# Train the reccomender model using Suprise's SVD and return the matrices 
# U and V in the SVD along with the errors and the biases
def train_model(trainFilePath, testFilePath, K, eta, reg, Y_train):

    # Get the training data from the file
    file_pathTrain = os.path.expanduser(trainFilePath)  #'./data/train.txt')
    reader = Reader(sep='\t')
    dataLocalTrain = Dataset.load_from_file(file_pathTrain, reader=reader)
    
    # Get the testing data from the file
    file_pathTest = os.path.expanduser(testFilePath) #'./data/train.txt')
    reader = Reader(sep='\t')
    dataLocalTest = Dataset.load_from_file(file_pathTrain, reader=reader)
    
    alg = SVD() # using the SVD algorithm
    
    alg.n_factors = K # n_factors is the number of factors, K in matrices
    alg.n_epochs = 100
    alg.lr_all = eta # set the learning rate
    alg.reg_all = reg # the reglarization constant


    trainset = dataLocalTrain.build_full_trainset()
    alg.fit(trainset)
    print('number of users:', trainset.n_users)
    print('number of movies:', trainset.n_items)
    print('number of ratings:', trainset.n_ratings)

    testset = trainset.build_testset()
    predictionsTrain = alg.test(testset) # testing on data in
    errorTrain = accuracy.rmse(predictionsTrain, verbose=True)

    # to make a test test for the testing data, we need to go through a convoluted
    # process of making it first a .build_full_trainset() -> .build_testset()
    trainsetForTest = dataLocalTrain.build_full_trainset()
    testsetForTest = trainsetForTest.build_testset()
    #testsetForTest = dataLocalTrain.build_testset()
    predictionsTest = alg.test(testsetForTest)
    errorTest = accuracy.rmse(predictionsTest, verbose=True)

    '''
    NOTE: Have to call alg.fit() for this to work, this is what we're returning:
     alg.pu is the numpy array of user factors U?
     alg.qi is the numpy array of item factors V?
     alg.bu is the numpy array of user biases
     alg.bi is the numpy array of item biases
    '''
    
    #print('pu array:', alg.pu) 
    #print('qi array:', alg.qi)


    return alg.pu, alg.qi, alg.bu, alg.bi, errorTrain, errorTest


    #trainset, testset = train_test_split(dataLocalTrain, test_size=.1)
    #result = cross_validate(alg, dataLocalTrain, measures=['RMSE', 'MAE'], cv=5, verbose=True)
    #rmse = sum(result['test_rmse'])/len(result['test_rmse'])
    #print('result[test_rmse] average', rmse)    
    #cross_validate(alg, dataLocalTrain, measures=['RMSE', 'MAE'], cv=5, verbose=True)

