from surprise import SVD
from surprise import Dataset # pretty sure we don't need
from surprise import Reader
from surprise import accuracy
from surprise.model_selection import train_test_split
from surprise.model_selection import cross_validate
from surprise.model_selection import PredefinedKFold
from surprise.model_selection import KFold
from basic_viz import load_data
import numpy as np
import os

def get_err(U, V, a, b, Y, reg=0.0):
    """
    Takes as input a matrix Y of triples (i, j, Y_ij) where i is the index of a user,
    j is the index of a movie, and Y_ij is user i's rating of movie j and
    user/movie matrices U and V.

    Returns the mean regularized squared-error of predictions made by
    estimating Y_{ij} as the dot product of the ith row of U and the jth column of V^T.
    """
    
    reg_term = 0.5 * reg * (np.sum(np.array(U) ** 2) + np.sum(np.array(V) ** 2))
    err_term = 0.5 * np.sum((Yij - np.asscalar(U[i-1] * V.T[:, j-1]) - a[i-1] - b[j-1]) ** 2 
        for i, j, Yij in Y)
    return (reg_term + err_term) / len(Y)


# Train the reccomender model using Suprise's SVD and return the matrices 
# U and V in the SVD along with the errors and the biases
def train_model(trainFilePath, testFilePath, K, eta, reg, Y_train, Y_test):
    print('Surprise! V.2')

    # Get the training and testing data from the file
    # NOTE: Had to concatenate both files because if we didn't we would get
    # that surprise would  not recognize that there's N movies, because not all
    # movies are rated in the train set
    file_pathTrain = os.path.expanduser('./data/trainTest1.txt')
    reader = Reader(sep='\t')
    dataLocal = Dataset.load_from_file(file_pathTrain, reader=reader)
    
    alg = SVD() # using the SVD algorithm
    
    alg.n_factors = K # n_factors is the number of factors, K in matrices
    alg.n_epochs = 100
    alg.lr_all = eta # set the learning rate
    alg.reg_all = reg # the reglarization constant

    trainset = dataLocal.build_full_trainset()
    alg.fit(trainset)

    print('number of users:', trainset.n_users)
    print('number of movies:', trainset.n_items)
    print('number of ratings:', trainset.n_ratings)

    #testset = trainset.build_testset()
    #predictionsTrain = alg.test(testset) # testing on data in
    #errorTrain = accuracy.rmse(predictionsTrain, verbose=True)
    
    print('U matrix', type(alg.pu))
    print('V matrix', type(alg.qi))
    print('bu array', type(alg.bu))
    print('bi array', type(alg.bi))

    U = np.asmatrix(alg.pu)
    V = np.asmatrix(alg.qi)
    
    print('U shape', U.shape)
    print('V shape', V.shape)
    
    errorTrain = get_err(U, V, alg.bu, alg.bi, Y_train, reg)
    errorTest = get_err(U, V, alg.bu, alg.bi, Y_test, reg)

    # to make a test test for the testing data, we need to go through a convoluted
    # process of making it first a .build_full_trainset() -> .build_testset()
    #trainsetForTest = dataLocalTrain.build_full_trainset()
    #testsetForTest = trainsetForTest.build_testset()
    #testsetForTest = dataLocalTrain.build_testset()
    #predictionsTest = alg.test(testsetForTest)
    #errorTest = accuracy.rmse(predictionsTest, verbose=True)

    '''
    NOTE: Have to call alg.fit() for this to work, this is what we're returning:
     alg.pu is the numpy array of user factors U?
     alg.qi is the numpy array of item factors V?
     alg.bu is the numpy array of user biases
     alg.bi is the numpy array of item biases
    '''
    
    print('pu array:', alg.pu) 
    print('qi array:', alg.qi)


    return alg.pu, alg.qi, alg.bu, alg.bi, errorTrain, errorTest

