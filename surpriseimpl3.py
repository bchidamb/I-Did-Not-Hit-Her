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
    print('Surprise! V.3')

    # Get the training and testing data from the file
    # NOTE: Had to concatenate both files because if we didn't we would get
    # that surprise would  not recognize that there's N movies, because not all
    # movies are rated in the train set
    file_pathTrain = os.path.expanduser('./data/trainTest1.txt')
    reader = Reader(sep='\t')
    dataLocal = Dataset.load_from_file(file_pathTrain, reader=reader)
    
    alg = SVD() # use the SVD algorithm
    
    # Set parameters:
    alg.n_factors = K # n_factors is the number of factors, K in matrices
    alg.n_epochs = 100
    alg.lr_all = eta # set the learning rate
    alg.reg_all = reg # the reglarization constant

    trainset = dataLocal.build_full_trainset() # use all data to train
    alg.fit(trainset) # train on the trainset
    testset = trainset.build_testset()
    prediction = alg.test(testset)
    acc = accuracy.rmse(prediction, verbose=True)

    '''
    NOTE: Have to call alg.fit() for this to work, this is what we're returning:
     alg.pu is the numpy array of user factors U
     alg.qi is the numpy array of item(movie) factors V
     alg.bu is the numpy array of user biases
     alg.bi is the numpy array of item(movie) biases
    '''

    U = np.asmatrix(alg.pu) # convert to numpy matrices for error function
    V = np.asmatrix(alg.qi)

    # Sanity checks:
    #print('number of users:', trainset.n_users, 'M:', M)
    #print('number of movies:', trainset.n_items, 'N:', N)
    #print('number of ratings:', trainset.n_ratings)

    #print('U shape, MxK', U.shape)
    #print('V shape, KxN', V.shape)
    
    # Get the training and test error using the same error function we used
    # with our other models
    errorTrain = get_err(U, V, alg.bu, alg.bi, Y_train, reg)
    errorTest = get_err(U, V, alg.bu, alg.bi, Y_test, reg)

    return U, V, alg.bu, alg.bi, errorTrain, errorTest

