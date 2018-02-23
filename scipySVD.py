import numpy as np
import scipy
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import svds, eigs

def get_err(U, V, Y, reg=0.0):
    """
    Takes as input a matrix Y of triples (i, j, Y_ij) where i is the index of a user,
    j is the index of a movie, and Y_ij is user i's rating of movie j and
    user/movie matrices U and V.

    Returns the mean regularized squared-error of predictions made by
    estimating Y_{ij} as the dot product of the ith row of U and the jth column of V^T.
    """
    
    reg_term = 0.5 * reg * (np.sum(np.array(U) ** 2) + np.sum(np.array(V) ** 2))
    err_term = 0.5 * np.sum((Yij - np.asscalar(U[i-1] * V.T[:, j-1])) ** 2 
        for i, j, Yij in Y)
    return (reg_term + err_term) / len(Y)

# train the model using scipy.linalg svds
def train_model(M, N, K, eta, reg, Y, eps=0.0001, max_epochs=2):

    row_indexes = np.array(Y[:,0])
    col_indexes = np.array(Y[:,1])
    data = np.array(Y[:,2])


    # have to do -1 on the row/column indexes since the data starts counting at 1
    A = csc_matrix((data, (row_indexes-1, col_indexes-1)), shape=(M, N), dtype=float)

    U, S, V = svds(A, k=K) # get U and V from our svd
    
    U = np.matrix(U) # convert into a numpy matrix
    V = np.matrix(V)
    
    ein = get_err(U, V.T, Y, reg)

    return U, V.T, ein
